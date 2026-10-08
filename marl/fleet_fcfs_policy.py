"""Deterministic stateful FCFS arbitration for variable-size fleets."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .fcfs_policy import ManeuverRequest
from .fleet_highway_env import FleetHighwayEnv
from .highway_env import ACCELERATE, KEEP_LANE, LANE_LEFT, LANE_RIGHT, YIELD


class FleetFCFSPolicy:
    """Serialize only nearby same-target or reciprocal lane-change requests."""

    def __init__(
        self,
        vehicle_count: int,
        *,
        priority_order: Iterable[int] | None = None,
        emergency_agents: Iterable[int] = (),
        conflict_distance_m: float = 50.0,
        ttc_threshold_s: float = 4.0,
        merge_clearance_m: float = 6.25,
    ) -> None:
        self.vehicle_count = int(vehicle_count)
        priority = tuple(
            range(self.vehicle_count)
            if priority_order is None
            else (int(value) for value in priority_order)
        )
        if sorted(priority) != list(range(self.vehicle_count)):
            raise ValueError("priority_order must contain every vehicle index once")
        self.priority_order = priority
        self.priority_rank = {agent: rank for rank, agent in enumerate(priority)}
        self.emergency_agents = {int(agent) for agent in emergency_agents}
        if not self.emergency_agents.issubset(set(range(self.vehicle_count))):
            raise ValueError("emergency agent index outside fleet")
        self.conflict_distance_m = float(conflict_distance_m)
        self.ttc_threshold_s = float(ttc_threshold_s)
        self.merge_clearance_m = float(merge_clearance_m)
        self.pending: dict[int, ManeuverRequest] = {}
        self.active: dict[int, ManeuverRequest] = {}
        self.arrival_counter = 0
        self.grant_history: list[tuple[int, int, int]] = []
        self.last_events: list[dict[str, object]] = []

    def actions(self, env: FleetHighwayEnv) -> np.ndarray:
        if env.cfg.num_vehicles != self.vehicle_count:
            raise ValueError("environment fleet size changed")
        self.last_events = []
        self._close_completed_grants(env)
        self._refresh_pending_requests(env)
        self._issue_compatible_grants(env)
        actions = np.full(self.vehicle_count, KEEP_LANE, dtype=np.int64)
        for agent, request in sorted(self.active.items()):
            actions[agent] = self._granted_action(env, request)
        return actions

    def _close_completed_grants(self, env: FleetHighwayEnv) -> None:
        for agent, request in list(self.active.items()):
            lane_error = abs(float(env.lane_pos[agent]) - request.target_lane)
            changing = abs(
                float(env.lane_destination[agent] - env.lane_pos[agent])
            ) > 1e-4
            if lane_error < 0.08 and not changing:
                del self.active[agent]
                self.pending.pop(agent, None)
                self.last_events.append(
                    {
                        "event": "clear",
                        "agent": agent,
                        "target_lane": request.target_lane,
                    }
                )

    def _refresh_pending_requests(self, env: FleetHighwayEnv) -> None:
        for agent in self.priority_order:
            if agent in self.active:
                continue
            target = self._next_target_lane(env, agent)
            existing = self.pending.get(agent)
            if target is None:
                self.pending.pop(agent, None)
                continue
            source = env.current_lane(agent)
            if (
                existing is not None
                and existing.source_lane == source
                and existing.target_lane == target
            ):
                continue
            request = ManeuverRequest(
                agent=agent,
                source_lane=source,
                target_lane=target,
                arrival_order=self.arrival_counter,
                emergency=agent in self.emergency_agents,
            )
            self.pending[agent] = request
            self.arrival_counter += 1
            self.last_events.append(
                {
                    "event": "request",
                    "agent": agent,
                    "source_lane": source,
                    "target_lane": target,
                    "arrival_order": request.arrival_order,
                    "emergency": request.emergency,
                }
            )

    def _issue_compatible_grants(self, env: FleetHighwayEnv) -> None:
        ordered = sorted(
            self.pending.values(),
            key=lambda request: (
                0 if request.emergency else 1,
                request.arrival_order,
                self.priority_rank[request.agent],
            ),
        )
        selected = list(self.active.values())
        for request in ordered:
            if request.agent in self.active:
                continue
            blocker = next(
                (
                    other
                    for other in selected
                    if self._requests_conflict(env, request, other)
                ),
                None,
            )
            if blocker is not None:
                self.last_events.append(
                    {
                        "event": "nack",
                        "agent": request.agent,
                        "conflicting_agent": blocker.agent,
                        "rule": "nearby_same_target_or_swap",
                    }
                )
                continue
            self.active[request.agent] = request
            selected.append(request)
            self.grant_history.append(
                (request.agent, request.source_lane, request.target_lane)
            )
            self.last_events.append(
                {
                    "event": "ack",
                    "agent": request.agent,
                    "target_lane": request.target_lane,
                    "rule": "emergency_then_arrival_order",
                }
            )

    def _requests_conflict(
        self,
        env: FleetHighwayEnv,
        first: ManeuverRequest,
        second: ManeuverRequest,
    ) -> bool:
        maneuver_conflict = first.target_lane == second.target_lane or (
            first.target_lane == second.source_lane
            and second.target_lane == first.source_lane
        )
        if not maneuver_conflict:
            return False
        dx = abs(float(env.x[first.agent] - env.x[second.agent]))
        if dx <= self.conflict_distance_m:
            return True
        return self._pair_ttc(env, first.agent, second.agent) < self.ttc_threshold_s

    @staticmethod
    def _pair_ttc(env: FleetHighwayEnv, first: int, second: int) -> float:
        if env.x[first] <= env.x[second]:
            rear, front = first, second
        else:
            rear, front = second, first
        closing = float(env.speed[rear] - env.speed[front])
        if closing <= 0.05:
            return float("inf")
        clearance = max(
            abs(float(env.x[first] - env.x[second])) - env.cfg.vehicle_length_m,
            0.0,
        )
        return clearance / closing

    @staticmethod
    def _next_target_lane(env: FleetHighwayEnv, agent: int) -> int | None:
        if not env.goal_active[agent] or env.completed[agent]:
            return None
        changing = abs(
            float(env.lane_destination[agent] - env.lane_pos[agent])
        ) > 1e-4
        if changing:
            return None
        current = env.current_lane(agent)
        goal = int(env.goal_lane[agent])
        if current == goal:
            return None
        return current + (1 if goal > current else -1)

    def _granted_action(
        self,
        env: FleetHighwayEnv,
        request: ManeuverRequest,
    ) -> int:
        agent = request.agent
        threats = [
            other
            for other in range(self.vehicle_count)
            if other != agent
            and abs(float(env.lane_pos[other]) - request.target_lane) < 0.75
            and not self._merge_gap_is_safe(env, agent, other)
        ]
        if threats:
            other = min(threats, key=lambda index: abs(float(env.x[index] - env.x[agent])))
            return YIELD if env.x[agent] <= env.x[other] else ACCELERATE
        changing = abs(
            float(env.lane_destination[agent] - env.lane_pos[agent])
        ) > 1e-4
        if changing:
            return KEEP_LANE
        lane_error = request.target_lane - float(env.lane_pos[agent])
        if abs(lane_error) < 0.08:
            return KEEP_LANE
        return LANE_RIGHT if lane_error > 0.0 else LANE_LEFT

    def _merge_gap_is_safe(
        self,
        env: FleetHighwayEnv,
        agent: int,
        other: int,
    ) -> bool:
        clearance = max(
            abs(float(env.x[agent] - env.x[other])) - env.cfg.vehicle_length_m,
            0.0,
        )
        if clearance < self.merge_clearance_m:
            return False
        if env.x[agent] <= env.x[other]:
            rear, front = agent, other
        else:
            rear, front = other, agent
        closing = float(env.speed[rear] - env.speed[front])
        if closing <= 0.05:
            return True
        predicted = clearance - closing * env.cfg.dt_s
        return (
            predicted >= self.merge_clearance_m
            and clearance / closing >= self.ttc_threshold_s
        )
