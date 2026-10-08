"""Stateful first-come-first-served policy for kinematic baselines.

The policy serializes conflicting lane-change requests, keeps grants active
until the requested one-lane maneuver completes, and never asks a non-granted
vehicle to decelerate. A granted vehicle may adjust its own speed to create a
safe merge gap before beginning the lane change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from marl.highway_env import (
    ACCELERATE,
    KEEP_LANE,
    LANE_LEFT,
    LANE_RIGHT,
    YIELD,
    VectorHighwayEnv,
)


LANE_ACTIONS = (LANE_LEFT, LANE_RIGHT)


@dataclass(frozen=True)
class ManeuverRequest:
    """One requested single-lane transition."""

    agent: int
    source_lane: int
    target_lane: int
    arrival_order: int
    emergency: bool = False


class StatefulFCFSPolicy:
    """Deterministic two-agent FCFS maneuver arbiter and executor."""

    def __init__(
        self,
        *,
        priority_order: Iterable[int] = (0, 1),
        emergency_agents: Iterable[int] = (),
        merge_clearance_m: float = 6.25,
    ) -> None:
        priority = tuple(int(agent) for agent in priority_order)
        if sorted(priority) != [0, 1]:
            raise ValueError("priority_order must contain agent indices 0 and 1")
        if merge_clearance_m <= 0.0:
            raise ValueError("merge_clearance_m must be positive")
        self.priority_order = priority
        self.priority_rank = {agent: rank for rank, agent in enumerate(priority)}
        self.emergency_agents = {int(agent) for agent in emergency_agents}
        if not self.emergency_agents.issubset({0, 1}):
            raise ValueError("emergency agent indices must be 0 or 1")
        self.merge_clearance_m = float(merge_clearance_m)
        self.pending: dict[int, ManeuverRequest] = {}
        self.active: dict[int, ManeuverRequest] = {}
        self.arrival_counter = 0
        self.grant_history: list[tuple[int, int, int]] = []

    def reset(self) -> None:
        """Clear session-local queue and grant state."""
        self.pending.clear()
        self.active.clear()
        self.arrival_counter = 0
        self.grant_history.clear()

    def actions(self, env: VectorHighwayEnv) -> np.ndarray:
        """Return one action per agent for a single-environment batch."""
        if env.num_envs != 1 or env.cfg.num_agents != 2:
            raise ValueError("StatefulFCFSPolicy requires one two-agent environment")

        self._close_completed_grants(env)
        self._refresh_pending_requests(env)
        self._issue_compatible_grants()

        # KEEP_LANE is the environment's lane-and-flow hold controller. Only a
        # granted vehicle may receive a lane or longitudinal gap-opening action.
        result = np.full(2, KEEP_LANE, dtype=np.int64)
        for agent, request in sorted(self.active.items()):
            result[agent] = self._granted_action(env, request)
        return result

    def _close_completed_grants(self, env: VectorHighwayEnv) -> None:
        for agent, request in list(self.active.items()):
            lane_error = abs(float(env.lane_pos[0, agent]) - request.target_lane)
            changing = (
                abs(
                    float(
                        env.lane_destination[0, agent]
                        - env.lane_pos[0, agent]
                    )
                )
                > 1e-4
            )
            if lane_error < 0.08 and not changing:
                del self.active[agent]
                self.pending.pop(agent, None)

    def _refresh_pending_requests(self, env: VectorHighwayEnv) -> None:
        for agent in self.priority_order:
            if agent in self.active:
                continue
            request_target = self._next_target_lane(env, agent)
            existing = self.pending.get(agent)
            if request_target is None:
                self.pending.pop(agent, None)
                continue
            source_lane = int(round(float(env.lane_pos[0, agent])))
            if (
                existing is not None
                and existing.source_lane == source_lane
                and existing.target_lane == request_target
            ):
                continue
            self.pending[agent] = ManeuverRequest(
                agent=agent,
                source_lane=source_lane,
                target_lane=request_target,
                arrival_order=self.arrival_counter,
                emergency=agent in self.emergency_agents,
            )
            self.arrival_counter += 1

    def _issue_compatible_grants(self) -> None:
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
            if any(self._requests_conflict(request, other) for other in selected):
                continue
            self.active[request.agent] = request
            selected.append(request)
            self.grant_history.append(
                (request.agent, request.source_lane, request.target_lane)
            )

    @staticmethod
    def _requests_conflict(
        first: ManeuverRequest,
        second: ManeuverRequest,
    ) -> bool:
        same_target = first.target_lane == second.target_lane
        lane_swap = (
            first.target_lane == second.source_lane
            and second.target_lane == first.source_lane
        )
        return same_target or lane_swap

    @staticmethod
    def _next_target_lane(
        env: VectorHighwayEnv,
        agent: int,
    ) -> int | None:
        changing = (
            abs(
                float(
                    env.lane_destination[0, agent]
                    - env.lane_pos[0, agent]
                )
            )
            > 1e-4
        )
        if changing:
            return None
        current_lane = int(round(float(env.lane_pos[0, agent])))
        goal_lane = int(env.goal_lane[0, agent])
        if current_lane == goal_lane:
            return None
        direction = 1 if goal_lane > current_lane else -1
        return current_lane + direction

    def _granted_action(
        self,
        env: VectorHighwayEnv,
        request: ManeuverRequest,
    ) -> int:
        agent = request.agent
        other = 1 - agent
        lane_error = request.target_lane - float(env.lane_pos[0, agent])
        changing = (
            abs(
                float(
                    env.lane_destination[0, agent]
                    - env.lane_pos[0, agent]
                )
            )
            > 1e-4
        )

        other_occupies_target = (
            abs(float(env.lane_pos[0, other]) - request.target_lane) < 0.75
        )
        if other_occupies_target and not self._merge_gap_is_safe(env, agent, other):
            return self._gap_opening_action(env, agent, other)
        if changing:
            return KEEP_LANE
        if abs(lane_error) < 0.08:
            return KEEP_LANE
        return LANE_RIGHT if lane_error > 0.0 else LANE_LEFT

    def _merge_gap_is_safe(
        self,
        env: VectorHighwayEnv,
        agent: int,
        other: int,
    ) -> bool:
        longitudinal_distance = abs(
            float(env.x[0, agent] - env.x[0, other])
        )
        bumper_clearance = max(
            longitudinal_distance - env.cfg.vehicle_length_m,
            0.0,
        )
        if bumper_clearance < self.merge_clearance_m:
            return False

        if env.x[0, agent] < env.x[0, other]:
            rear, front = agent, other
        else:
            rear, front = other, agent
        closing_speed = float(env.speed[0, rear] - env.speed[0, front])
        if closing_speed <= 0.05:
            return True
        predicted_clearance = bumper_clearance - closing_speed * env.cfg.dt_s
        if predicted_clearance < self.merge_clearance_m:
            return False
        ttc_s = bumper_clearance / closing_speed
        return ttc_s >= env.cfg.ttc_threshold_s

    @staticmethod
    def _gap_opening_action(
        env: VectorHighwayEnv,
        agent: int,
        other: int,
    ) -> int:
        # The granted vehicle changes only its own speed. The non-granted
        # vehicle remains under KEEP_LANE and is never direction-agnostically
        # told to brake.
        if env.x[0, agent] <= env.x[0, other]:
            return YIELD
        return ACCELERATE
