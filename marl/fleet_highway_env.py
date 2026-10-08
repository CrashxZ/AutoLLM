"""Variable-size kinematic highway environment for coordination feasibility.

This environment is deliberately separate from :mod:`marl.highway_env` so the
frozen two-agent MAPPO baseline and its observation contract remain unchanged.
It models the same lane-change and longitudinal dynamics for 2--32 vehicles and
exposes pairwise safety measurements needed by the MIND-CAV scaling study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .highway_env import ACCELERATE, ACTION_NAMES, KEEP_LANE, LANE_LEFT, LANE_RIGHT, YIELD


@dataclass(frozen=True)
class FleetHighwayConfig:
    num_vehicles: int = 4
    num_lanes: int = 4
    dt_s: float = 0.2
    lane_width_m: float = 3.5
    lane_change_duration_s: float = 3.0
    flow_speed_kmh: float = 50.0
    max_speed_kmh: float = 90.0
    accel_mps2: float = 2.0
    yield_decel_mps2: float = 3.0
    vehicle_length_m: float = 4.7
    vehicle_width_m: float = 1.9
    safe_gap_m: float = 5.0
    localization_error_m: float = 0.25
    model_error_margin_m: float = 0.75
    ttc_threshold_s: float = 4.0
    route_length_m: float = 650.0
    exit_position_m: float = 500.0
    max_steps: int = 300
    local_safety_shield: bool = True

    @property
    def robust_clearance_m(self) -> float:
        return (
            self.safe_gap_m
            + 2.0 * self.localization_error_m
            + self.model_error_margin_m
        )


class FleetHighwayEnv:
    """One deterministic episode with a variable-size fleet."""

    def __init__(
        self,
        config: Optional[FleetHighwayConfig] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = config or FleetHighwayConfig()
        if self.cfg.num_vehicles < 2:
            raise ValueError("num_vehicles must be at least two")
        self.rng = np.random.default_rng(seed)
        count = self.cfg.num_vehicles
        self.vehicle_ids = np.arange(1000, 1000 + count, dtype=np.int64)
        self.x = np.zeros(count, dtype=np.float32)
        self.speed = np.zeros(count, dtype=np.float32)
        self.lane_pos = np.zeros(count, dtype=np.float32)
        self.lane_destination = np.zeros(count, dtype=np.float32)
        self.goal_lane = np.zeros(count, dtype=np.int64)
        self.goal_is_exit = np.zeros(count, dtype=bool)
        self.goal_active = np.zeros(count, dtype=bool)
        self.completed = np.zeros(count, dtype=bool)
        self.collision = False
        self.steps = 0
        self.scenario_id = 1

    def reset(self, scenario_id: int = 1) -> Dict[str, np.ndarray]:
        if scenario_id not in {1, 2, 3}:
            raise ValueError("scenario_id must be 1, 2, or 3")
        self.scenario_id = scenario_id
        count = self.cfg.num_vehicles

        # Lane 0 is leftmost and lane 3 is rightmost. Additional vehicles are
        # deterministic background traffic distributed behind/ahead of the two
        # focal vehicles, with seed-controlled jitter.
        self.lane_pos[:] = np.arange(count) % self.cfg.num_lanes
        self.lane_pos[0] = 0.0
        self.lane_pos[1] = 1.0
        self.lane_destination[:] = self.lane_pos
        self.goal_lane[:] = np.rint(self.lane_pos).astype(np.int64)
        self.goal_lane[0] = self.cfg.num_lanes - 1
        self.goal_lane[1] = 0 if scenario_id == 2 else 1
        self.goal_active[:] = False
        self.goal_active[0] = True
        self.goal_active[1] = scenario_id == 2
        self.goal_is_exit[:] = False
        self.goal_is_exit[0] = scenario_id == 3

        focal = float(self.rng.uniform(0.0, 8.0))
        self.x[0] = focal
        self.x[1] = focal + float(self.rng.uniform(-8.0, 8.0))
        for index in range(2, count):
            rank = (index - 2) // self.cfg.num_lanes + 1
            direction = -1.0 if index % 2 == 0 else 1.0
            self.x[index] = focal + direction * (35.0 + 28.0 * rank)

        self.speed[:] = self.rng.uniform(38.0, 42.0, size=count) / 3.6
        self.completed[:] = ~self.goal_active
        self.collision = False
        self.steps = 0
        self._update_completion()
        return self.snapshot()

    def step(self, actions: np.ndarray) -> Tuple[bool, Dict[str, object]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.cfg.num_vehicles,):
            raise ValueError("actions must have shape (num_vehicles,)")
        if np.any((actions < 0) | (actions >= len(ACTION_NAMES))):
            raise ValueError("invalid action")
        requested_actions = actions.copy()
        actions, safety_interventions = self.apply_local_safety_shield(actions)

        changing = np.abs(self.lane_destination - self.lane_pos) > 1e-4
        start_left = (actions == LANE_LEFT) & ~changing
        start_right = (actions == LANE_RIGHT) & ~changing
        self.lane_destination = np.where(
            start_left,
            np.maximum(0.0, np.rint(self.lane_pos) - 1.0),
            self.lane_destination,
        )
        self.lane_destination = np.where(
            start_right,
            np.minimum(float(self.cfg.num_lanes - 1), np.rint(self.lane_pos) + 1.0),
            self.lane_destination,
        )

        acceleration = np.zeros_like(self.speed)
        acceleration[actions == ACCELERATE] = self.cfg.accel_mps2
        acceleration[actions == YIELD] = -self.cfg.yield_decel_mps2
        flow_mps = self.cfg.flow_speed_kmh / 3.6
        cruising = (actions == KEEP_LANE) | (actions == LANE_LEFT) | (actions == LANE_RIGHT)
        acceleration[cruising] += np.clip(flow_mps - self.speed[cruising], -1.0, 1.0)
        previous_speed = self.speed.copy()
        self.speed += acceleration * self.cfg.dt_s
        self.speed[:] = np.clip(self.speed, 0.0, self.cfg.max_speed_kmh / 3.6)
        self.x += self.speed * self.cfg.dt_s

        lane_step = self.cfg.dt_s / self.cfg.lane_change_duration_s
        lane_delta = self.lane_destination - self.lane_pos
        self.lane_pos += np.clip(lane_delta, -lane_step, lane_step)
        self.steps += 1
        self._update_completion()

        pair_metrics = self._pair_metrics()
        self.collision = self.collision or bool(np.any(pair_metrics["collision_pairs"]))
        success = bool(np.all(self.completed[self.goal_active]))
        timeout = self.steps >= self.cfg.max_steps
        done = success or timeout or self.collision
        info: Dict[str, object] = {
            "success": success,
            "timeout": timeout,
            "collision": self.collision,
            "completed": self.completed.copy(),
            "speed_delta_mps": self.speed - previous_speed,
            "requested_actions": requested_actions,
            "executed_actions": actions.copy(),
            "safety_interventions": safety_interventions,
            **pair_metrics,
        }
        return done, info

    def apply_local_safety_shield(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Override a closing rear vehicle with YIELD below the TTC threshold.

        The shield is a shared low-level safeguard, independent of coordination
        mode. It implements the emergency responsibility assumed by the
        protocol and returns an explicit intervention mask for measurement.
        """
        executed = np.asarray(actions, dtype=np.int64).copy()
        interventions = np.zeros(self.cfg.num_vehicles, dtype=bool)
        if not self.cfg.local_safety_shield:
            return executed, interventions
        for first in range(self.cfg.num_vehicles):
            for second in range(first + 1, self.cfg.num_vehicles):
                if abs(float(self.lane_pos[first] - self.lane_pos[second])) >= 0.75:
                    continue
                if self.x[first] <= self.x[second]:
                    rear, front = first, second
                else:
                    rear, front = second, first
                closing = float(self.speed[rear] - self.speed[front])
                clearance = max(
                    0.0,
                    abs(float(self.x[rear] - self.x[front])) - self.cfg.vehicle_length_m,
                )
                unsafe_ttc = (
                    closing > 0.05
                    and clearance / closing < self.cfg.ttc_threshold_s
                )
                recovery_needed = clearance < self.cfg.robust_clearance_m
                if unsafe_ttc or recovery_needed:
                    if executed[rear] != YIELD:
                        interventions[rear] = True
                    executed[rear] = YIELD
        return executed, interventions

    def snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "vehicle_ids": self.vehicle_ids.copy(),
            "x_m": self.x.copy(),
            "speed_mps": self.speed.copy(),
            "lane_position": self.lane_pos.copy(),
            "lane_destination": self.lane_destination.copy(),
            "goal_lane": self.goal_lane.copy(),
            "goal_active": self.goal_active.copy(),
            "goal_is_exit": self.goal_is_exit.copy(),
            "completed": self.completed.copy(),
        }

    def current_lane(self, index: int) -> int:
        return int(np.rint(self.lane_pos[index]))

    def _update_completion(self) -> None:
        lane_reached = np.abs(self.lane_pos - self.goal_lane) < 0.08
        exit_reached = self.x >= self.cfg.exit_position_m
        reached = np.where(self.goal_is_exit, lane_reached & exit_reached, lane_reached)
        self.completed |= reached
        self.completed[~self.goal_active] = True

    def _pair_metrics(self) -> Dict[str, np.ndarray]:
        pairs = []
        centre_distances = []
        longitudinal_clearances = []
        corridor_overlaps = []
        ttc_values = []
        collisions = []
        gap_violations = []
        ttc_violations = []
        for first in range(self.cfg.num_vehicles):
            for second in range(first + 1, self.cfg.num_vehicles):
                dx = abs(float(self.x[first] - self.x[second]))
                dy = abs(float(self.lane_pos[first] - self.lane_pos[second])) * self.cfg.lane_width_m
                centre_distance = float(np.hypot(dx, dy))
                same_corridor = abs(float(self.lane_pos[first] - self.lane_pos[second])) < 0.75
                longitudinal_clearance = max(0.0, dx - self.cfg.vehicle_length_m)
                if self.x[first] <= self.x[second]:
                    closing = float(self.speed[first] - self.speed[second])
                else:
                    closing = float(self.speed[second] - self.speed[first])
                ttc = (
                    longitudinal_clearance / closing
                    if same_corridor and closing > 0.05
                    else float("inf")
                )
                pairs.append((int(self.vehicle_ids[first]), int(self.vehicle_ids[second])))
                centre_distances.append(centre_distance)
                longitudinal_clearances.append(
                    longitudinal_clearance if same_corridor else float("inf")
                )
                corridor_overlaps.append(same_corridor)
                ttc_values.append(ttc)
                collisions.append(
                    dx < self.cfg.vehicle_length_m and dy < self.cfg.vehicle_width_m
                )
                gap_violations.append(
                    same_corridor and longitudinal_clearance < self.cfg.safe_gap_m
                )
                ttc_violations.append(ttc < self.cfg.ttc_threshold_s)
        return {
            "pairs": np.asarray(pairs, dtype=np.int64),
            "pair_center_distance_m": np.asarray(centre_distances, dtype=np.float32),
            "pair_gap_m": np.asarray(longitudinal_clearances, dtype=np.float32),
            "corridor_overlap_pairs": np.asarray(corridor_overlaps, dtype=bool),
            "pair_ttc_s": np.asarray(ttc_values, dtype=np.float32),
            "collision_pairs": np.asarray(collisions, dtype=bool),
            "gap_violation_pairs": np.asarray(gap_violations, dtype=bool),
            "ttc_violation_pairs": np.asarray(ttc_violations, dtype=bool),
        }
