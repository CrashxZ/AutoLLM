"""Variable-fleet stress scenarios for coordination experiments.

This module layers reproducible conflict topologies and communication metadata
over :mod:`marl.fleet_highway_env` without changing its shared vehicle dynamics
or safety shield.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from marl.fleet_highway_env import FleetHighwayConfig, FleetHighwayEnv


@dataclass(frozen=True)
class CommunicationProfile:
    name: str
    delay_ms: int
    packet_loss_probability: float


@dataclass(frozen=True)
class StressScenarioDefinition:
    name: str
    description: str
    active_vehicle_count: int
    minimum_conflict_edges: int


COMMUNICATION_PROFILES = {
    "ideal": CommunicationProfile("ideal", delay_ms=0, packet_loss_probability=0.0),
    "delayed": CommunicationProfile(
        "delayed",
        delay_ms=400,
        packet_loss_probability=0.0,
    ),
    "lossy": CommunicationProfile(
        "lossy",
        delay_ms=0,
        packet_loss_probability=0.20,
    ),
    "delayed_lossy": CommunicationProfile(
        "delayed_lossy",
        delay_ms=400,
        packet_loss_probability=0.20,
    ),
}


SCENARIO_DEFINITIONS = {
    "converging_merge": StressScenarioDefinition(
        name="converging_merge",
        description="Two vehicles request the same intermediate lane.",
        active_vehicle_count=2,
        minimum_conflict_edges=1,
    ),
    "dual_swap": StressScenarioDefinition(
        name="dual_swap",
        description="Two spatially separated reciprocal lane swaps occur together.",
        active_vehicle_count=4,
        minimum_conflict_edges=2,
    ),
    "exit_weave": StressScenarioDefinition(
        name="exit_weave",
        description="An exit-bound vehicle crosses a reciprocal and chained merge.",
        active_vehicle_count=3,
        minimum_conflict_edges=2,
    ),
    "cascade": StressScenarioDefinition(
        name="cascade",
        description="Four adjacent vehicles form a coupled lane-change cascade.",
        active_vehicle_count=4,
        minimum_conflict_edges=2,
    ),
}


CORE_LAYOUTS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "converging_merge": {
        "lane": (0.0, 2.0, 1.0, 3.0),
        "x_m": (0.0, 7.0, 32.0, -32.0),
        "goal_lane": (1.0, 1.0, 1.0, 3.0),
        "active": (1.0, 1.0, 0.0, 0.0),
        "exit": (0.0, 0.0, 0.0, 0.0),
    },
    "dual_swap": {
        "lane": (0.0, 1.0, 2.0, 3.0),
        "x_m": (0.0, 8.0, 40.0, 48.0),
        "goal_lane": (1.0, 0.0, 3.0, 2.0),
        "active": (1.0, 1.0, 1.0, 1.0),
        "exit": (0.0, 0.0, 0.0, 0.0),
    },
    "exit_weave": {
        "lane": (0.0, 1.0, 2.0, 3.0),
        "x_m": (0.0, 8.0, 16.0, 38.0),
        "goal_lane": (3.0, 0.0, 1.0, 3.0),
        "active": (1.0, 1.0, 1.0, 0.0),
        "exit": (1.0, 0.0, 0.0, 0.0),
    },
    "cascade": {
        "lane": (0.0, 1.0, 2.0, 3.0),
        "x_m": (0.0, 14.0, 28.0, 42.0),
        "goal_lane": (1.0, 2.0, 3.0, 2.0),
        "active": (1.0, 1.0, 1.0, 1.0),
        "exit": (0.0, 0.0, 0.0, 0.0),
    },
}


class StressFleetHighwayEnv(FleetHighwayEnv):
    """Fleet environment initialized from a named coordination topology."""

    def __init__(
        self,
        config: FleetHighwayConfig,
        seed: int,
    ) -> None:
        if config.num_vehicles < 4:
            raise ValueError("stress scenarios require at least four vehicles")
        super().__init__(config=config, seed=seed)
        self.seed = int(seed)
        self.scenario_name = ""
        self.communication_profile = COMMUNICATION_PROFILES["ideal"]
        self.cooperative = np.ones(config.num_vehicles, dtype=bool)

    def reset_stress(
        self,
        scenario_name: str,
        *,
        communication_profile: str = "ideal",
        one_noncooperative: bool = False,
    ) -> Dict[str, np.ndarray]:
        if scenario_name not in SCENARIO_DEFINITIONS:
            raise ValueError(f"unknown stress scenario: {scenario_name}")
        if communication_profile not in COMMUNICATION_PROFILES:
            raise ValueError(
                f"unknown communication profile: {communication_profile}"
            )
        self.rng = np.random.default_rng(self.seed)
        super().reset(scenario_id=1)
        self.scenario_name = scenario_name
        self.communication_profile = COMMUNICATION_PROFILES[communication_profile]
        layout = CORE_LAYOUTS[scenario_name]

        global_shift = float(self.rng.uniform(4.0, 12.0))
        core_jitter = self.rng.uniform(-0.35, 0.35, size=4)
        self.lane_pos[:4] = np.asarray(layout["lane"], dtype=np.float32)
        self.x[:4] = (
            np.asarray(layout["x_m"], dtype=np.float32)
            + global_shift
            + core_jitter.astype(np.float32)
        )
        self.goal_lane[:4] = np.asarray(layout["goal_lane"], dtype=np.int64)
        self.goal_active[:4] = np.asarray(layout["active"], dtype=bool)
        self.goal_is_exit[:4] = np.asarray(layout["exit"], dtype=bool)

        for index in range(4, self.cfg.num_vehicles):
            self.lane_pos[index] = float(index % self.cfg.num_lanes)
            rank = (index - 4) // self.cfg.num_lanes + 1
            direction = 1.0 if index % 2 == 0 else -1.0
            self.x[index] = global_shift + direction * (90.0 + 45.0 * rank)
            self.goal_lane[index] = int(round(float(self.lane_pos[index])))
            self.goal_active[index] = False
            self.goal_is_exit[index] = False

        self.lane_destination[:] = self.lane_pos
        self.speed[:] = self.rng.uniform(
            38.0,
            42.0,
            size=self.cfg.num_vehicles,
        ) / 3.6
        self.completed[:] = ~self.goal_active
        self.collision = False
        self.steps = 0
        self._update_completion()

        self.cooperative[:] = True
        if one_noncooperative:
            active_indices = np.flatnonzero(self.goal_active)
            self.cooperative[int(active_indices[-1])] = False
        return self.snapshot()

    def snapshot(self) -> Dict[str, np.ndarray]:
        snapshot = super().snapshot()
        snapshot.update(
            {
                "cooperative": self.cooperative.copy(),
                "communication_delay_ms": np.asarray(
                    [self.communication_profile.delay_ms],
                    dtype=np.int64,
                ),
                "packet_loss_probability": np.asarray(
                    [self.communication_profile.packet_loss_probability],
                    dtype=np.float32,
                ),
            }
        )
        return snapshot

    def proposed_lane_targets(self) -> Dict[int, int]:
        """Return each incomplete active vehicle's next single-lane target."""
        targets: Dict[int, int] = {}
        for index in np.flatnonzero(self.goal_active & ~self.completed):
            current = self.current_lane(int(index))
            goal = int(self.goal_lane[index])
            if current == goal:
                continue
            direction = 1 if goal > current else -1
            targets[int(index)] = current + direction
        return targets

    def maneuver_conflict_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """Return same-target and reciprocal-swap request conflicts."""
        targets = self.proposed_lane_targets()
        conflicts = []
        indices = sorted(targets)
        for position, first in enumerate(indices):
            first_source = self.current_lane(first)
            for second in indices[position + 1 :]:
                second_source = self.current_lane(second)
                same_target = targets[first] == targets[second]
                lane_swap = (
                    targets[first] == second_source
                    and targets[second] == first_source
                )
                if same_target or lane_swap:
                    conflicts.append((first, second))
        return tuple(conflicts)
