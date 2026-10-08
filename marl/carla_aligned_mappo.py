"""CARLA-aligned kinematic scenarios for the adapted MAPPO baseline.

The layouts reproduce the maneuver topology used by the Town04 conflict-active
comparison while randomizing longitudinal offsets and initial speeds.  Train,
validation, and test instances use disjoint seed namespaces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from .fleet_highway_env import FleetHighwayConfig, FleetHighwayEnv
from .fleet_mappo import (
    PRIORITY_OBSERVATION_DIM,
    FleetMAPPOAdapter,
    actor_observations,
    centralized_context,
)


FLEET_SIZES = (2, 4, 8)
SCENARIO_FAMILIES = (
    "close_reciprocal",
    "contested_merge",
    "blocked_merge",
)
CORE_CELLS = tuple(
    (family, fleet_size)
    for family in SCENARIO_FAMILIES
    for fleet_size in FLEET_SIZES
)
ALIGNED_MINIMUM_CLEARANCE_M = 5.0


class CarlaAlignedSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SPLIT_SEED_BASE = {
    CarlaAlignedSplit.TRAIN: 12_100_000,
    CarlaAlignedSplit.VALIDATION: 13_100_000,
    CarlaAlignedSplit.TEST: 14_100_000,
}
SPLIT_SEED_LIMIT = {
    split: base + 900_000 for split, base in SPLIT_SEED_BASE.items()
}


@dataclass(frozen=True)
class CarlaAlignedScenarioSpec:
    split: CarlaAlignedSplit
    family: str
    fleet_size: int
    ordinal: int
    seed: int


def scenario_spec(
    split: CarlaAlignedSplit,
    family: str,
    fleet_size: int,
    ordinal: int,
) -> CarlaAlignedScenarioSpec:
    if family not in SCENARIO_FAMILIES:
        raise ValueError(f"unknown CARLA-aligned family: {family}")
    if fleet_size not in FLEET_SIZES:
        raise ValueError(f"unsupported CARLA-aligned fleet size: {fleet_size}")
    if ordinal < 0:
        raise ValueError("scenario ordinal cannot be negative")
    seed = SPLIT_SEED_BASE[split] + int(ordinal)
    if seed >= SPLIT_SEED_LIMIT[split]:
        raise ValueError("scenario ordinal exhausted the split namespace")
    return CarlaAlignedScenarioSpec(split, family, fleet_size, ordinal, seed)


def split_seed_ranges_disjoint() -> bool:
    intervals = sorted(
        (SPLIT_SEED_BASE[split], SPLIT_SEED_LIMIT[split])
        for split in SPLIT_SEED_BASE
    )
    return all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:]))


def balanced_priority_order(fleet_size: int, ordinal: int) -> tuple[int, ...]:
    """Return a reproducible rotation that balances priority across episodes.

    Random shuffling can accidentally assign the same role ordering throughout
    a small validation cell.  Adjacent ordinals instead rotate which vehicle
    has first priority, preventing priority rank from becoming a role proxy.
    """
    if fleet_size < 1:
        raise ValueError("fleet_size must be positive")
    if ordinal < 0:
        raise ValueError("scenario ordinal cannot be negative")
    order = tuple(range(int(fleet_size)))
    start = int(ordinal) % int(fleet_size)
    return order[start:] + order[:start]


def _base_layout(family: str, fleet_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if family == "close_reciprocal":
        if fleet_size == 2:
            lane = np.asarray([1, 2], dtype=np.float32)
            goal = np.asarray([2, 1], dtype=np.int64)
        else:
            lane = np.resize(np.arange(4, dtype=np.float32), fleet_size)
            goal = np.resize(np.asarray([1, 0, 3, 2], dtype=np.int64), fleet_size)
    elif family == "contested_merge":
        if fleet_size == 2:
            lane = np.asarray([0, 2], dtype=np.float32)
            goal = np.asarray([1, 1], dtype=np.int64)
        elif fleet_size == 4:
            lane = np.asarray([0, 2, 0, 2], dtype=np.float32)
            goal = np.asarray([1, 1, 1, 1], dtype=np.int64)
        else:
            lane = np.resize(np.arange(4, dtype=np.float32), fleet_size)
            goal = np.resize(np.asarray([1, 2, 1, 2], dtype=np.int64), fleet_size)
    elif family == "blocked_merge":
        if fleet_size in {2, 4}:
            lane = np.resize(np.asarray([2, 1], dtype=np.float32), fleet_size)
            goal = lane.astype(np.int64)
            goal[::2] = 1
        else:
            lane = np.resize(np.arange(4, dtype=np.float32), fleet_size)
            goal = lane.astype(np.int64)
            goal[0::4] = 1
            goal[3::4] = 2
    else:  # pragma: no cover - guarded by scenario_spec
        raise ValueError(f"unknown CARLA-aligned family: {family}")
    return lane, goal


def _goal_active(family: str, fleet_size: int) -> np.ndarray:
    active = np.ones(fleet_size, dtype=bool)
    if family == "blocked_merge":
        active[:] = False
        if fleet_size in {2, 4}:
            active[::2] = True
        else:
            active[0::4] = True
            active[3::4] = True
    return active


class CarlaAlignedFleetEnv(FleetHighwayEnv):
    """Kinematic fleet initialized from a randomized CARLA maneuver topology."""

    def __init__(self, config: FleetHighwayConfig, seed: int) -> None:
        super().__init__(config=config, seed=seed)
        self.cooperative = np.ones(config.num_vehicles, dtype=bool)
        self.scenario_spec: CarlaAlignedScenarioSpec | None = None
        self.initial_state_sha256 = ""

    def reset_aligned(self, spec: CarlaAlignedScenarioSpec) -> Dict[str, np.ndarray]:
        if spec.fleet_size != self.cfg.num_vehicles:
            raise ValueError("scenario and environment fleet sizes differ")
        self.scenario_spec = spec
        self.rng = np.random.default_rng(spec.seed)
        lane, goal = _base_layout(spec.family, spec.fleet_size)
        active = _goal_active(spec.family, spec.fleet_size)

        group_size = (
            4
            if spec.fleet_size == 8
            or (spec.family == "close_reciprocal" and spec.fleet_size == 4)
            else 2
        )
        group = np.arange(spec.fleet_size, dtype=np.float32) // group_size
        spacing_m = float(self.rng.uniform(20.0, 30.0))
        origin_m = float(self.rng.uniform(30.0, 60.0))
        self.x[:] = origin_m + group * spacing_m
        self.x[:] += self.rng.uniform(-1.0, 1.0, size=spec.fleet_size)
        self.lane_pos[:] = lane
        self.lane_destination[:] = lane
        self.goal_lane[:] = goal
        self.goal_active[:] = active
        self.goal_is_exit[:] = False
        if spec.family == "blocked_merge":
            # A blocked merge must require an explicit support decision.  If
            # pair speeds are sampled independently over a wide range, many
            # episodes resolve through an accidental natural gap and the actor
            # never has to learn which non-requesting vehicle should maintain
            # flow.  CARLA begins these conflict-active blocks with the pair at
            # nearly equal speed, so reproduce that condition here.
            base_speed_kmh = float(self.rng.uniform(30.0, 34.0))
            speed_kmh = base_speed_kmh + self.rng.uniform(
                -0.75, 0.75, size=spec.fleet_size
            )
        else:
            speed_kmh = self.rng.uniform(30.0, 52.0, size=spec.fleet_size)
        self.speed[:] = speed_kmh / 3.6
        self.completed[:] = ~active
        self.cooperative[:] = True
        self.collision = False
        self.steps = 0
        self._update_completion()
        self.initial_state_sha256 = self._state_hash()
        return self.snapshot()

    def _state_hash(self) -> str:
        payload = {
            "family": self.scenario_spec.family if self.scenario_spec else None,
            "fleet_size": self.cfg.num_vehicles,
            "x_m": np.round(self.x, 7).tolist(),
            "lane": np.round(self.lane_pos, 7).tolist(),
            "speed_mps": np.round(self.speed, 7).tolist(),
            "goal_lane": self.goal_lane.tolist(),
            "goal_active": self.goal_active.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def snapshot(self) -> Dict[str, np.ndarray]:
        output = super().snapshot()
        output["cooperative"] = self.cooperative.copy()
        return output


class CarlaAlignedMAPPOAdapter(FleetMAPPOAdapter):
    """Expose priority-aware actor observations for one aligned environment."""

    observation_dim = PRIORITY_OBSERVATION_DIM

    def __init__(
        self,
        env: CarlaAlignedFleetEnv,
        priority_order: tuple[int, ...],
    ) -> None:
        super().__init__(env)
        self.env = env
        self.priority_order = tuple(int(value) for value in priority_order)

    def reset_aligned(
        self, spec: CarlaAlignedScenarioSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.env.reset_aligned(spec)
        return self.observations()

    def observations(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            actor_observations(self.env, priority_order=self.priority_order),
            centralized_context(self.env),
        )
