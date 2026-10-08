"""Procedural, conflict-controlled scenarios for coordination benchmark v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple

import numpy as np

from .coordination_channel import DelayedLossyActionChannel
from .fleet_highway_env import FleetHighwayConfig, FleetHighwayEnv


class ConflictDensity(str, Enum):
    NULL = "null"
    SPARSE = "sparse"
    DENSE = "dense"


class ArrivalPattern(str, Enum):
    SIMULTANEOUS = "simultaneous"
    STAGGERED = "staggered"


class SpeedDifferential(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScenarioFamily(str, Enum):
    FACTORIAL = "factorial"
    DEPENDENCY_CASCADE = "dependency_cascade"
    SHARED_EXIT_BOTTLENECK = "shared_exit_bottleneck"
    DYNAMIC_GOAL_REVISION = "dynamic_goal_revision"
    DELAYED_COMMITMENT_MERGE = "delayed_commitment_merge"


class GeometryProfile(str, Enum):
    FIXED_TEMPLATE = "fixed_template"
    RANDOMIZED_CONNECTED = "randomized_connected"


class ScenarioSplit(str, Enum):
    UNSPECIFIED = "unspecified"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class ProceduralScenarioSpec:
    active_vehicle_count: int
    background_vehicle_count: int = 0
    conflict_density: ConflictDensity = ConflictDensity.SPARSE
    maneuver_depth: int = 1
    arrival_pattern: ArrivalPattern = ArrivalPattern.SIMULTANEOUS
    speed_differential: SpeedDifferential = SpeedDifferential.LOW
    deadline_s: float | None = None
    seed: int = 0
    conflict_distance_m: float = 50.0
    ttc_threshold_s: float = 4.0
    scenario_family: ScenarioFamily = ScenarioFamily.FACTORIAL
    goal_revision_time_s: float | None = None
    communication_delay_ms: int = 0
    packet_loss_probability: float = 0.0
    communication_seed: int = 0
    geometry_profile: GeometryProfile = GeometryProfile.FIXED_TEMPLATE
    scenario_split: ScenarioSplit = ScenarioSplit.UNSPECIFIED

    def __post_init__(self) -> None:
        if self.active_vehicle_count < 2 or self.active_vehicle_count % 2:
            raise ValueError("active_vehicle_count must be an even integer >= 2")
        if self.background_vehicle_count < 0:
            raise ValueError("background_vehicle_count cannot be negative")
        if self.maneuver_depth not in {1, 2}:
            raise ValueError("maneuver_depth must be 1 or 2")
        if self.deadline_s is not None and self.deadline_s <= 0.0:
            raise ValueError("deadline_s must be positive")
        if self.conflict_distance_m <= 0.0 or self.ttc_threshold_s <= 0.0:
            raise ValueError("conflict thresholds must be positive")
        if self.goal_revision_time_s is not None and self.goal_revision_time_s <= 0.0:
            raise ValueError("goal_revision_time_s must be positive")
        if self.communication_delay_ms < 0:
            raise ValueError("communication_delay_ms cannot be negative")
        if not 0.0 <= self.packet_loss_probability <= 1.0:
            raise ValueError("packet_loss_probability must be in [0, 1]")
        if (
            self.scenario_family is not ScenarioFamily.FACTORIAL
            and self.geometry_profile is GeometryProfile.FIXED_TEMPLATE
            and self.active_vehicle_count % 4
        ):
            raise ValueError("stress scenario active_vehicle_count must be divisible by 4")
        if (
            self.geometry_profile is GeometryProfile.RANDOMIZED_CONNECTED
            and self.scenario_family is ScenarioFamily.FACTORIAL
        ):
            raise ValueError("randomized connected geometry requires a stress family")
        if self.geometry_profile is GeometryProfile.RANDOMIZED_CONNECTED:
            if self.active_vehicle_count not in {4, 6, 8}:
                raise ValueError("randomized connected fleet size must be 4, 6, or 8")
            if self.background_vehicle_count:
                raise ValueError("randomized connected scenarios do not use background vehicles")
            if self.scenario_split is ScenarioSplit.UNSPECIFIED:
                raise ValueError("randomized connected scenarios require a declared split")

    @property
    def vehicle_count(self) -> int:
        return self.active_vehicle_count + self.background_vehicle_count


@dataclass(frozen=True)
class PlannedManeuver:
    vehicle_index: int
    source_lane: int
    first_target_lane: int
    final_goal_lane: int
    x_m: float
    speed_mps: float
    activation_step: int


@dataclass(frozen=True)
class GeneratedScenario:
    spec: ProceduralScenarioSpec
    lane: Tuple[float, ...]
    x_m: Tuple[float, ...]
    speed_mps: Tuple[float, ...]
    goal_lane: Tuple[int, ...]
    goal_is_exit: Tuple[bool, ...]
    intended_active: Tuple[bool, ...]
    activation_step: Tuple[int, ...]
    deadline_step: Tuple[int | None, ...]
    planned_conflict_edges: Tuple[Tuple[int, int], ...]
    goal_revision_step: Tuple[int | None, ...]
    revised_goal_lane: Tuple[int | None, ...]

    def apply(self, env: FleetHighwayEnv) -> Dict[str, np.ndarray]:
        """Reset an environment to this generated physical state."""
        if env.cfg.num_vehicles != self.spec.vehicle_count:
            raise ValueError("environment vehicle count does not match scenario")
        env.lane_pos[:] = np.asarray(self.lane, dtype=np.float32)
        env.lane_destination[:] = env.lane_pos
        env.x[:] = np.asarray(self.x_m, dtype=np.float32)
        env.speed[:] = np.asarray(self.speed_mps, dtype=np.float32)
        env.goal_lane[:] = np.asarray(self.goal_lane, dtype=np.int64)
        env.goal_is_exit[:] = np.asarray(self.goal_is_exit, dtype=bool)
        active_now = np.asarray(self.activation_step, dtype=np.int64) <= 0
        env.goal_active[:] = np.asarray(self.intended_active, dtype=bool) & active_now
        env.completed[:] = ~env.goal_active
        env.collision = False
        env.steps = 0
        env._update_completion()
        return env.snapshot()

    def activate_due_goals(self, env: FleetHighwayEnv) -> Tuple[int, ...]:
        """Activate scheduled maneuver requests at the current environment step."""
        if env.cfg.num_vehicles != self.spec.vehicle_count:
            raise ValueError("environment vehicle count does not match scenario")
        intended = np.asarray(self.intended_active, dtype=bool)
        due = np.asarray(self.activation_step, dtype=np.int64) <= env.steps
        new_indices = np.flatnonzero(intended & due & ~env.goal_active)
        for index in new_indices:
            env.goal_active[index] = True
            env.completed[index] = False
        env._update_completion()
        return tuple(int(index) for index in new_indices)

    def initial_state_sha256(self) -> str:
        payload = {
            "spec": _jsonable_spec(self.spec),
            "lane": self.lane,
            "x_m": self.x_m,
            "speed_mps": self.speed_mps,
            "goal_lane": self.goal_lane,
            "goal_is_exit": self.goal_is_exit,
            "intended_active": self.intended_active,
            "activation_step": self.activation_step,
            "deadline_step": self.deadline_step,
            "planned_conflict_edges": self.planned_conflict_edges,
            "goal_revision_step": self.goal_revision_step,
            "revised_goal_lane": self.revised_goal_lane,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def planned_maneuvers(self) -> Tuple[PlannedManeuver, ...]:
        output = []
        for index in range(self.spec.active_vehicle_count):
            source = int(round(self.lane[index]))
            goal = int(self.goal_lane[index])
            direction = 1 if goal > source else -1
            output.append(
                PlannedManeuver(
                    vehicle_index=index,
                    source_lane=source,
                    first_target_lane=source + direction,
                    final_goal_lane=goal,
                    x_m=float(self.x_m[index]),
                    speed_mps=float(self.speed_mps[index]),
                    activation_step=int(self.activation_step[index]),
                )
            )
        return tuple(output)


class ProceduralFleetHighwayEnv(FleetHighwayEnv):
    """Fleet environment with a reproducible generated-scenario lifecycle."""

    def __init__(
        self,
        config: FleetHighwayConfig,
        seed: int,
    ) -> None:
        super().__init__(config=config, seed=seed)
        self.seed = int(seed)
        self.cooperative = np.ones(config.num_vehicles, dtype=bool)
        self.generated_scenario: GeneratedScenario | None = None
        self.last_goal_revision_indices: Tuple[int, ...] = ()
        self.goal_revision_events: list[dict] = []
        self.action_channel: DelayedLossyActionChannel | None = None

    def reset_procedural(
        self,
        spec: ProceduralScenarioSpec,
    ) -> Dict[str, np.ndarray]:
        if spec.vehicle_count != self.cfg.num_vehicles:
            raise ValueError("scenario and environment vehicle counts differ")
        scenario = generate_scenario(spec, dt_s=self.cfg.dt_s)
        self.generated_scenario = scenario
        self.cooperative[:] = True
        self.last_goal_revision_indices = ()
        self.goal_revision_events = []
        self.action_channel = DelayedLossyActionChannel(
            self.cfg.num_vehicles,
            dt_s=self.cfg.dt_s,
            delay_ms=spec.communication_delay_ms,
            packet_loss_probability=spec.packet_loss_probability,
            seed=spec.communication_seed,
        )
        return scenario.apply(self)

    def activate_due_goals(self) -> Tuple[int, ...]:
        if self.generated_scenario is None:
            raise RuntimeError("reset_procedural must be called first")
        activated = self.generated_scenario.activate_due_goals(self)
        revised = []
        for index, due_step in enumerate(self.generated_scenario.goal_revision_step):
            if due_step is None or due_step != self.steps:
                continue
            target = self.generated_scenario.revised_goal_lane[index]
            if target is None:
                continue
            previous = int(self.goal_lane[index])
            self.goal_lane[index] = int(target)
            self.goal_active[index] = True
            self.completed[index] = False
            revised.append(index)
            self.goal_revision_events.append(
                {
                    "step": int(self.steps),
                    "vehicle_index": int(index),
                    "previous_goal_lane": previous,
                    "revised_goal_lane": int(target),
                }
            )
        self.last_goal_revision_indices = tuple(revised)
        self._update_completion()
        return activated

    def step(self, actions: np.ndarray):
        proposed = np.asarray(actions, dtype=np.int64)
        if self.action_channel is None:
            delivered = proposed
        else:
            fallback = np.zeros(self.cfg.num_vehicles, dtype=np.int64)
            delivered = self.action_channel.transmit(
                self.steps,
                proposed,
                fallback,
            )
        _done, info = super().step(delivered)
        if self.generated_scenario is None:
            raise RuntimeError("reset_procedural must be called first")
        intended = np.asarray(self.generated_scenario.intended_active, dtype=bool)
        all_activated = bool(np.all(self.goal_active[intended]))
        pending_revision = any(
            due_step is not None and due_step >= self.steps
            for due_step in self.generated_scenario.goal_revision_step
        )
        success = (
            all_activated
            and not pending_revision
            and bool(np.all(self.completed[intended]))
        )
        info["success"] = success
        done = success or bool(info["collision"]) or bool(info["timeout"])
        info["coordinator_actions"] = proposed.copy()
        info["channel_actions"] = delivered.copy()
        return done, info

    def snapshot(self) -> Dict[str, np.ndarray]:
        snapshot = super().snapshot()
        snapshot["cooperative"] = self.cooperative.copy()
        if self.generated_scenario is not None:
            snapshot["intended_active"] = np.asarray(
                self.generated_scenario.intended_active,
                dtype=bool,
            )
            snapshot["activation_step"] = np.asarray(
                self.generated_scenario.activation_step,
                dtype=np.int64,
            )
            snapshot["goal_revision_step"] = np.asarray(
                [
                    -1 if value is None else int(value)
                    for value in self.generated_scenario.goal_revision_step
                ],
                dtype=np.int64,
            )
        return snapshot


def generate_scenario(
    spec: ProceduralScenarioSpec,
    *,
    dt_s: float = 0.2,
) -> GeneratedScenario:
    """Generate a reproducible scenario with an exact planned conflict graph."""
    if spec.vehicle_count < 2:
        raise ValueError("scenario must contain at least two vehicles")
    rng = np.random.default_rng(spec.seed)
    if spec.scenario_family is not ScenarioFamily.FACTORIAL:
        return _generate_stress_scenario(spec, rng=rng, dt_s=dt_s)
    active_count = spec.active_vehicle_count
    lane = np.zeros(spec.vehicle_count, dtype=np.float64)
    x_m = np.zeros(spec.vehicle_count, dtype=np.float64)
    goal_lane = np.zeros(spec.vehicle_count, dtype=np.int64)
    goal_is_exit = np.zeros(spec.vehicle_count, dtype=bool)
    intended_active = np.zeros(spec.vehicle_count, dtype=bool)
    intended_active[:active_count] = True

    global_shift = float(rng.uniform(20.0, 40.0))
    jitter = rng.uniform(-0.15, 0.15, size=active_count)
    for index in range(active_count):
        pair_index = index // 2
        member = index % 2
        if spec.conflict_density is ConflictDensity.NULL:
            source = 0 if member == 0 else 3
            goal = (
                source + (1 if source == 0 else -1) * spec.maneuver_depth
            )
            longitudinal = global_shift + pair_index * 140.0
        elif spec.conflict_density is ConflictDensity.SPARSE:
            source = 0 if member == 0 else 2
            goal = 1 if spec.maneuver_depth == 1 else (2 if source == 0 else 0)
            longitudinal = global_shift + pair_index * 140.0
        else:
            source = 0 if member == 0 else 2
            goal = 1 if spec.maneuver_depth == 1 else (2 if source == 0 else 0)
            same_lane_slot = index // 2
            longitudinal = global_shift + same_lane_slot * 13.0
        lane[index] = source
        goal_lane[index] = goal
        x_m[index] = longitudinal + jitter[index]

    for index in range(active_count, spec.vehicle_count):
        background_rank = index - active_count
        lane[index] = background_rank % 4
        direction = -1.0 if background_rank % 2 == 0 else 1.0
        x_m[index] = global_shift + direction * (
            220.0 + 70.0 * (background_rank // 4)
        )
        goal_lane[index] = int(lane[index])

    speed_mps = _speeds(spec, rng)
    activation_step = np.full(spec.vehicle_count, np.iinfo(np.int32).max)
    if spec.arrival_pattern is ArrivalPattern.SIMULTANEOUS:
        activation_step[:active_count] = 0
    else:
        activation_step[:active_count] = np.repeat(
            np.arange(active_count // 2, dtype=np.int64) * 10,
            2,
        )
    deadline_step: list[int | None] = [None] * spec.vehicle_count
    if spec.deadline_s is not None:
        deadline_offset = int(round(spec.deadline_s / dt_s))
        for index in range(active_count):
            deadline_step[index] = int(activation_step[index] + deadline_offset)

    maneuvers = _planned_maneuvers(
        lane=lane,
        x_m=x_m,
        speed_mps=speed_mps,
        goal_lane=goal_lane,
        activation_step=activation_step,
        active_count=active_count,
    )
    edges = planned_conflict_edges(
        maneuvers,
        conflict_distance_m=spec.conflict_distance_m,
        ttc_threshold_s=spec.ttc_threshold_s,
        vehicle_length_m=4.7,
    )
    expected = expected_conflict_edge_count(spec)
    if len(edges) != expected:
        raise RuntimeError(
            f"generated {len(edges)} conflict edges; expected {expected}"
        )

    return GeneratedScenario(
        spec=spec,
        lane=tuple(float(value) for value in lane),
        x_m=tuple(float(value) for value in x_m),
        speed_mps=tuple(float(value) for value in speed_mps),
        goal_lane=tuple(int(value) for value in goal_lane),
        goal_is_exit=tuple(bool(value) for value in goal_is_exit),
        intended_active=tuple(bool(value) for value in intended_active),
        activation_step=tuple(int(value) for value in activation_step),
        deadline_step=tuple(deadline_step),
        planned_conflict_edges=edges,
        goal_revision_step=tuple(None for _ in range(spec.vehicle_count)),
        revised_goal_lane=tuple(None for _ in range(spec.vehicle_count)),
    )


def _generate_stress_scenario(
    spec: ProceduralScenarioSpec,
    *,
    rng: np.random.Generator,
    dt_s: float,
) -> GeneratedScenario:
    if spec.geometry_profile is GeometryProfile.RANDOMIZED_CONNECTED:
        return _generate_randomized_connected_scenario(
            spec,
            rng=rng,
            dt_s=dt_s,
        )
    count = spec.vehicle_count
    active_count = spec.active_vehicle_count
    lane = np.zeros(count, dtype=np.float64)
    x_m = np.zeros(count, dtype=np.float64)
    goal_lane = np.zeros(count, dtype=np.int64)
    goal_is_exit = np.zeros(count, dtype=bool)
    intended_active = np.zeros(count, dtype=bool)
    intended_active[:active_count] = True
    activation_step = np.full(count, np.iinfo(np.int32).max, dtype=np.int64)
    activation_step[:active_count] = 0
    deadline_step: list[int | None] = [None] * count
    revision_step: list[int | None] = [None] * count
    revised_goal: list[int | None] = [None] * count
    global_shift = float(rng.uniform(20.0, 40.0))
    jitter = rng.uniform(-0.15, 0.15, size=active_count)

    family = spec.scenario_family
    for index in range(active_count):
        group = index // 4
        member = index % 4
        if family is ScenarioFamily.DEPENDENCY_CASCADE:
            source = member
            target = (1, 2, 3, 2)[member]
            longitudinal = global_shift + group * 140.0 + member * 9.0
        elif family is ScenarioFamily.SHARED_EXIT_BOTTLENECK:
            source = member
            target = 3
            longitudinal = global_shift + 260.0 + group * 120.0 + member * 8.0
            goal_is_exit[index] = True
        elif family is ScenarioFamily.DYNAMIC_GOAL_REVISION:
            source = member
            target = (3, 0, 3, 0)[member]
            longitudinal = global_shift + group * 140.0 + member * 8.0
            if member == 0:
                event_time = spec.goal_revision_time_s or 1.0
                revision_step[index] = max(1, int(round(event_time / dt_s)))
                revised_goal[index] = source
        elif family is ScenarioFamily.DELAYED_COMMITMENT_MERGE:
            pair = index // 2
            source = 0 if index % 2 == 0 else 2
            target = 1
            longitudinal = global_shift + pair * 90.0 + (index % 2) * 7.0
        else:  # pragma: no cover - exhaustive guard for future enum values
            raise ValueError(f"unsupported stress family: {family.value}")
        lane[index] = source
        goal_lane[index] = target
        x_m[index] = longitudinal + jitter[index]

    for index in range(active_count, count):
        background_rank = index - active_count
        lane[index] = background_rank % 4
        x_m[index] = global_shift + 300.0 + background_rank * 70.0
        goal_lane[index] = int(lane[index])

    speed_mps = _speeds(spec, rng)
    if spec.deadline_s is not None:
        deadline_offset = int(round(spec.deadline_s / dt_s))
        for index in range(active_count):
            deadline_step[index] = deadline_offset

    maneuvers = _planned_maneuvers(
        lane=lane,
        x_m=x_m,
        speed_mps=speed_mps,
        goal_lane=goal_lane,
        activation_step=activation_step,
        active_count=active_count,
    )
    edges = stress_dependency_edges(
        maneuvers,
        conflict_distance_m=spec.conflict_distance_m,
    )
    if not edges:
        raise RuntimeError(f"stress scenario {family.value} has no planned dependency edge")

    return GeneratedScenario(
        spec=spec,
        lane=tuple(float(value) for value in lane),
        x_m=tuple(float(value) for value in x_m),
        speed_mps=tuple(float(value) for value in speed_mps),
        goal_lane=tuple(int(value) for value in goal_lane),
        goal_is_exit=tuple(bool(value) for value in goal_is_exit),
        intended_active=tuple(bool(value) for value in intended_active),
        activation_step=tuple(int(value) for value in activation_step),
        deadline_step=tuple(deadline_step),
        planned_conflict_edges=edges,
        goal_revision_step=tuple(revision_step),
        revised_goal_lane=tuple(revised_goal),
    )


def _generate_randomized_connected_scenario(
    spec: ProceduralScenarioSpec,
    *,
    rng: np.random.Generator,
    dt_s: float,
) -> GeneratedScenario:
    """Generate one connected, independently randomized stress instance."""
    count = spec.active_vehicle_count
    safety_config = FleetHighwayConfig(num_vehicles=count)
    for _attempt in range(128):
        lane = np.resize(np.arange(4, dtype=np.float64), count)
        rng.shuffle(lane)
        goal_lane = np.zeros(count, dtype=np.int64)
        goal_is_exit = np.zeros(count, dtype=bool)

        if spec.scenario_family is ScenarioFamily.DEPENDENCY_CASCADE:
            target_by_lane = np.asarray([1, 2, 1, 2], dtype=np.int64)
            goal_lane[:] = target_by_lane[lane.astype(np.int64)]
        elif spec.scenario_family is ScenarioFamily.SHARED_EXIT_BOTTLENECK:
            goal_lane[:] = 3
            goal_is_exit[:] = True
        elif spec.scenario_family is ScenarioFamily.DYNAMIC_GOAL_REVISION:
            target_by_lane = np.asarray([3, 2, 1, 0], dtype=np.int64)
            goal_lane[:] = target_by_lane[lane.astype(np.int64)]
        elif spec.scenario_family is ScenarioFamily.DELAYED_COMMITMENT_MERGE:
            lane[:] = np.resize(np.asarray([0.0, 2.0]), count)
            rng.shuffle(lane)
            goal_lane[:] = 1
        else:  # pragma: no cover - guarded by ProceduralScenarioSpec
            raise ValueError("randomized geometry requires a stress scenario family")

        gaps = rng.uniform(7.0, 14.0, size=count - 1)
        ordered_x = np.concatenate(([0.0], np.cumsum(gaps)))
        permutation = rng.permutation(count)
        x_m = np.empty(count, dtype=np.float64)
        x_m[permutation] = ordered_x
        base = (
            float(rng.uniform(235.0, 285.0))
            if spec.scenario_family is ScenarioFamily.SHARED_EXIT_BOTTLENECK
            else float(rng.uniform(20.0, 80.0))
        )
        x_m += base

        speed_mps = rng.uniform(34.0, 46.0, size=count) / 3.6
        activation_step = rng.integers(0, 11, size=count, dtype=np.int64)
        activation_step[int(rng.integers(0, count))] = 0
        intended_active = np.ones(count, dtype=bool)
        deadline_step: list[int | None] = [None] * count
        if spec.deadline_s is not None:
            deadline = int(round(spec.deadline_s / dt_s))
            deadline_step = [deadline] * count

        revision_step: list[int | None] = [None] * count
        revised_goal: list[int | None] = [None] * count
        if spec.scenario_family is ScenarioFamily.DYNAMIC_GOAL_REVISION:
            revision_index = int(rng.integers(0, count))
            revision_time_s = (
                spec.goal_revision_time_s
                if spec.goal_revision_time_s is not None
                else float(rng.uniform(1.0, 4.0))
            )
            revision_step[revision_index] = max(
                int(activation_step[revision_index]) + 1,
                int(round(revision_time_s / dt_s)),
            )
            revised_goal[revision_index] = int(lane[revision_index])

        maneuvers = _planned_maneuvers(
            lane=lane,
            x_m=x_m,
            speed_mps=speed_mps,
            goal_lane=goal_lane,
            activation_step=activation_step,
            active_count=count,
        )
        edges = stress_dependency_edges(
            maneuvers,
            conflict_distance_m=spec.conflict_distance_m,
        )
        if not dependency_graph_connected(count, edges):
            continue
        if _initial_same_lane_state_safe(
            lane,
            x_m,
            speed_mps,
            minimum_clearance_m=safety_config.robust_clearance_m,
            vehicle_length_m=safety_config.vehicle_length_m,
            ttc_threshold_s=safety_config.ttc_threshold_s,
        ):
            return GeneratedScenario(
                spec=spec,
                lane=tuple(float(value) for value in lane),
                x_m=tuple(float(value) for value in x_m),
                speed_mps=tuple(float(value) for value in speed_mps),
                goal_lane=tuple(int(value) for value in goal_lane),
                goal_is_exit=tuple(bool(value) for value in goal_is_exit),
                intended_active=tuple(bool(value) for value in intended_active),
                activation_step=tuple(int(value) for value in activation_step),
                deadline_step=tuple(deadline_step),
                planned_conflict_edges=edges,
                goal_revision_step=tuple(revision_step),
                revised_goal_lane=tuple(revised_goal),
            )
    raise RuntimeError("unable to generate a connected randomized stress scenario")


def dependency_graph_connected(
    vehicle_count: int,
    edges: Iterable[Tuple[int, int]],
) -> bool:
    adjacency = [set() for _ in range(vehicle_count)]
    for first, second in edges:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == vehicle_count


def _initial_same_lane_state_safe(
    lane: np.ndarray,
    x_m: np.ndarray,
    speed_mps: np.ndarray,
    *,
    minimum_clearance_m: float,
    vehicle_length_m: float,
    ttc_threshold_s: float,
) -> bool:
    """Reject initial states that would immediately activate the shared shield."""
    for first in range(len(lane)):
        for second in range(first + 1, len(lane)):
            if int(lane[first]) != int(lane[second]):
                continue
            if x_m[first] <= x_m[second]:
                rear, front = first, second
            else:
                rear, front = second, first
            clearance = max(
                abs(float(x_m[first] - x_m[second])) - vehicle_length_m,
                0.0,
            )
            if clearance < minimum_clearance_m:
                return False
            closing_speed_mps = float(speed_mps[rear] - speed_mps[front])
            if (
                closing_speed_mps > 0.05
                and clearance / closing_speed_mps < ttc_threshold_s
            ):
                return False
    return True


def stress_dependency_edges(
    maneuvers: Iterable[PlannedManeuver],
    *,
    conflict_distance_m: float,
) -> Tuple[Tuple[int, int], ...]:
    """Return close maneuver dependencies, including occupied target lanes."""
    items = tuple(maneuvers)
    edges = []
    for position, first in enumerate(items):
        for second in items[position + 1 :]:
            corridors_first = {first.source_lane, first.first_target_lane}
            corridors_second = {second.source_lane, second.first_target_lane}
            if not corridors_first.intersection(corridors_second):
                continue
            if abs(first.x_m - second.x_m) <= conflict_distance_m:
                edges.append((first.vehicle_index, second.vehicle_index))
    return tuple(edges)


def planned_conflict_edges(
    maneuvers: Iterable[PlannedManeuver],
    *,
    conflict_distance_m: float,
    ttc_threshold_s: float,
    vehicle_length_m: float,
) -> Tuple[Tuple[int, int], ...]:
    """Return maneuver conflicts satisfying the declared distance/TTC rule."""
    items = tuple(maneuvers)
    edges = []
    for position, first in enumerate(items):
        for second in items[position + 1 :]:
            semantic_conflict = (
                first.first_target_lane == second.first_target_lane
                or (
                    first.first_target_lane == second.source_lane
                    and second.first_target_lane == first.source_lane
                )
            )
            if not semantic_conflict:
                continue
            distance = abs(first.x_m - second.x_m)
            ttc = _pair_ttc(first, second, vehicle_length_m)
            if distance <= conflict_distance_m or ttc < ttc_threshold_s:
                edges.append((first.vehicle_index, second.vehicle_index))
    return tuple(edges)


def expected_conflict_edge_count(spec: ProceduralScenarioSpec) -> int:
    if spec.conflict_density is ConflictDensity.NULL:
        return 0
    if spec.conflict_density is ConflictDensity.SPARSE:
        return spec.active_vehicle_count // 2
    count = spec.active_vehicle_count
    return count * (count - 1) // 2


def _planned_maneuvers(
    *,
    lane: np.ndarray,
    x_m: np.ndarray,
    speed_mps: np.ndarray,
    goal_lane: np.ndarray,
    activation_step: np.ndarray,
    active_count: int,
) -> Tuple[PlannedManeuver, ...]:
    maneuvers = []
    for index in range(active_count):
        source = int(round(float(lane[index])))
        goal = int(goal_lane[index])
        direction = 0 if goal == source else (1 if goal > source else -1)
        maneuvers.append(
            PlannedManeuver(
                vehicle_index=index,
                source_lane=source,
                first_target_lane=source + direction,
                final_goal_lane=goal,
                x_m=float(x_m[index]),
                speed_mps=float(speed_mps[index]),
                activation_step=int(activation_step[index]),
            )
        )
    return tuple(maneuvers)


def _speeds(
    spec: ProceduralScenarioSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    speed_kmh = np.full(spec.vehicle_count, 40.0, dtype=np.float64)
    if spec.speed_differential is SpeedDifferential.LOW:
        speed_kmh[: spec.active_vehicle_count] += rng.uniform(
            -0.5,
            0.5,
            size=spec.active_vehicle_count,
        )
    elif spec.speed_differential is SpeedDifferential.MEDIUM:
        pattern = np.asarray([36.0, 44.0], dtype=np.float64)
        speed_kmh[: spec.active_vehicle_count] = np.resize(
            pattern,
            spec.active_vehicle_count,
        )
    else:
        pattern = np.asarray([30.0, 50.0], dtype=np.float64)
        speed_kmh[: spec.active_vehicle_count] = np.resize(
            pattern,
            spec.active_vehicle_count,
        )
    if spec.background_vehicle_count:
        speed_kmh[spec.active_vehicle_count :] = rng.uniform(
            38.0,
            42.0,
            size=spec.background_vehicle_count,
        )
    return speed_kmh / 3.6


def _pair_ttc(
    first: PlannedManeuver,
    second: PlannedManeuver,
    vehicle_length_m: float,
) -> float:
    if first.x_m <= second.x_m:
        rear, front = first, second
    else:
        rear, front = second, first
    closing = rear.speed_mps - front.speed_mps
    if closing <= 0.05:
        return float("inf")
    clearance = max(abs(first.x_m - second.x_m) - vehicle_length_m, 0.0)
    return clearance / closing


def _jsonable_spec(spec: ProceduralScenarioSpec) -> dict:
    payload = asdict(spec)
    for key, value in tuple(payload.items()):
        if isinstance(value, Enum):
            payload[key] = value.value
    return payload
