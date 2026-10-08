"""Frozen split contract for adapted MAPPO scenario sampling."""

from __future__ import annotations

from dataclasses import dataclass

from .procedural_scenarios import (
    GeometryProfile,
    ProceduralScenarioSpec,
    ScenarioFamily,
    ScenarioSplit,
)


FLEET_SIZES = (4, 6, 8)
SCENARIO_FAMILIES = (
    ScenarioFamily.DEPENDENCY_CASCADE,
    ScenarioFamily.SHARED_EXIT_BOTTLENECK,
    ScenarioFamily.DYNAMIC_GOAL_REVISION,
    ScenarioFamily.DELAYED_COMMITMENT_MERGE,
)
CORE_CELLS = tuple(
    (family, fleet_size)
    for family in SCENARIO_FAMILIES
    for fleet_size in FLEET_SIZES
)
SPLIT_SEED_BASE = {
    ScenarioSplit.TRAIN: 2_100_000,
    ScenarioSplit.VALIDATION: 3_100_000,
    ScenarioSplit.TEST: 4_100_000,
}
SPLIT_SEED_LIMIT = {
    split: base + 900_000 for split, base in SPLIT_SEED_BASE.items()
}
DELAY_PROFILES = ((200, 0.0), (400, 0.0), (400, 0.2))


@dataclass(frozen=True)
class AdaptedScenarioCell:
    family: ScenarioFamily
    fleet_size: int


def scenario_seed(split: ScenarioSplit, ordinal: int) -> int:
    if split not in SPLIT_SEED_BASE:
        raise ValueError("adapted MAPPO requires train, validation, or test split")
    if ordinal < 0:
        raise ValueError("scenario ordinal cannot be negative")
    seed = SPLIT_SEED_BASE[split] + int(ordinal)
    if seed >= SPLIT_SEED_LIMIT[split]:
        raise ValueError("scenario ordinal exhausted the declared split namespace")
    return seed


def adapted_scenario_spec(
    split: ScenarioSplit,
    family: ScenarioFamily,
    fleet_size: int,
    ordinal: int,
) -> ProceduralScenarioSpec:
    if family not in SCENARIO_FAMILIES:
        raise ValueError("unknown adapted MAPPO scenario family")
    if fleet_size not in FLEET_SIZES:
        raise ValueError("adapted MAPPO fleet size must be 4, 6, or 8")
    seed = scenario_seed(split, ordinal)
    delay_ms = 0
    loss_probability = 0.0
    if family is ScenarioFamily.DELAYED_COMMITMENT_MERGE:
        delay_ms, loss_probability = DELAY_PROFILES[ordinal % len(DELAY_PROFILES)]
    return ProceduralScenarioSpec(
        active_vehicle_count=fleet_size,
        scenario_family=family,
        geometry_profile=GeometryProfile.RANDOMIZED_CONNECTED,
        scenario_split=split,
        seed=seed,
        communication_delay_ms=delay_ms,
        packet_loss_probability=loss_probability,
        communication_seed=seed + 17,
    )


def split_seed_ranges_disjoint() -> bool:
    intervals = sorted(
        (SPLIT_SEED_BASE[split], SPLIT_SEED_LIMIT[split])
        for split in SPLIT_SEED_BASE
    )
    return all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:]))
