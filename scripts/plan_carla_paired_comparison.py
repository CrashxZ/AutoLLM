#!/usr/bin/env python3
"""Plan the outcome-blind paired CARLA coordination comparison.

This script creates only the randomized schedule and a priori precision/power
calculations.  It never reads experiment outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binom, norm


METHODS = (
    "IA",
    "FCFS_QUEUE",
    "FCFS_GAP",
    "MAPPO_ADAPTED",
    "MIND_CAV",
)
SCENARIO_FAMILIES = (
    "parallel_control",
    "staggered_merge",
    "reciprocal_exchange",
)
FLEET_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class PlanningConfig:
    repetitions_per_cell: int = 45
    test_seed_base: int = 7_100_000
    method_order_seed: int = 2_026_082_701
    block_order_seed: int = 2_026_082_704
    power_seed: int = 2_026_082_702
    power_simulations: int = 100_000
    primary_risk_difference: float = 0.10
    primary_alpha: float = 0.05
    primary_comparisons: int = 2
    noninferiority_margin: float = 0.05


def _two_sided_exact_mcnemar_reject(
    mind_only: np.ndarray,
    baseline_only: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    discordant = mind_only + baseline_only
    lower = binom.cdf(mind_only, discordant, 0.5)
    upper = binom.sf(mind_only - 1, discordant, 0.5)
    p_value = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
    return p_value < alpha


def simulated_mcnemar_power(
    *,
    block_count: int,
    risk_difference: float,
    discordance: float,
    alpha: float,
    simulations: int,
    seed: int,
) -> float:
    """Monte Carlo power for an exact paired binary superiority test."""
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must be in (0, 1]")
    if abs(risk_difference) > discordance:
        raise ValueError("absolute risk difference cannot exceed discordance")
    p_mind_only = (discordance + risk_difference) / 2.0
    p_baseline_only = (discordance - risk_difference) / 2.0
    probabilities = (p_mind_only, p_baseline_only, 1.0 - discordance)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(block_count, probabilities, size=simulations)
    rejected = _two_sided_exact_mcnemar_reject(
        draws[:, 0], draws[:, 1], alpha=alpha
    )
    return float(np.mean(rejected))


def simultaneous_risk_difference_half_width(
    *,
    block_count: int,
    discordance: float,
    risk_difference: float = 0.0,
    family_alpha: float = 0.05,
    comparisons: int = 2,
) -> float:
    """Normal-approximation planning width for paired risk differences.

    Final inference uses the pre-specified stratified paired bootstrap.  This
    closed-form quantity is used only to select a conservative run count.
    """
    if block_count < 1:
        raise ValueError("block_count must be positive")
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    variance = discordance - risk_difference**2
    if variance < 0.0:
        raise ValueError("discordance is incompatible with risk_difference")
    quantile = norm.ppf(1.0 - family_alpha / (2.0 * comparisons))
    return float(quantile * math.sqrt(variance / block_count))


def build_blocks(config: PlanningConfig) -> list[dict]:
    blocks: list[dict] = []
    ordinal = 0
    for family in SCENARIO_FAMILIES:
        for fleet_size in FLEET_SIZES:
            for repetition in range(1, config.repetitions_per_cell + 1):
                seed = config.test_seed_base + ordinal
                block_id = f"{family}-n{fleet_size}-r{repetition:03d}"
                method_order = list(METHODS)
                random.Random(
                    f"{config.method_order_seed}:{block_id}"
                ).shuffle(method_order)
                priority_order = list(range(fleet_size))
                random.Random(f"priority:{seed}").shuffle(priority_order)
                blocks.append(
                    {
                        "block_id": block_id,
                        "scenario_family": family,
                        "executor_pattern": (
                            "parallel"
                            if family == "parallel_control"
                            else "reciprocal"
                            if family == "reciprocal_exchange"
                            else family
                        ),
                        "fleet_size": fleet_size,
                        "repetition": repetition,
                        "seed": seed,
                        "method_order": method_order,
                        "priority_order": priority_order,
                    }
                )
                ordinal += 1
    random.Random(config.block_order_seed).shuffle(blocks)
    return blocks


def build_schedule(config: PlanningConfig) -> dict:
    blocks = build_blocks(config)
    return {
        "schema_version": "1.0-draft",
        "label": "carla_paired_coordination_confirmatory",
        "claim_status": "draft_not_frozen",
        "map": "Town04",
        "mappo_actor": (
            "data/experiments/adapted_mappo_full_20260826/"
            "seed_2026082601/actor.npz"
        ),
        "mind_ranker": "data/models/candidate_ranker.npz",
        "methods": list(METHODS),
        "scenario_families": list(SCENARIO_FAMILIES),
        "fleet_sizes": list(FLEET_SIZES),
        "repetitions_per_cell": config.repetitions_per_cell,
        "block_count": len(blocks),
        "episode_count": len(blocks) * len(METHODS),
        "test_seed_namespace": [
            config.test_seed_base,
            config.test_seed_base + len(blocks) - 1,
        ],
        "method_order_seed": config.method_order_seed,
        "block_order_seed": config.block_order_seed,
        "persist_frames": False,
        "camera_sensors_enabled": False,
        "traffic_manager_collision_avoidance": False,
        "flow_speed_kmh": 50.0,
        "trial_timeout_sim_s": 60.0,
        "trial_timeout_wall_s": 210.0,
        "ready_timeout_wall_s": 90.0,
        "max_command_attempts": 10,
        "lane_center_tolerance_m": 0.5,
        "lane_center_dwell_sim_s": 0.75,
        "gap_threshold_m": 5.0,
        "primary_comparisons": [
            ["MIND_CAV", "MAPPO_ADAPTED"],
            ["MIND_CAV", "FCFS_GAP"],
        ],
        "noninferiority_margin": config.noninferiority_margin,
        "blocks": blocks,
    }


def validate_schedule(schedule: dict) -> None:
    blocks = schedule.get("blocks") or []
    expected_blocks = (
        len(SCENARIO_FAMILIES)
        * len(FLEET_SIZES)
        * int(schedule["repetitions_per_cell"])
    )
    if len(blocks) != expected_blocks:
        raise ValueError("schedule has the wrong number of blocks")
    if schedule.get("claim_status") not in {
        "draft_not_frozen",
        "exploratory_pilot",
        "frozen_confirmatory",
    }:
        raise ValueError("schedule has an invalid claim status")
    if schedule.get("persist_frames") is not False:
        raise ValueError("confirmatory runs must not persist camera frames")
    if schedule.get("camera_sensors_enabled") is not False:
        raise ValueError("paired runs require the frozen headless sensor configuration")
    if schedule.get("traffic_manager_collision_avoidance") is not False:
        raise ValueError("Traffic Manager collision avoidance would mask outcomes")
    if int(schedule.get("max_command_attempts") or 0) <= 0:
        raise ValueError("max_command_attempts must be positive")
    seeds = [int(block["seed"]) for block in blocks]
    block_ids = [str(block["block_id"]) for block in blocks]
    if len(seeds) != len(set(seeds)) or len(block_ids) != len(set(block_ids)):
        raise ValueError("block seeds and identifiers must be unique")
    cell_counts: dict[tuple[str, int], int] = {}
    for block in blocks:
        if tuple(sorted(block["method_order"])) != tuple(sorted(METHODS)):
            raise ValueError(f"invalid method order in {block['block_id']}")
        fleet_size = int(block["fleet_size"])
        if sorted(int(value) for value in block["priority_order"]) != list(
            range(fleet_size)
        ):
            raise ValueError(f"invalid priority order in {block['block_id']}")
        key = (str(block["scenario_family"]), fleet_size)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    expected_cells = {
        (family, fleet_size)
        for family in SCENARIO_FAMILIES
        for fleet_size in FLEET_SIZES
    }
    if set(cell_counts) != expected_cells:
        raise ValueError("schedule is missing a scenario/fleet cell")
    if set(cell_counts.values()) != {int(schedule["repetitions_per_cell"])}:
        raise ValueError("scenario/fleet cells are unbalanced")


def power_rows(config: PlanningConfig, block_counts: Iterable[int]) -> list[dict]:
    rows: list[dict] = []
    for block_count in block_counts:
        for discordance in (0.10, 0.15, 0.20, 0.30):
            rows.append(
                {
                    "block_count": int(block_count),
                    "discordance": discordance,
                    "risk_difference": config.primary_risk_difference,
                    "mcnemar_power": simulated_mcnemar_power(
                        block_count=int(block_count),
                        risk_difference=config.primary_risk_difference,
                        discordance=discordance,
                        alpha=config.primary_alpha,
                        simulations=config.power_simulations,
                        seed=config.power_seed
                        + int(block_count) * 100
                        + int(discordance * 100),
                    ),
                    "simultaneous_half_width_at_null": (
                        simultaneous_risk_difference_half_width(
                            block_count=int(block_count),
                            discordance=discordance,
                            family_alpha=config.primary_alpha,
                            comparisons=config.primary_comparisons,
                        )
                    ),
                }
            )
    return rows


def write_power_outputs(output_dir: Path, config: PlanningConfig) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = power_rows(config, (216, 324, 405))
    with (output_dir / "power_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "claim_scope": "a priori planning only; no experiment outcomes read",
        "config": config.__dict__,
        "rows": rows,
    }
    (output_dir / "power_sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule-output",
        type=Path,
        default=Path("experiments/carla_paired_comparison_draft.json"),
    )
    parser.add_argument(
        "--power-output",
        type=Path,
        default=Path("results/carla_paired_comparison_power_20260827"),
    )
    parser.add_argument("--repetitions-per-cell", type=int, default=45)
    args = parser.parse_args()

    config = PlanningConfig(repetitions_per_cell=args.repetitions_per_cell)
    schedule = build_schedule(config)
    validate_schedule(schedule)
    args.schedule_output.parent.mkdir(parents=True, exist_ok=True)
    args.schedule_output.write_text(
        json.dumps(schedule, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = write_power_outputs(args.power_output, config)
    selected = next(
        row
        for row in rows
        if row["block_count"] == schedule["block_count"]
        and row["discordance"] == 0.20
    )
    print(
        f"planned {schedule['block_count']} paired blocks / "
        f"{schedule['episode_count']} episodes; "
        f"10-point power at 20% discordance={selected['mcnemar_power']:.3f}; "
        f"simultaneous half-width={selected['simultaneous_half_width_at_null']:.4f}"
    )


if __name__ == "__main__":
    main()
