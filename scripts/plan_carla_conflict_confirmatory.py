#!/usr/bin/env python3
"""Create the outcome-blind CARLA conflict-active confirmatory schedule."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_carla_conflict_active_pilot import (
    FLEET_SIZES,
    HIGHWAY_TM_ROUTE,
    LONG_LAYOUT_PROFILE,
    METHODS,
    SCENARIO_FAMILIES,
    conflict_layout,
)


@dataclass(frozen=True)
class ConfirmatoryConfig:
    repetitions_per_cell: int = 21
    seed_base: int = 9_300_000
    method_order_seed: int = 2_026_091_611
    block_order_seed: int = 2_026_091_612
    bootstrap_seed: int = 2_026_091_613
    trial_timeout_sim_s: float = 60.0


def build_schedule(config: ConfirmatoryConfig) -> dict[str, Any]:
    if config.repetitions_per_cell < 1:
        raise ValueError("repetitions must be positive")
    blocks: list[dict[str, Any]] = []
    ordinal = 0
    for family in SCENARIO_FAMILIES:
        for fleet_size in FLEET_SIZES:
            for repetition in range(1, config.repetitions_per_cell + 1):
                seed = config.seed_base + ordinal
                block_id = f"{family}-n{fleet_size}-r{repetition:03d}"
                method_order = list(METHODS)
                random.Random(
                    f"{config.method_order_seed}:{block_id}"
                ).shuffle(method_order)
                priority_order = list(range(fleet_size))
                random.Random(f"priority:{seed}").shuffle(priority_order)
                blocks.append(
                    {
                        "claim_status": "draft_not_frozen",
                        "block_id": block_id,
                        "scenario_family": family,
                        "executor_pattern": family,
                        "fleet_size": fleet_size,
                        "repetition": repetition,
                        "seed": seed,
                        "method_order": method_order,
                        "priority_order": priority_order,
                        "require_initial_conflict": True,
                        "tm_route": list(HIGHWAY_TM_ROUTE),
                        **conflict_layout(
                            family, fleet_size, LONG_LAYOUT_PROFILE
                        ),
                    }
                )
                ordinal += 1
    random.Random(config.block_order_seed).shuffle(blocks)
    efficacy_blocks = len(SCENARIO_FAMILIES) * 2 * config.repetitions_per_cell
    return {
        "schema_version": "1.0-draft",
        "label": "carla_conflict_active_confirmatory_corridor_repair_v4",
        "claim_status": "draft_not_frozen",
        "map": "Town04_Opt",
        "layout_profile": LONG_LAYOUT_PROFILE,
        "methods": list(METHODS),
        "primary_method": "MIND_CAV_DETERMINISTIC",
        "primary_comparators": ["FCFS_GAP", "MAPPO_ADAPTED"],
        "ablation_method": "MIND_CAV_LEARNED",
        "scenario_families": list(SCENARIO_FAMILIES),
        "fleet_sizes": list(FLEET_SIZES),
        "primary_efficacy_fleet_sizes": [2, 4],
        "secondary_scalability_fleet_sizes": [8],
        "repetitions_per_cell": config.repetitions_per_cell,
        "block_count": len(blocks),
        "primary_efficacy_block_count": efficacy_blocks,
        "episode_count": len(blocks) * len(METHODS),
        "seed_namespace": [
            config.seed_base,
            config.seed_base + len(blocks) - 1,
        ],
        "method_order_seed": config.method_order_seed,
        "block_order_seed": config.block_order_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "mappo_actor": (
            "data/experiments/carla_aligned_mappo_candidate_20260916/"
            "actor.npz"
        ),
        "mind_ranker": "data/models/candidate_ranker.npz",
        "persist_frames": False,
        "camera_sensors_enabled": False,
        "traffic_manager_collision_avoidance": False,
        "traffic_manager_route": list(HIGHWAY_TM_ROUTE),
        "initial_target_speed_kmh": 50.0,
        "trial_timeout_sim_s": config.trial_timeout_sim_s,
        "trial_timeout_wall_s": 210.0,
        "ready_timeout_wall_s": 90.0,
        "max_command_attempts": 10,
        "simulation_execution_mode": "fixed_step",
        "simulation_step_ticks": 10,
        "lane_center_tolerance_m": 0.5,
        "lane_center_dwell_sim_s": 0.75,
        "gap_threshold_m": 5.0,
        "blocks": blocks,
    }


def validate_schedule(schedule: dict[str, Any]) -> None:
    statuses = {"draft_not_frozen", "frozen_confirmatory"}
    if schedule.get("claim_status") not in statuses:
        raise ValueError("invalid confirmatory schedule status")
    repetitions = int(schedule.get("repetitions_per_cell") or 0)
    expected_blocks = len(SCENARIO_FAMILIES) * len(FLEET_SIZES) * repetitions
    blocks = list(schedule.get("blocks") or [])
    if len(blocks) != expected_blocks:
        raise ValueError("schedule has the wrong block count")
    if int(schedule.get("episode_count") or 0) != expected_blocks * len(METHODS):
        raise ValueError("schedule has the wrong episode count")
    if list(schedule.get("methods") or []) != list(METHODS):
        raise ValueError("method set or order changed")
    if schedule.get("layout_profile") != LONG_LAYOUT_PROFILE:
        raise ValueError("confirmatory layout profile changed")
    if float(schedule.get("trial_timeout_sim_s") or 0.0) != 60.0:
        raise ValueError("confirmatory horizon must remain 60 simulation seconds")
    if schedule.get("persist_frames") is not False:
        raise ValueError("confirmatory runs must not persist frames")
    if schedule.get("camera_sensors_enabled") is not False:
        raise ValueError("confirmatory runs must not create camera sensors")
    if schedule.get("traffic_manager_route") != list(HIGHWAY_TM_ROUTE):
        raise ValueError("Traffic Manager highway route changed")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    cell_counts: dict[tuple[str, int], int] = {}
    expected_block_status = str(schedule["claim_status"])
    for block in blocks:
        block_id = str(block["block_id"])
        seed = int(block["seed"])
        if block_id in seen_ids or seed in seen_seeds:
            raise ValueError("block identifiers and seeds must be unique")
        if block.get("claim_status") != expected_block_status:
            raise ValueError(f"claim status changed in {block_id}")
        seen_ids.add(block_id)
        seen_seeds.add(seed)
        if set(block["method_order"]) != set(METHODS):
            raise ValueError(f"method coverage changed in {block_id}")
        fleet_size = int(block["fleet_size"])
        if sorted(int(value) for value in block["priority_order"]) != list(
            range(fleet_size)
        ):
            raise ValueError(f"invalid priority order in {block_id}")
        layout = conflict_layout(
            str(block["scenario_family"]), fleet_size, LONG_LAYOUT_PROFILE
        )
        for key in (
            "spawn_indices",
            "spawn_longitudinal_offsets_m",
            "commands",
        ):
            if block.get(key) != layout[key]:
                raise ValueError(f"{key} changed in {block_id}")
        if block.get("require_initial_conflict") is not True:
            raise ValueError(f"initial conflict gate missing in {block_id}")
        if block.get("tm_route") != list(HIGHWAY_TM_ROUTE):
            raise ValueError(f"Traffic Manager route changed in {block_id}")
        cell = (str(block["scenario_family"]), fleet_size)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    expected_cells = {
        (family, fleet_size)
        for family in SCENARIO_FAMILIES
        for fleet_size in FLEET_SIZES
    }
    if set(cell_counts) != expected_cells or set(cell_counts.values()) != {
        repetitions
    }:
        raise ValueError("schedule cells are incomplete or unbalanced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=21)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    schedule = build_schedule(
        ConfirmatoryConfig(repetitions_per_cell=args.repetitions)
    )
    validate_schedule(schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"blocks={schedule['block_count']} "
        f"episodes={schedule['episode_count']}"
    )


if __name__ == "__main__":
    main()
