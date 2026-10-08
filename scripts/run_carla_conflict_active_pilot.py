#!/usr/bin/env python3
"""Run a paired, conflict-active CARLA comparison.

This campaign compares deterministic and learned candidate ranking inside the
same repaired MIND-CAV transaction path against FCFS-GAP and the frozen,
validator-masked MAPPO actor. Camera sensors and frame persistence stay off;
each episode retains its raw trajectory, event stream, and metadata. Frozen
confirmatory schedules additionally require a source/model/environment lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marl.numpy_policy import NumpyActorPolicy
from scripts.run_carla_paired_coordination import (
    RunnerBudget,
    build_provenance,
    run_method_episode,
    sha256_file,
    sha256_json,
    write_summary,
)


METHODS = (
    "FCFS_GAP",
    "MAPPO_ADAPTED",
    "MIND_CAV_DETERMINISTIC",
    "MIND_CAV_LEARNED",
)
SCENARIO_FAMILIES = (
    "close_reciprocal",
    "contested_merge",
    "blocked_merge",
)
FLEET_SIZES = (2, 4, 8)
SEED_BASE = 8_300_000
METHOD_ORDER_SEED = 2_026_090_401
BLOCK_ORDER_SEED = 2_026_090_402
DEFAULT_LAYOUT_PROFILE = "negative_corridor_v1"
LEGACY_LONG_LAYOUT_PROFILE = "positive_long_corridor_v1"
LONG_LAYOUT_PROFILE = "positive_long_corridor_v2"
POSITIVE_LAYOUT_PROFILES = (LEGACY_LONG_LAYOUT_PROFILE, LONG_LAYOUT_PROFILE)
LAYOUT_PROFILES = (DEFAULT_LAYOUT_PROFILE, *POSITIVE_LAYOUT_PROFILES)
HIGHWAY_TM_ROUTE = ("Straight",) * 16

# Town04 road 6, ordered from lane -4 through lane -1. These calibrated
# clusters have more than 250 m of forward non-junction roadway. Cluster B is
# approximately 23 m longitudinally from A.
CLUSTER_A = (339, 340, 341, 342)
CLUSTER_B = (335, 336, 337, 338)
LONG_CLUSTER = (40, 41, 42, 43)


def command(slot: int, direction: str, offset_s: float = 0.0) -> dict[str, Any]:
    return {
        "slot": int(slot),
        "direction": str(direction),
        "issue_offset_sim_s": float(offset_s),
    }


def conflict_layout(
    family: str,
    fleet_size: int,
    layout_profile: str = DEFAULT_LAYOUT_PROFILE,
) -> dict[str, Any]:
    """Return fixed close-proximity layouts selected from the live Town04 map."""
    if fleet_size not in FLEET_SIZES:
        raise ValueError(f"unsupported fleet size: {fleet_size}")
    if layout_profile not in LAYOUT_PROFILES:
        raise ValueError(f"unsupported layout profile: {layout_profile}")

    if layout_profile in POSITIVE_LAYOUT_PROFILES:
        cluster_a = LONG_CLUSTER
        cluster_b = LONG_CLUSTER
        if layout_profile == LEGACY_LONG_LAYOUT_PROFILE:
            offset_a = -100.0
            offset_b = -75.0
        else:
            offset_a = 0.0
            offset_b = 25.0
    else:
        cluster_a = CLUSTER_A
        cluster_b = CLUSTER_B
        offset_a = 0.0
        offset_b = 0.0

    if family == "close_reciprocal":
        if fleet_size == 2:
            spawns = (cluster_a[1], cluster_a[2])
            offsets = (offset_a, offset_a)
            commands = (command(0, "left"), command(1, "right"))
        else:
            spawns = cluster_a if fleet_size == 4 else cluster_a + cluster_b
            offsets = (
                (offset_a,) * 4
                if fleet_size == 4
                else (offset_a,) * 4 + (offset_b,) * 4
            )
            commands = tuple(
                entry
                for base in range(0, fleet_size, 4)
                for entry in (
                    command(base, "left"),
                    command(base + 1, "right"),
                    command(base + 2, "left"),
                    command(base + 3, "right"),
                )
            )
    elif family == "contested_merge":
        if fleet_size == 2:
            spawns = (cluster_a[0], cluster_a[2])
            offsets = (offset_a, offset_a)
            commands = (command(0, "left"), command(1, "right"))
        elif fleet_size == 4:
            spawns = (cluster_a[0], cluster_a[2], cluster_b[0], cluster_b[2])
            offsets = (offset_a, offset_a, offset_b, offset_b)
            commands = tuple(
                entry
                for base in (0, 2)
                for entry in (command(base, "left"), command(base + 1, "right"))
            )
        else:
            spawns = cluster_a + cluster_b
            offsets = (offset_a,) * 4 + (offset_b,) * 4
            commands = tuple(
                entry
                for base in (0, 4)
                for entry in (
                    command(base, "left"),
                    command(base + 2, "right"),
                    command(base + 1, "left"),
                    command(base + 3, "right"),
                )
            )
    elif family == "blocked_merge":
        if fleet_size == 2:
            spawns = (cluster_a[2], cluster_a[1])
            offsets = (offset_a, offset_a)
            commands = (command(0, "right"),)
        elif fleet_size == 4:
            spawns = (cluster_a[2], cluster_a[1], cluster_b[2], cluster_b[1])
            offsets = (offset_a, offset_a, offset_b, offset_b)
            commands = (command(0, "right"), command(2, "right"))
        else:
            spawns = cluster_a + cluster_b
            offsets = (offset_a,) * 4 + (offset_b,) * 4
            commands = (
                command(0, "left"),
                command(3, "right"),
                command(4, "left"),
                command(7, "right"),
            )
    else:
        raise ValueError(f"unknown scenario family: {family}")

    if layout_profile in POSITIVE_LAYOUT_PROFILES:
        opposite = {"left": "right", "right": "left"}
        commands = tuple(
            {**entry, "direction": opposite[str(entry["direction"])]}
            for entry in commands
        )

    return {
        "spawn_indices": list(spawns),
        "spawn_longitudinal_offsets_m": list(offsets),
        "commands": list(commands),
    }


def build_schedule(
    repetitions: int,
    layout_profile: str = DEFAULT_LAYOUT_PROFILE,
    trial_timeout_sim_s: float = 60.0,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < trial_timeout_sim_s <= 60.0:
        raise ValueError("trial timeout must be in (0, 60] simulation seconds")
    if layout_profile not in LAYOUT_PROFILES:
        raise ValueError(f"unsupported layout profile: {layout_profile}")
    blocks = []
    ordinal = 0
    for family in SCENARIO_FAMILIES:
        for fleet_size in FLEET_SIZES:
            for repetition in range(1, repetitions + 1):
                seed = SEED_BASE + ordinal
                block_id = f"{family}-n{fleet_size}-r{repetition:03d}"
                method_order = list(METHODS)
                random.Random(f"{METHOD_ORDER_SEED}:{block_id}").shuffle(method_order)
                priority_order = list(range(fleet_size))
                random.Random(f"priority:{seed}").shuffle(priority_order)
                layout = conflict_layout(family, fleet_size, layout_profile)
                blocks.append(
                    {
                        "claim_status": "exploratory_pilot",
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
                        **layout,
                    }
                )
                ordinal += 1
    random.Random(BLOCK_ORDER_SEED).shuffle(blocks)
    return {
        "schema_version": "1.0-exploratory",
        "label": "carla_conflict_active_ranker_pilot",
        "claim_status": "exploratory_pilot",
        "map": "Town04_Opt",
        "layout_profile": layout_profile,
        "methods": list(METHODS),
        "scenario_families": list(SCENARIO_FAMILIES),
        "fleet_sizes": list(FLEET_SIZES),
        "repetitions_per_cell": repetitions,
        "block_count": len(blocks),
        "episode_count": len(blocks) * len(METHODS),
        "seed_namespace": [SEED_BASE, SEED_BASE + len(blocks) - 1],
        "method_order_seed": METHOD_ORDER_SEED,
        "block_order_seed": BLOCK_ORDER_SEED,
        "mappo_actor": (
            "data/experiments/adapted_mappo_full_20260826/"
            "seed_2026082601/actor.npz"
        ),
        "mind_ranker": "data/models/candidate_ranker.npz",
        "persist_frames": False,
        "camera_sensors_enabled": False,
        "traffic_manager_route": list(HIGHWAY_TM_ROUTE),
        "trial_timeout_sim_s": float(trial_timeout_sim_s),
        "trial_timeout_wall_s": 210.0,
        "ready_timeout_wall_s": 90.0,
        "max_command_attempts": 10,
        "initial_target_speed_kmh": 50.0,
        "simulation_execution_mode": "fixed_step",
        "simulation_step_ticks": 10,
        "blocks": blocks,
    }


def validate_schedule(schedule: dict[str, Any]) -> None:
    blocks = list(schedule.get("blocks") or [])
    repetitions = int(schedule.get("repetitions_per_cell") or 0)
    expected = len(SCENARIO_FAMILIES) * len(FLEET_SIZES) * repetitions
    if schedule.get("claim_status") not in {
        "exploratory_pilot",
        "frozen_confirmatory",
    }:
        raise ValueError(
            "conflict-active execution requires an exploratory or frozen schedule"
        )
    if len(blocks) != expected or int(schedule.get("block_count") or 0) != expected:
        raise ValueError("schedule has the wrong number of blocks")
    if schedule.get("persist_frames") is not False:
        raise ValueError("frame persistence must remain disabled")
    if schedule.get("camera_sensors_enabled") is not False:
        raise ValueError("camera sensors must remain disabled")
    if schedule.get("traffic_manager_route") != list(HIGHWAY_TM_ROUTE):
        raise ValueError("Traffic Manager highway route changed")
    if float(schedule.get("initial_target_speed_kmh") or 0.0) != 50.0:
        raise ValueError("initial target speed must remain 50 km/h")
    timeout_sim_s = float(schedule.get("trial_timeout_sim_s") or 0.0)
    if not 0.0 < timeout_sim_s <= 60.0:
        raise ValueError("trial timeout must be in (0, 60] simulation seconds")
    if schedule.get("simulation_execution_mode") != "fixed_step":
        raise ValueError("conflict-active execution must use fixed stepping")
    if int(schedule.get("simulation_step_ticks") or 0) != 10:
        raise ValueError("fixed-step cadence must remain 10 ticks")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    cells: dict[tuple[str, int], int] = {}
    layout_profile = str(
        schedule.get("layout_profile") or DEFAULT_LAYOUT_PROFILE
    )
    if layout_profile not in LAYOUT_PROFILES:
        raise ValueError(f"unsupported layout profile: {layout_profile}")
    for block in blocks:
        block_id = str(block["block_id"])
        seed = int(block["seed"])
        if block_id in seen_ids or seed in seen_seeds:
            raise ValueError("block ids and seeds must be unique")
        seen_ids.add(block_id)
        seen_seeds.add(seed)
        if set(block["method_order"]) != set(METHODS):
            raise ValueError(f"invalid method order in {block_id}")
        layout = conflict_layout(
            str(block["scenario_family"]),
            int(block["fleet_size"]),
            layout_profile,
        )
        if block["spawn_indices"] != layout["spawn_indices"]:
            raise ValueError(f"spawn layout changed in {block_id}")
        if block["commands"] != layout["commands"]:
            raise ValueError(f"command layout changed in {block_id}")
        if list(block.get("spawn_longitudinal_offsets_m") or [0.0] * int(block["fleet_size"])) != layout["spawn_longitudinal_offsets_m"]:
            raise ValueError(f"spawn offsets changed in {block_id}")
        if block.get("require_initial_conflict") is not True:
            raise ValueError(f"initial conflict assertion missing in {block_id}")
        if block.get("tm_route") != list(HIGHWAY_TM_ROUTE):
            raise ValueError(f"Traffic Manager route changed in {block_id}")
        key = (str(block["scenario_family"]), int(block["fleet_size"]))
        cells[key] = cells.get(key, 0) + 1
    expected_cells = {
        (family, fleet_size)
        for family in SCENARIO_FAMILIES
        for fleet_size in FLEET_SIZES
    }
    if set(cells) != expected_cells or set(cells.values()) != {repetitions}:
        raise ValueError("scenario/fleet cells are incomplete or unbalanced")


def configure_ranker(
    client: httpx.Client, api_base: str, method: str
) -> dict[str, Any] | None:
    variants = {
        "MIND_CAV_DETERMINISTIC": "deterministic",
        "MIND_CAV_LEARNED": "learned",
    }
    variant = variants.get(method)
    if variant is None:
        return None
    response = client.post(
        f"{api_base}/mec/v2/ranker", json={"variant": variant}
    )
    response.raise_for_status()
    status = response.json()
    expected_name = f"constrained-{variant}-ranker"
    if status.get("proposer") != expected_name:
        raise RuntimeError(
            f"ranker switch failed: expected {expected_name}, got {status}"
        )
    if status.get("liveness_preparation") is not True:
        raise RuntimeError("ranker switch disabled liveness preparation")
    if variant == "learned" and status.get("ranker_model_loaded") is not True:
        raise RuntimeError("learned ranker checkpoint did not load")
    return status


def validate_spawn_geometry(
    client: httpx.Client,
    api_base: str,
    blocks: Iterable[dict[str, Any]],
    layout_profile: str = DEFAULT_LAYOUT_PROFILE,
) -> dict[str, Any]:
    """Fail before execution if a selected spawn is not on usable highway."""
    response = client.get(f"{api_base}/spawns", timeout=120.0)
    response.raise_for_status()
    rows = {int(row["index"]): row for row in response.json()["spawns"]}
    selected_indices = sorted(
        {
            int(spawn_index)
            for block in blocks
            for spawn_index in block["spawn_indices"]
        }
    )
    selected = []
    allowed_lane_ids = (
        {3, 4, 5, 6}
        if layout_profile in POSITIVE_LAYOUT_PROFILES
        else {-4, -3, -2, -1}
    )
    for spawn_index in selected_indices:
        row = rows.get(spawn_index)
        if row is None:
            raise RuntimeError(f"spawn {spawn_index} is unavailable")
        if row.get("is_junction"):
            raise RuntimeError(f"spawn {spawn_index} lies in a junction")
        if int(row.get("lane_id") or 0) not in allowed_lane_ids:
            raise RuntimeError(f"spawn {spawn_index} is outside the four-lane corridor")
        if float(row.get("forward_non_junction_m") or 0.0) < 150.0:
            raise RuntimeError(
                f"spawn {spawn_index} has insufficient non-junction road"
            )
        selected.append(row)
    if len({int(row["road_id"]) for row in selected}) != 1:
        raise RuntimeError("selected conflict-active spawns do not share one road")
    return {
        "minimum_forward_non_junction_m": min(
            float(row["forward_non_junction_m"]) for row in selected
        ),
        "road_ids": sorted({int(row["road_id"]) for row in selected}),
        "selected_spawns": selected,
        "selected_spawns_sha256": sha256_json(selected),
    }


def selected_values(raw: str, allowed: Iterable[str]) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown selections: {unknown}")
    return values


def write_frozen_schedule(schedule: dict[str, Any], path: Path) -> str:
    """Validate and persist an immutable-before-execution schedule artifact."""
    validate_schedule(schedule)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def stable_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return fields that must remain identical when resuming a campaign."""
    return {
        key: value
        for key, value in provenance.items()
        if key != "created_at_unix_s"
    }


def audit_campaign_artifacts(
    output: Path,
    blocks: Iterable[dict[str, Any]],
    methods: Iterable[str],
) -> dict[str, Any]:
    """Verify that every scheduled episode has complete non-visual artifacts."""
    required = ("metadata.json", "trajectory.jsonl", "events.jsonl")
    selected_methods = list(methods)
    missing: list[str] = []
    episode_count = 0
    for block in blocks:
        for method in selected_methods:
            episode_count += 1
            run_dir = output / str(block["block_id"]) / method
            for filename in required:
                path = run_dir / filename
                if not path.is_file():
                    missing.append(str(path.relative_to(output)))
    image_files = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if missing or image_files:
        raise ValueError(
            "campaign artifact audit failed: "
            f"missing={missing}, image_files={image_files}"
        )
    return {
        "passed": True,
        "episode_count": episode_count,
        "required_artifacts_per_episode": list(required),
        "missing_artifacts": [],
        "persisted_image_files": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--trial-timeout-s", type=float, default=60.0)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--confirmatory-lock", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument(
        "--layout-profile",
        choices=LAYOUT_PROFILES,
        default=DEFAULT_LAYOUT_PROFILE,
    )
    parser.add_argument(
        "--write-schedule",
        type=Path,
        help="validate, write the schedule, print its SHA-256, and exit",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    parser.add_argument(
        "--fleet-sizes", default=",".join(str(value) for value in FLEET_SIZES)
    )
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/carla_conflict_active_ranker_pilot_20260904"
        ),
    )
    parser.add_argument(
        "--actor",
        type=Path,
        default=Path(
            "data/experiments/adapted_mappo_full_20260826/"
            "seed_2026082601/actor.npz"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a checksum-matching campaign without rerunning completed episodes",
    )
    args = parser.parse_args()

    if args.schedule is not None:
        schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
        if (
            args.repetitions is not None
            and args.repetitions
            != int(schedule.get("repetitions_per_cell") or 0)
        ):
            raise ValueError("--repetitions differs from the frozen schedule")
    else:
        schedule = build_schedule(
            args.repetitions or 1,
            layout_profile=args.layout_profile,
            trial_timeout_sim_s=args.trial_timeout_s,
        )
    validate_schedule(schedule)
    claim_status = str(schedule["claim_status"])
    is_confirmatory = claim_status == "frozen_confirmatory"
    lock_arguments = (
        args.confirmatory_lock,
        args.registration,
        args.environment_lock,
    )
    if is_confirmatory:
        from scripts.carla_conflict_confirmatory_freeze import (
            validate_lock as validate_frozen_lock,
        )

        if args.schedule is None or any(value is None for value in lock_arguments):
            raise ValueError(
                "confirmatory execution requires --schedule, --confirmatory-lock, "
                "--registration, and --environment-lock"
            )
        frozen_lock = validate_frozen_lock(
            schedule=args.schedule,
            registration=args.registration,
            environment_lock=args.environment_lock,
            lock_path=args.confirmatory_lock,
        )
    else:
        if any(value is not None for value in lock_arguments):
            raise ValueError("exploratory execution cannot use a confirmatory lock")
        frozen_lock = None
    if args.write_schedule is not None:
        digest = write_frozen_schedule(schedule, args.write_schedule)
        print(f"schedule_sha256={digest}")
        return
    methods = selected_values(args.methods, METHODS)
    families = set(selected_values(args.families, SCENARIO_FAMILIES))
    fleet_sizes = {
        int(value) for value in selected_values(args.fleet_sizes, map(str, FLEET_SIZES))
    }
    blocks = [
        block
        for block in schedule["blocks"]
        if block["scenario_family"] in families
        and int(block["fleet_size"]) in fleet_sizes
    ]
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
    if not blocks:
        raise ValueError("no blocks selected")
    if is_confirmatory:
        protocol_deviations = {
            "methods": methods != list(schedule["methods"]),
            "families": families != set(schedule["scenario_families"]),
            "fleet_sizes": fleet_sizes != set(schedule["fleet_sizes"]),
            "max_blocks": args.max_blocks is not None,
            "overwrite": args.overwrite,
        }
        changed = sorted(
            key for key, changed in protocol_deviations.items() if changed
        )
        if changed:
            raise ValueError(f"confirmatory protocol deviation: {changed}")
    if args.output.exists():
        if args.overwrite:
            shutil.rmtree(args.output)
            args.output.mkdir(parents=True)
        elif not args.resume:
            raise FileExistsError(args.output)
        else:
            persisted_schedule = args.output / "schedule.json"
            if not persisted_schedule.exists():
                raise ValueError("resume output has no persisted schedule")
            observed_schedule = json.loads(
                persisted_schedule.read_text(encoding="utf-8")
            )
            if sha256_json(observed_schedule) != sha256_json(schedule):
                raise ValueError("resume schedule checksum mismatch")
    else:
        args.output.mkdir(parents=True)
    schedule_path = args.output / "schedule.json"
    if not schedule_path.exists():
        schedule_path.write_text(
            json.dumps(schedule, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    actor_path = (
        (REPO_ROOT / args.actor).resolve()
        if not args.actor.is_absolute()
        else args.actor
    )
    if is_confirmatory:
        expected_actor = (REPO_ROOT / schedule["mappo_actor"]).resolve()
        if actor_path != expected_actor:
            raise ValueError("MAPPO actor path differs from the frozen schedule")
    actor = NumpyActorPolicy.load(str(actor_path))
    provenance = build_provenance(actor_path)
    provenance["source_files_sha256"][
        "scripts/run_carla_conflict_active_pilot.py"
    ] = sha256_file(Path(__file__))
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        observed_provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        if stable_provenance(observed_provenance) != stable_provenance(
            provenance
        ):
            raise ValueError("resume provenance differs from the current source state")
    else:
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    budget = RunnerBudget(
        trial_timeout_sim_s=float(schedule["trial_timeout_sim_s"]),
        max_command_attempts=int(schedule["max_command_attempts"]),
    )
    if budget.simulation_step_ticks != int(schedule["simulation_step_ticks"]):
        raise ValueError("runner step cadence differs from schedule")
    rows: list[dict[str, Any]] = []
    infrastructure_failures = sorted(
        args.output.glob(
            "_infrastructure_attempts/*/*/attempt_*/infrastructure_failure.json"
        )
    )
    ranker_preflights: dict[str, dict[str, Any]] = {}
    consecutive_infrastructure_failures = 0
    api_base = args.api_base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{api_base}/health")
        health.raise_for_status()
        spawn_preflight = validate_spawn_geometry(
            client,
            api_base,
            blocks,
            layout_profile=str(
                schedule.get("layout_profile") or DEFAULT_LAYOUT_PROFILE
            ),
        )
        for block in blocks:
            for method in block["method_order"]:
                if method not in methods:
                    continue
                run_dir = args.output / block["block_id"] / method
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    row = json.loads(metadata_path.read_text(encoding="utf-8"))
                    expected = {
                        "block_id": str(block["block_id"]),
                        "scenario_family": str(block["scenario_family"]),
                        "fleet_size": int(block["fleet_size"]),
                        "seed": int(block["seed"]),
                        "method": method,
                    }
                    mismatches = {
                        key: (row.get(key), value)
                        for key, value in expected.items()
                        if row.get(key) != value
                    }
                    if mismatches:
                        raise ValueError(
                            f"resume metadata mismatch in {metadata_path}: {mismatches}"
                        )
                    rows.append(row)
                    write_summary(args.output / "summary.csv", rows)
                    print(
                        f"[SKIP] {block['block_id']} {method} already complete",
                        flush=True,
                    )
                    continue
                print(f"[RUN] {block['block_id']} {method}", flush=True)
                while True:
                    try:
                        ranker_status = configure_ranker(client, api_base, method)
                        if ranker_status is not None:
                            ranker_preflights[method] = ranker_status
                        row = run_method_episode(
                            client,
                            api_base,
                            block,
                            method,
                            run_dir,
                            actor if method == "MAPPO_ADAPTED" else None,
                            budget,
                            archive=False,
                            enable_cameras=False,
                        )
                    except Exception as exc:
                        failure_index = 1 + len(
                            list(
                                (
                                    args.output
                                    / "_infrastructure_attempts"
                                    / str(block["block_id"])
                                    / method
                                ).glob("attempt_*")
                            )
                        )
                        retained_dir = (
                            args.output
                            / "_infrastructure_attempts"
                            / str(block["block_id"])
                            / method
                            / f"attempt_{failure_index:03d}"
                        )
                        retained_dir.parent.mkdir(parents=True, exist_ok=True)
                        if run_dir.exists():
                            shutil.move(str(run_dir), str(retained_dir))
                        else:
                            retained_dir.mkdir()
                        failure = {
                            "block_id": block["block_id"],
                            "method": method,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "recorded_at_s": time.time(),
                            "replacement_required": True,
                        }
                        failure_path = retained_dir / "infrastructure_failure.json"
                        failure_path.write_text(
                            json.dumps(failure, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        infrastructure_failures.append(failure_path)
                        consecutive_infrastructure_failures += 1
                        print(
                            f"[INFRASTRUCTURE ERROR] {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if consecutive_infrastructure_failures >= 3:
                            raise RuntimeError(
                                "three consecutive infrastructure failures; "
                                "campaign stopped for diagnosis"
                            ) from exc
                        print("[RETRY] identical scheduled episode", flush=True)
                        continue
                    consecutive_infrastructure_failures = 0
                    print(
                        f"[DONE] status={row['status']} "
                        f"safe={row['safe_task_success']} "
                        f"conflicts={row['initial_conflict_count']}",
                        flush=True,
                    )
                    break
                rows.append(row)
                write_summary(args.output / "summary.csv", rows)

    completed_blocks = {
        block["block_id"]
        for block in blocks
        if all(
            any(
                row.get("block_id") == block["block_id"]
                and row.get("method") == method
                for row in rows
            )
            for method in methods
        )
    }
    artifact_audit = audit_campaign_artifacts(args.output, blocks, methods)
    manifest = {
        "schema_version": "1.0",
        "claim_status": claim_status,
        "schedule_sha256": sha256_json(schedule),
        "provenance_sha256": sha256_file(provenance_path),
        "actor": str(actor_path),
        "actor_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "methods": methods,
        "rows": len(rows),
        "blocks": len(completed_blocks),
        "scheduled_blocks": len(blocks),
        "ranker_preflights": ranker_preflights,
        "confirmatory_lock": (
            None
            if frozen_lock is None
            else {
                "path": str(args.confirmatory_lock.resolve()),
                "sha256": sha256_file(args.confirmatory_lock),
                "registration_freeze_commit": frozen_lock["registration"][
                    "freeze_commit"
                ],
            }
        ),
        "infrastructure_failure_attempts": len(infrastructure_failures),
        "spawn_preflight": spawn_preflight,
        "persist_frames": False,
        "camera_sensors_enabled": False,
        "raw_episode_artifacts": ["trajectory.jsonl", "events.jsonl", "metadata.json"],
        "artifact_audit": artifact_audit,
        "runner_budget": budget.__dict__,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
