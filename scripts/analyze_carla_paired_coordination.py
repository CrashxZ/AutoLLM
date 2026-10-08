#!/usr/bin/env python3
"""Independently replay and audit paired CARLA coordination runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


CENTER_THRESHOLD_M = 0.5
SETTLING_TIME_SIM_S = 0.75
GAP_THRESHOLD_M = 5.0
TRIAL_TIMEOUT_SIM_S = 60.0
PAIRING_S_TOLERANCE_M = 1.5
PAIRING_SPEED_TOLERANCE_KMH = 1.5
MIND_METHODS = {
    "MIND_CAV",
    "MIND_CAV_DETERMINISTIC",
    "MIND_CAV_LEARNED",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def unique_trajectory_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        frames = [
            int(vehicle.get("sim_frame") or 0)
            for vehicle in (row.get("vehicles") or {}).values()
        ]
        if frames:
            by_frame[max(frames)] = row
    return [by_frame[frame] for frame in sorted(by_frame)]


def occupied_lanes(vehicle: Mapping[str, Any]) -> set[int]:
    raw = vehicle.get("occupied_lane_ids") or (
        vehicle.get("lane_change") or {}
    ).get("occupied_lane_ids")
    if raw:
        return {int(value) for value in raw}
    lane_id = vehicle.get("lane_id")
    return set() if lane_id is None else {int(lane_id)}


def replay_gap_metrics(
    trajectory: Iterable[dict[str, Any]],
    threshold_m: float = GAP_THRESHOLD_M,
) -> tuple[Optional[float], int]:
    minimum_shared: Optional[float] = None
    violation_samples = 0
    for outer in unique_trajectory_rows(trajectory):
        vehicles = list((outer.get("vehicles") or {}).values())
        frame_minimum: Optional[float] = None
        for index, first in enumerate(vehicles):
            for second in vehicles[index + 1 :]:
                if not occupied_lanes(first).intersection(occupied_lanes(second)):
                    continue
                try:
                    distance = math.hypot(
                        float(first["pose"]["x"]) - float(second["pose"]["x"]),
                        float(first["pose"]["y"]) - float(second["pose"]["y"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                frame_minimum = (
                    distance if frame_minimum is None else min(frame_minimum, distance)
                )
        if frame_minimum is not None:
            minimum_shared = (
                frame_minimum
                if minimum_shared is None
                else min(minimum_shared, frame_minimum)
            )
            violation_samples += int(frame_minimum < threshold_m)
    return minimum_shared, violation_samples


def replay_collision_count(trajectory: Iterable[dict[str, Any]]) -> int:
    maximum = 0
    for outer in trajectory:
        maximum = max(
            maximum,
            sum(
                int(vehicle.get("collision_count") or 0)
                for vehicle in (outer.get("vehicles") or {}).values()
            ),
        )
    return maximum


def replay_command(
    trajectory: Iterable[dict[str, Any]], command: Mapping[str, Any]
) -> dict[str, Any]:
    veh_id = str(command["veh_id"])
    target_lane_id = int(command["target_lane_id"])
    initial_sequence = int(command.get("initial_request_sequence") or 0)
    accepted_sequence: Optional[int] = None
    accepted_sim_s: Optional[float] = None
    centered_since_sim_s: Optional[float] = None
    completion_sim_s: Optional[float] = None
    raw_centered_since_sim_s: Optional[float] = None
    raw_completion_sim_s: Optional[float] = None
    terminal_state: Optional[str] = None
    terminal_reason: Optional[str] = None
    terminal_sim_s: Optional[float] = None
    terminal_done_in_target = False

    for outer in unique_trajectory_rows(trajectory):
        vehicle = (outer.get("vehicles") or {}).get(veh_id)
        if not vehicle:
            continue
        sim_time_s = float(vehicle.get("sim_time_s") or outer.get("sim_time_s") or 0.0)
        lane_change = vehicle.get("lane_change") or {}
        sequence = int(lane_change.get("request_sequence") or 0)
        if accepted_sim_s is None and sequence > initial_sequence:
            accepted_sequence = sequence
            accepted_sim_s = sim_time_s

        centered = (
            int(vehicle.get("lane_id")) == target_lane_id
            and float(vehicle.get("distance_to_center") or 0.0)
            <= CENTER_THRESHOLD_M
        )
        if centered:
            if raw_centered_since_sim_s is None:
                raw_centered_since_sim_s = sim_time_s
            elif (
                raw_completion_sim_s is None
                and sim_time_s - raw_centered_since_sim_s + 1e-9
                >= SETTLING_TIME_SIM_S
            ):
                raw_completion_sim_s = sim_time_s
        else:
            raw_centered_since_sim_s = None

        if accepted_sim_s is None:
            continue
        if centered:
            if centered_since_sim_s is None:
                centered_since_sim_s = sim_time_s
            elif (
                completion_sim_s is None
                and sim_time_s - centered_since_sim_s + 1e-9
                >= SETTLING_TIME_SIM_S
            ):
                completion_sim_s = sim_time_s
        else:
            centered_since_sim_s = None

        candidate_terminal_sim_s = lane_change.get("last_terminal_sim_s")
        if (
            lane_change.get("last_terminal_state") == "DONE"
            and int(vehicle.get("lane_id")) == target_lane_id
            and float(vehicle.get("distance_to_center") or 0.0)
            <= CENTER_THRESHOLD_M
        ):
            terminal_done_in_target = True
        if (
            candidate_terminal_sim_s is not None
            and float(candidate_terminal_sim_s) >= accepted_sim_s
            and sequence >= int(accepted_sequence or sequence)
            and (
                terminal_sim_s is None
                or float(candidate_terminal_sim_s) > terminal_sim_s
            )
        ):
            terminal_state = lane_change.get("last_terminal_state")
            terminal_reason = lane_change.get("last_terminal_reason")
            terminal_sim_s = float(candidate_terminal_sim_s)

    return {
        "command_id": command["command_id"],
        "veh_id": int(veh_id),
        "target_lane_id": target_lane_id,
        "accepted": accepted_sim_s is not None,
        "accepted_sequence": accepted_sequence,
        "accepted_sim_s": accepted_sim_s,
        "raw_target_dwell_completed": raw_completion_sim_s is not None,
        "raw_target_dwell_completion_sim_s": raw_completion_sim_s,
        "centered_dwell_completed": completion_sim_s is not None,
        "completion_sim_s": completion_sim_s,
        "terminal_state": terminal_state,
        "terminal_reason": terminal_reason,
        "terminal_sim_s": terminal_sim_s,
        "terminal_done_in_target": terminal_done_in_target,
    }


def audit_transactions(method: str, events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    if method not in MIND_METHODS:
        mec_decisions = sum(row.get("event") == "mec_decision" for row in rows)
        return {
            "applicable": False,
            "decision_count": mec_decisions,
            "executable_decision_count": 0,
            "closed_transaction_count": 0,
            "open_transaction_ids": [],
            "complete": mec_decisions == 0,
        }
    executable_ids = {
        str(row["decision"]["decision_id"])
        for row in rows
        if row.get("event") == "mec_decision"
        and row.get("executed")
        and (row.get("decision") or {}).get("decision_id")
    }
    closed_ids = {
        str(row["transaction_id"])
        for row in rows
        if row.get("event") in {"transaction_completed", "transaction_failed"}
        and row.get("transaction_id")
    }
    return {
        "applicable": True,
        "decision_count": sum(row.get("event") == "mec_decision" for row in rows),
        "executable_decision_count": len(executable_ids),
        "closed_transaction_count": len(executable_ids.intersection(closed_ids)),
        "open_transaction_ids": sorted(executable_ids - closed_ids),
        "complete": executable_ids.issubset(closed_ids),
    }


def audit_method(
    directory: Path,
    trial_timeout_sim_s: float = TRIAL_TIMEOUT_SIM_S,
) -> dict[str, Any]:
    required = [
        directory / "metadata.json",
        directory / "events.jsonl",
        directory / "trajectory.jsonl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {
            "method": directory.name,
            "audit_complete": False,
            "missing": missing,
        }
    metadata = read_json(required[0])
    events = read_jsonl(required[1])
    trajectory = read_jsonl(required[2])
    command_audits = [
        replay_command(trajectory, command) for command in metadata.get("commands", [])
    ]
    collision_count = replay_collision_count(trajectory)
    minimum_shared, gap_samples = replay_gap_metrics(trajectory)
    completed_count = sum(
        command["centered_dwell_completed"] for command in command_audits
    )
    raw_completed_count = sum(
        command["raw_target_dwell_completed"] for command in command_audits
    )
    lane_alias_without_acceptance_count = sum(
        command["raw_target_dwell_completed"] and not command["accepted"]
        for command in command_audits
    )
    done_count = sum(command["terminal_done_in_target"] for command in command_audits)
    accepted_count = sum(command["accepted"] for command in command_audits)
    independently_safe = bool(
        command_audits
        and completed_count == len(command_audits)
        and done_count == len(command_audits)
        and collision_count == 0
        and gap_samples == 0
        and float(metadata.get("elapsed_sim_s") or float("inf"))
        <= trial_timeout_sim_s + 0.1
    )
    transactions = audit_transactions(str(metadata["method"]), events)
    checks = {
        "trajectory_hash": sha256_file(required[2])
        == metadata.get("trajectory_sha256"),
        "accepted_count": accepted_count
        == sum(bool(command.get("accepted_sequences")) for command in metadata.get("commands", [])),
        # The online tracker advances only after the controller accepts a
        # maneuver request; raw lane occupancy remains a separate diagnostic.
        "completion_count": completed_count
        == int(metadata.get("completed_command_count") or 0),
        "terminal_done_count": done_count
        == int(metadata.get("terminal_done_count") or 0),
        "collision_count": collision_count == int(metadata.get("collision_count") or 0),
        "gap_violation_samples": gap_samples
        == int(metadata.get("gap_violation_samples") or 0),
        "safe_task_success": independently_safe
        == bool(metadata.get("safe_task_success")),
        "transaction_audit": bool(transactions["complete"]),
        "frames_disabled": metadata.get("persist_frames") is False,
        "tm_auto_lane_change_disabled": metadata.get(
            "traffic_manager_auto_lane_change"
        )
        is False,
        "tm_collision_avoidance_disabled": metadata.get(
            "traffic_manager_collision_avoidance"
        )
        is False,
        "initial_conflict_requirement": bool(
            not metadata.get("initial_conflict_required")
            or int(metadata.get("initial_conflict_count") or 0) > 0
        ),
    }
    return {
        "block_id": metadata["block_id"],
        "method": metadata["method"],
        "scenario_family": metadata["scenario_family"],
        "fleet_size": int(metadata["fleet_size"]),
        "seed": int(metadata["seed"]),
        "status": metadata["status"],
        "audit_complete": all(checks.values()),
        "checks": checks,
        "planned_command_count": len(command_audits),
        "accepted_command_count": accepted_count,
        "raw_target_dwell_completion_count": raw_completed_count,
        "centered_dwell_completion_count": completed_count,
        "lane_alias_without_acceptance_count": lane_alias_without_acceptance_count,
        "terminal_done_count": done_count,
        "collision_count": collision_count,
        "gap_violation_samples": gap_samples,
        "minimum_shared_corridor_distance_m": minimum_shared,
        "elapsed_sim_s": float(metadata.get("elapsed_sim_s") or 0.0),
        "elapsed_wall_s": float(metadata.get("elapsed_wall_s") or 0.0),
        "safe_task_success": independently_safe,
        "initial_conflict_count": int(metadata.get("initial_conflict_count") or 0),
        "decision_counts": dict(metadata.get("decision_counts") or {}),
        "reason_code_counts": dict(metadata.get("reason_code_counts") or {}),
        "mappo_policy_steps": int(metadata.get("mappo_policy_steps") or 0),
        "mappo_validator_interventions": int(
            metadata.get("mappo_validator_interventions") or 0
        ),
        "paired_design_sha256": metadata.get("paired_design_sha256"),
        "initial_state_signature": metadata.get("initial_state_signature") or {},
        "transactions": transactions,
        "commands": command_audits,
        "input_sha256": {path.name: sha256_file(path) for path in required},
    }


def audit_pairing(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    methods = list(rows)
    design_hashes = {row.get("paired_design_sha256") for row in methods}
    slots = sorted(
        {slot for row in methods for slot in row.get("initial_state_signature", {})}
    )
    slot_checks: dict[str, Any] = {}
    for slot in slots:
        states = [row["initial_state_signature"].get(slot) for row in methods]
        missing = any(state is None for state in states)
        roads = {state.get("road_id") for state in states if state is not None}
        lanes = {state.get("lane_id") for state in states if state is not None}
        s_values = [float(state["s_m"]) for state in states if state is not None]
        speeds = [
            float(state["speed_kmh"]) for state in states if state is not None
        ]
        s_spread = max(s_values) - min(s_values) if s_values else float("inf")
        speed_spread = max(speeds) - min(speeds) if speeds else float("inf")
        slot_checks[slot] = {
            "missing": missing,
            "road_ids": sorted(roads, key=str),
            "lane_ids": sorted(lanes, key=str),
            "longitudinal_spread_m": s_spread,
            "speed_spread_kmh": speed_spread,
            "paired": bool(
                not missing
                and len(roads) == 1
                and len(lanes) == 1
                and s_spread <= PAIRING_S_TOLERANCE_M
                and speed_spread <= PAIRING_SPEED_TOLERANCE_KMH
            ),
        }
    return {
        "design_hashes": sorted(value for value in design_hashes if value),
        "design_hash_match": len(design_hashes) == 1 and None not in design_hashes,
        "slots": slot_checks,
        "paired": bool(slot_checks)
        and all(value["paired"] for value in slot_checks.values())
        and len(design_hashes) == 1
        and None not in design_hashes,
    }


def write_trial_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = [
        "block_id",
        "scenario_family",
        "fleet_size",
        "seed",
        "method",
        "status",
        "safe_task_success",
        "planned_command_count",
        "accepted_command_count",
        "centered_dwell_completion_count",
        "terminal_done_count",
        "collision_count",
        "gap_violation_samples",
        "minimum_shared_corridor_distance_m",
        "elapsed_sim_s",
        "elapsed_wall_s",
        "audit_complete",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_report(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    schedule_path = root / "schedule.json"
    provenance_path = root / "provenance.json"
    manifest = read_json(manifest_path)
    schedule = read_json(schedule_path)
    trial_timeout_sim_s = float(
        schedule.get("trial_timeout_sim_s") or TRIAL_TIMEOUT_SIM_S
    )
    block_directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and any((child / "metadata.json").exists() for child in path.iterdir())
    )
    method_rows = [
        audit_method(method_dir, trial_timeout_sim_s)
        for block_dir in block_directories
        for method_dir in sorted(block_dir.iterdir())
        if method_dir.is_dir()
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        grouped[str(row.get("block_id"))].append(row)
    expected_methods = set(manifest.get("methods") or [])
    block_audits = {}
    for block_id, rows in sorted(grouped.items()):
        observed_methods = {str(row.get("method")) for row in rows}
        pairing = audit_pairing(rows)
        block_audits[block_id] = {
            "observed_methods": sorted(observed_methods),
            "method_coverage": observed_methods == expected_methods,
            "pairing": pairing,
            "all_method_audits_complete": all(row.get("audit_complete") for row in rows),
        }
    persisted_frames = sorted(
        str(path.relative_to(root))
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in root.rglob(suffix)
    )
    provenance_required = manifest.get("claim_status") != "exploratory_pilot"
    provenance_valid = bool(manifest.get("provenance_sha256")) and provenance_path.exists()
    if provenance_valid:
        provenance_valid = (
            sha256_file(provenance_path) == manifest.get("provenance_sha256")
        )
    lock_required = schedule.get("claim_status") == "frozen_confirmatory"
    lock_record = manifest.get("confirmatory_lock") or {}
    lock_path = Path(lock_record["path"]) if lock_record.get("path") else None
    lock_valid = bool(
        lock_path
        and lock_path.exists()
        and sha256_file(lock_path) == lock_record.get("sha256")
        and lock_record.get("registration_freeze_commit")
    )
    checks = {
        "manifest_row_count": len(method_rows) == int(manifest.get("rows") or 0),
        "manifest_block_count": len(grouped) == int(manifest.get("blocks") or 0),
        "schedule_hash": sha256_json(schedule) == manifest.get("schedule_sha256"),
        "all_methods_audited": bool(method_rows)
        and all(row.get("audit_complete") for row in method_rows),
        "all_blocks_have_method_coverage": bool(block_audits)
        and all(row["method_coverage"] for row in block_audits.values()),
        "all_blocks_paired": bool(block_audits)
        and all(row["pairing"]["paired"] for row in block_audits.values()),
        "no_persisted_frames": not persisted_frames,
        "provenance": bool(provenance_valid or not provenance_required),
        "confirmatory_lock": bool(lock_valid or not lock_required),
    }
    method_outcomes = Counter(
        (str(row.get("method")), bool(row.get("safe_task_success")))
        for row in method_rows
    )
    return {
        "schema_version": "1.0",
        "claim_scope": (
            "confirmatory paired CARLA trajectory audit"
            if schedule.get("claim_status") == "frozen_confirmatory"
            else "exploratory paired CARLA integration audit"
        ),
        "root": str(root.resolve()),
        "thresholds": {
            "center_threshold_m": CENTER_THRESHOLD_M,
            "settling_time_sim_s": SETTLING_TIME_SIM_S,
            "gap_threshold_m": GAP_THRESHOLD_M,
            "trial_timeout_sim_s": trial_timeout_sim_s,
            "pairing_s_tolerance_m": PAIRING_S_TOLERANCE_M,
            "pairing_speed_tolerance_kmh": PAIRING_SPEED_TOLERANCE_KMH,
        },
        "checks": checks,
        "audit_passed": all(checks.values()),
        "persisted_frames": persisted_frames,
        "block_audits": block_audits,
        "method_outcomes": [
            {"method": method, "safe_task_success": safe, "count": count}
            for (method, safe), count in sorted(method_outcomes.items())
        ],
        "methods": method_rows,
        "input_sha256": {
            "manifest.json": sha256_file(manifest_path),
            "schedule.json": sha256_file(schedule_path),
            "provenance.json": (
                sha256_file(provenance_path) if provenance_path.exists() else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    output = args.output or args.root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_trial_csv(output.with_suffix(".csv"), report["methods"])
    print(json.dumps({"audit_passed": report["audit_passed"], "checks": report["checks"]}, indent=2))
    if not report["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
