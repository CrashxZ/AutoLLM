#!/usr/bin/env python3
"""Run the exploratory method-neutral CARLA multi-vehicle executor campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_carla_lane_changes import PhysicalCompletionTracker


STATIONS: tuple[dict[int, int], ...] = (
    {-4: 343, -3: 344, -2: 345, -1: 346},
    {-4: 16, -3: 17, -2: 18, -1: 19},
    {-4: 339, -3: 340, -2: 341, -1: 342},
    {-4: 335, -3: 336, -2: 337, -1: 338},
    {-4: 0, -3: 1, -2: 2, -1: 3},
)


@dataclass(frozen=True)
class CampaignBudget:
    trial_timeout_s: float = 60.0
    total_timeout_s: float = 900.0
    maximum_output_mb: float = 250.0
    maximum_rss_mb: float = 2048.0
    maximum_consecutive_failures: int = 3


@dataclass(frozen=True)
class PlannedCommand:
    slot: int
    direction: str
    issue_offset_sim_s: float


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    phase: str
    fleet_size: int
    pattern: str
    repetition: int
    seed: int


@dataclass(frozen=True)
class TrialLayout:
    spawn_indices: tuple[int, ...]
    commands: tuple[PlannedCommand, ...]


class CampaignGuard:
    def __init__(self, budget: CampaignBudget, started_at_s: float):
        self.budget = budget
        self.started_at_s = started_at_s
        self.consecutive_failures = 0

    def pretrial_reason(
        self, *, now_s: float, output_mb: float, rss_mb: float
    ) -> Optional[str]:
        if now_s - self.started_at_s >= self.budget.total_timeout_s:
            return "total_timeout"
        if output_mb >= self.budget.maximum_output_mb:
            return "output_budget_exceeded"
        if rss_mb >= self.budget.maximum_rss_mb:
            return "memory_budget_exceeded"
        if self.consecutive_failures >= self.budget.maximum_consecutive_failures:
            return "consecutive_failure_limit"
        return None

    def observe_trial(self, result: dict[str, Any]) -> Optional[str]:
        if int(result.get("collision_count") or 0) > 0:
            return "collision_abort"
        if result.get("success"):
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        if self.consecutive_failures >= self.budget.maximum_consecutive_failures:
            return "consecutive_failure_limit"
        return None


def trial_timeout_reached(now_s: float, deadline_s: float) -> bool:
    return now_s >= deadline_s


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024.0 * 1024.0
    )


def runner_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_telemetry(client: httpx.Client, api_base: str) -> dict[str, dict]:
    response = client.get(f"{api_base}/telemetry.jsonl")
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return {}
    return rows[-1].get("vehicles") or {}


def wait_for_ready_telemetry(
    client: httpx.Client,
    api_base: str,
    vehicle_ids: Iterable[int],
    *,
    minimum_speed_kmh: float = 12.0,
    timeout_s: float = 25.0,
) -> dict[str, dict]:
    expected = {str(value) for value in vehicle_ids}
    deadline = time.time() + timeout_s
    last: dict[str, dict] = {}
    while time.time() < deadline:
        last = latest_telemetry(client, api_base)
        if expected.issubset(last) and all(
            float(last[veh_id].get("speed_kmh") or 0.0) >= minimum_speed_kmh
            and last[veh_id].get("lane_id") is not None
            and last[veh_id].get("sim_time_s") is not None
            and not last[veh_id].get("is_junction")
            for veh_id in expected
        ):
            return last
        time.sleep(0.05)
    speeds = {
        veh_id: (last.get(veh_id) or {}).get("speed_kmh") for veh_id in expected
    }
    raise TimeoutError(f"fleet did not become ready; speeds={speeds}")


def build_layout(pattern: str, fleet_size: int) -> TrialLayout:
    if fleet_size not in {2, 4, 8}:
        raise ValueError("fleet_size must be 2, 4, or 8")
    spawns: list[int] = []
    commands: list[PlannedCommand] = []

    def add(station: int, lane_id: int, direction: Optional[str], offset: float = 0.0):
        slot = len(spawns)
        spawns.append(STATIONS[station][lane_id])
        if direction is not None:
            commands.append(PlannedCommand(slot, direction, offset))

    if pattern == "known_answer":
        add(0, -3, "left")
        add(2, -1, None)
    elif pattern == "parallel":
        station_sets = {2: (0,), 4: (0, 2), 8: (0, 1, 2, 3)}
        for station in station_sets[fleet_size]:
            add(station, -3, "right")
            add(station, -2, "left")
    elif pattern == "staggered_merge":
        sources = ((0, -3), (1, -1), (2, -3), (3, -1), (4, -3))
        command_count = {2: 2, 4: 4, 8: 5}[fleet_size]
        for ordinal, (station, lane_id) in enumerate(sources[:command_count]):
            direction = "left" if lane_id == -3 else "right"
            add(station, lane_id, direction, float(ordinal))
        backgrounds = ((1, -4), (3, -4), (4, -1))
        for station, lane_id in backgrounds[: fleet_size - command_count]:
            add(station, lane_id, None)
    elif pattern == "reciprocal":
        if fleet_size == 2:
            add(0, -4, "left")
            add(1, -3, "right")
        else:
            station_count = fleet_size // 2
            for station in range(station_count):
                if station % 2 == 0:
                    add(station, -4, "left")
                    add(station, -2, "left")
                else:
                    add(station, -3, "right")
                    add(station, -1, "right")
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    if len(spawns) != fleet_size or len(set(spawns)) != fleet_size:
        raise AssertionError(f"invalid {pattern}/{fleet_size} layout: {spawns}")
    return TrialLayout(tuple(spawns), tuple(commands))


def build_schedules(
    *, campaign_repetitions: int = 2, randomization_seed: int = 2026083000
) -> dict[str, list[TrialSpec]]:
    minimal = [
        TrialSpec("minimal-known-answer", "minimal", 2, "known_answer", 1, 2026083001)
    ]
    probes = [
        TrialSpec(
            f"probe-parallel-n{fleet_size}",
            "probe",
            fleet_size,
            "parallel",
            1,
            2026083020 + fleet_size,
        )
        for fleet_size in (2, 4, 8)
    ]
    campaign: list[TrialSpec] = []
    ordinal = 0
    for fleet_size in (2, 4, 8):
        for pattern in ("parallel", "staggered_merge", "reciprocal"):
            for repetition in range(1, campaign_repetitions + 1):
                ordinal += 1
                campaign.append(
                    TrialSpec(
                        f"campaign-{pattern}-n{fleet_size}-r{repetition:02d}",
                        "campaign",
                        fleet_size,
                        pattern,
                        repetition,
                        2026083100 + ordinal,
                    )
                )
    random.Random(randomization_seed).shuffle(campaign)
    return {"minimal": minimal, "probes": probes, "campaign": campaign}


def expected_target_lane(initial_lane_id: int, direction: str) -> int:
    left_delta = 1 if initial_lane_id < 0 else -1
    return initial_lane_id + (
        left_delta if direction == "left" else -left_delta
    )


def occupied_lanes(row: dict[str, Any]) -> set[int]:
    raw = row.get("occupied_lane_ids") or (row.get("lane_change") or {}).get(
        "occupied_lane_ids"
    )
    if raw:
        return {int(value) for value in raw}
    return {int(row["lane_id"])} if row.get("lane_id") is not None else set()


def pair_distance_metrics(telemetry: dict[str, dict]) -> tuple[Optional[float], Optional[float]]:
    rows = list(telemetry.values())
    minimum_any: Optional[float] = None
    minimum_shared: Optional[float] = None
    for first_index, first in enumerate(rows):
        for second in rows[first_index + 1 :]:
            try:
                distance = math.hypot(
                    float(first["pose"]["x"]) - float(second["pose"]["x"]),
                    float(first["pose"]["y"]) - float(second["pose"]["y"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            minimum_any = distance if minimum_any is None else min(minimum_any, distance)
            if occupied_lanes(first).intersection(occupied_lanes(second)):
                minimum_shared = (
                    distance if minimum_shared is None else min(minimum_shared, distance)
                )
    return minimum_any, minimum_shared


def download_archive(
    client: httpx.Client, api_base: str, destination: Path
) -> Optional[str]:
    try:
        response = client.get(f"{api_base}/archive", timeout=120.0)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return hashlib.sha256(response.content).hexdigest()
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}"


def run_trial(
    client: httpx.Client,
    api_base: str,
    spec: TrialSpec,
    output_dir: Path,
    budget: CampaignBudget,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    layout = build_layout(spec.pattern, spec.fleet_size)
    trajectory_path = output_dir / "trajectory.jsonl"
    events_path = output_dir / "events.jsonl"
    client.post(f"{api_base}/reset", json={}).raise_for_status()
    configured = client.post(
        f"{api_base}/config",
        json={
            "num_cars": spec.fleet_size,
            "spawn_indices": list(layout.spawn_indices),
            "initial_speeds": [35.0] * spec.fleet_size,
            "coordination_mode": "IA",
            "scenario_id": f"EXECUTOR_{spec.trial_id.upper()}",
            "seed": spec.seed,
            "persist_frames": False,
        },
    )
    configured.raise_for_status()
    vehicle_ids = [int(value) for value in configured.json()["veh_ids"]]
    ready = wait_for_ready_telemetry(client, api_base, vehicle_ids)
    base_sim_s = max(float(ready[str(veh_id)]["sim_time_s"]) for veh_id in vehicle_ids)
    started_wall_s = time.time()
    deadline_wall_s = started_wall_s + budget.trial_timeout_s
    commands: dict[str, dict[str, Any]] = {}
    for ordinal, planned in enumerate(layout.commands, start=1):
        veh_id = vehicle_ids[planned.slot]
        row = ready[str(veh_id)]
        initial_lane_id = int(row["lane_id"])
        command_id = f"{spec.trial_id}-c{ordinal:02d}-v{veh_id}"
        commands[command_id] = {
            "command_id": command_id,
            "slot": planned.slot,
            "veh_id": veh_id,
            "direction": planned.direction,
            "issue_offset_sim_s": planned.issue_offset_sim_s,
            "initial_lane_id": initial_lane_id,
            "expected_target_lane_id": expected_target_lane(
                initial_lane_id, planned.direction
            ),
            "initial_request_sequence": int(
                (row.get("lane_change") or {}).get("request_sequence") or 0
            ),
            "sent": False,
            "accepted": False,
            "terminal": False,
            "independent_completed": False,
            "tracker": None,
        }

    maximum_collision_count = 0
    minimum_center_distance_m: Optional[float] = None
    minimum_shared_corridor_distance_m: Optional[float] = None
    last_frame: Optional[int] = None
    last_telemetry = ready

    while not trial_timeout_reached(time.time(), deadline_wall_s):
        telemetry = latest_telemetry(client, api_base)
        if not all(str(veh_id) in telemetry for veh_id in vehicle_ids):
            time.sleep(0.03)
            continue
        last_telemetry = {str(veh_id): telemetry[str(veh_id)] for veh_id in vehicle_ids}
        current_frame = max(
            int(row.get("sim_frame") or 0) for row in last_telemetry.values()
        )
        if current_frame == last_frame:
            time.sleep(0.03)
            continue
        last_frame = current_frame
        current_sim_s = max(
            float(row["sim_time_s"]) for row in last_telemetry.values()
        )
        now_wall_s = time.time()
        append_jsonl(
            trajectory_path,
            {
                "wall_s": now_wall_s,
                "elapsed_wall_s": now_wall_s - started_wall_s,
                "sim_time_s": current_sim_s,
                "elapsed_sim_s": current_sim_s - base_sim_s,
                "vehicles": last_telemetry,
            },
        )
        collision_count = sum(
            int(row.get("collision_count") or 0) for row in last_telemetry.values()
        )
        maximum_collision_count = max(maximum_collision_count, collision_count)
        any_distance, shared_distance = pair_distance_metrics(last_telemetry)
        if any_distance is not None:
            minimum_center_distance_m = (
                any_distance
                if minimum_center_distance_m is None
                else min(minimum_center_distance_m, any_distance)
            )
        if shared_distance is not None:
            minimum_shared_corridor_distance_m = (
                shared_distance
                if minimum_shared_corridor_distance_m is None
                else min(minimum_shared_corridor_distance_m, shared_distance)
            )
        if maximum_collision_count:
            append_jsonl(
                events_path,
                {
                    "event": "collision_abort",
                    "wall_s": now_wall_s,
                    "sim_time_s": current_sim_s,
                    "collision_count": maximum_collision_count,
                },
            )
            break

        for command in commands.values():
            row = last_telemetry[str(command["veh_id"])]
            lane_change = row.get("lane_change") or {}
            if (
                not command["sent"]
                and current_sim_s - base_sim_s
                >= float(command["issue_offset_sim_s"])
            ):
                client.post(
                    f"{api_base}/command",
                    json={
                        "cmd": "lane",
                        "veh_id": command["veh_id"],
                        "dir": command["direction"],
                    },
                ).raise_for_status()
                command["sent"] = True
                command["sent_wall_s"] = now_wall_s
                command["sent_sim_s"] = current_sim_s
                append_jsonl(events_path, {"event": "command_sent", **serializable_command(command)})

            sequence = int(lane_change.get("request_sequence") or 0)
            if (
                command["sent"]
                and not command["accepted"]
                and sequence > int(command["initial_request_sequence"])
            ):
                command["accepted"] = True
                command["request_sequence"] = sequence
                command["accepted_wall_s"] = lane_change.get("request_started_at")
                command["accepted_sim_s"] = lane_change.get("request_started_sim_s")
                command["target_lane_id"] = int(
                    lane_change.get("target_lane_id")
                    if lane_change.get("target_lane_id") is not None
                    else command["expected_target_lane_id"]
                )
                command["tracker"] = PhysicalCompletionTracker(
                    int(command["target_lane_id"])
                )
                append_jsonl(
                    events_path,
                    {"event": "command_accepted", **serializable_command(command)},
                )

            if command["accepted"] and not command["independent_completed"]:
                tracker = command["tracker"]
                if tracker.observe(
                    lane_id=int(row["lane_id"]),
                    distance_to_center_m=float(row.get("distance_to_center") or 0.0),
                    sim_time_s=float(row["sim_time_s"]),
                    wall_time_s=now_wall_s,
                ):
                    command["independent_completed"] = True
                    command["independent_completed_sim_s"] = tracker.completed_sim_s
                    command["independent_completed_wall_s"] = tracker.completed_wall_s
                    append_jsonl(
                        events_path,
                        {
                            "event": "independent_centered_completion",
                            **serializable_command(command),
                        },
                    )

            terminal_sim_s = lane_change.get("last_terminal_sim_s")
            accepted_sim_s = command.get("accepted_sim_s")
            if (
                command["accepted"]
                and not command["terminal"]
                and terminal_sim_s is not None
                and accepted_sim_s is not None
                and float(terminal_sim_s) >= float(accepted_sim_s)
            ):
                command["terminal"] = True
                command["terminal_state"] = lane_change.get("last_terminal_state")
                command["terminal_reason"] = lane_change.get("last_terminal_reason")
                command["terminal_sim_s"] = terminal_sim_s
                command["terminal_wall_s"] = lane_change.get("last_terminal_at")
                append_jsonl(
                    events_path,
                    {"event": "fsm_terminal", **serializable_command(command)},
                )

        if commands and all(
            command["accepted"]
            and command["terminal"]
            and command["independent_completed"]
            for command in commands.values()
        ):
            break
        time.sleep(0.03)

    ended_wall_s = time.time()
    accepted = sum(bool(command["accepted"]) for command in commands.values())
    terminals = sum(bool(command["terminal"]) for command in commands.values())
    done = sum(
        command.get("terminal_state") == "DONE" for command in commands.values()
    )
    completed = sum(
        bool(command["independent_completed"]) for command in commands.values()
    )
    all_sent = all(command["sent"] for command in commands.values())
    success = bool(commands) and (
        maximum_collision_count == 0
        and all_sent
        and accepted == len(commands)
        and terminals == accepted
        and done == accepted
        and completed == accepted
    )
    status = (
        "collision"
        if maximum_collision_count
        else "completed"
        if success
        else "timeout"
        if trial_timeout_reached(ended_wall_s, deadline_wall_s)
        else "failed"
    )
    archive_sha256 = download_archive(
        client, api_base, output_dir / "server_archive.zip"
    )
    result = {
        **asdict(spec),
        "status": status,
        "success": success,
        "spawn_indices": list(layout.spawn_indices),
        "vehicle_ids": vehicle_ids,
        "planned_command_count": len(commands),
        "sent_command_count": sum(bool(command["sent"]) for command in commands.values()),
        "accepted_command_count": accepted,
        "terminal_command_count": terminals,
        "done_command_count": done,
        "independent_completion_count": completed,
        "traceability_rate": terminals / accepted if accepted else 0.0,
        "completion_rate": completed / accepted if accepted else 0.0,
        "collision_count": maximum_collision_count,
        "minimum_center_distance_m": minimum_center_distance_m,
        "minimum_shared_corridor_distance_m": minimum_shared_corridor_distance_m,
        "started_wall_s": started_wall_s,
        "ended_wall_s": ended_wall_s,
        "elapsed_wall_s": ended_wall_s - started_wall_s,
        "base_sim_s": base_sim_s,
        "last_sim_s": max(
            float(row.get("sim_time_s") or base_sim_s)
            for row in last_telemetry.values()
        ),
        "trajectory_samples": (
            len(trajectory_path.read_text(encoding="utf-8").splitlines())
            if trajectory_path.exists()
            else 0
        ),
        "archive_sha256": archive_sha256,
        "commands": [serializable_command(command) for command in commands.values()],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def serializable_command(command: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in command.items() if key != "tracker"}


def verify_kill_switches(output: Path, budget: CampaignBudget) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    checks: dict[str, bool] = {}
    guard = CampaignGuard(budget, 100.0)
    checks["trial_timeout"] = trial_timeout_reached(160.0, 160.0)
    checks["total_timeout"] = guard.pretrial_reason(
        now_s=100.0 + budget.total_timeout_s,
        output_mb=0.0,
        rss_mb=0.0,
    ) == "total_timeout"
    checks["output_budget"] = guard.pretrial_reason(
        now_s=100.0,
        output_mb=budget.maximum_output_mb,
        rss_mb=0.0,
    ) == "output_budget_exceeded"
    checks["memory_budget"] = guard.pretrial_reason(
        now_s=100.0,
        output_mb=0.0,
        rss_mb=budget.maximum_rss_mb,
    ) == "memory_budget_exceeded"
    failure_guard = CampaignGuard(budget, 100.0)
    reasons = [
        failure_guard.observe_trial({"success": False, "collision_count": 0})
        for _ in range(budget.maximum_consecutive_failures)
    ]
    checks["consecutive_failure_limit"] = (
        reasons[-1] == "consecutive_failure_limit"
    )
    collision_guard = CampaignGuard(budget, 100.0)
    checks["collision_abort"] = collision_guard.observe_trial(
        {"success": False, "collision_count": 1}
    ) == "collision_abort"
    report = {
        "schema_version": "1.0",
        "budget": asdict(budget),
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output / "kill_switch_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        default="data/experiments/carla_multi_vehicle_executor_20260827",
    )
    parser.add_argument("--trial-timeout-s", type=float, default=60.0)
    parser.add_argument("--total-timeout-s", type=float, default=900.0)
    parser.add_argument("--maximum-output-mb", type=float, default=250.0)
    parser.add_argument("--maximum-rss-mb", type=float, default=2048.0)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    parser.add_argument("--campaign-repetitions", type=int, default=2)
    parser.add_argument("--randomization-seed", type=int, default=2026083000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-kill-switches", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    budget = CampaignBudget(
        trial_timeout_s=args.trial_timeout_s,
        total_timeout_s=args.total_timeout_s,
        maximum_output_mb=args.maximum_output_mb,
        maximum_rss_mb=args.maximum_rss_mb,
        maximum_consecutive_failures=args.maximum_consecutive_failures,
    )
    if args.verify_kill_switches:
        verify_kill_switches(output, budget)
        return

    schedules = build_schedules(
        campaign_repetitions=args.campaign_repetitions,
        randomization_seed=args.randomization_seed,
    )
    flattened = schedules["minimal"] + schedules["probes"] + schedules["campaign"]
    output.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "claim_scope": "exploratory multi-vehicle CARLA executor feasibility only",
        "created_at_s": time.time(),
        "api_base": args.api_base.rstrip("/"),
        "budget": asdict(budget),
        "campaign_repetitions": args.campaign_repetitions,
        "randomization_seed": args.randomization_seed,
        "persist_frames": False,
        "completion_definition": {
            "center_threshold_m": 0.5,
            "settling_time_sim_s": 0.75,
        },
        "schedule": [
            {**asdict(spec), **asdict(build_layout(spec.pattern, spec.fleet_size))}
            for spec in flattened
        ],
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.dry_run:
        print(
            f"dry-run: minimal={len(schedules['minimal'])} "
            f"probes={len(schedules['probes'])} campaign={len(schedules['campaign'])}"
        )
        return

    api_base = args.api_base.rstrip("/")
    started_at_s = time.time()
    guard = CampaignGuard(budget, started_at_s)
    results: list[dict[str, Any]] = []
    stop_reason: Optional[str] = None
    with httpx.Client(timeout=120.0) as client:
        client.get(f"{api_base}/health").raise_for_status()
        unsafe = client.get(f"{api_base}/tm/unsafe").json().get("unsafe_mode")
        if unsafe:
            client.post(f"{api_base}/tm/unsafe", json={"unsafe": False}).raise_for_status()
        for phase_name in ("minimal", "probes", "campaign"):
            phase_results: list[dict[str, Any]] = []
            for spec in schedules[phase_name]:
                stop_reason = guard.pretrial_reason(
                    now_s=time.time(),
                    output_mb=directory_size_mb(output),
                    rss_mb=runner_rss_mb(),
                )
                if stop_reason:
                    break
                try:
                    result = run_trial(
                        client, api_base, spec, output / spec.trial_id, budget
                    )
                except Exception as exc:
                    result = {
                        **asdict(spec),
                        "status": "runner_error",
                        "success": False,
                        "collision_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(result)
                phase_results.append(result)
                append_jsonl(output / "results.jsonl", result)
                print(
                    f"[{len(results)}/{len(flattened)}] {spec.trial_id}: "
                    f"{result['status']} accepted={result.get('accepted_command_count')} "
                    f"done={result.get('done_command_count')} "
                    f"collision={result.get('collision_count')}"
                )
                stop_reason = guard.observe_trial(result)
                if stop_reason:
                    break
            if stop_reason:
                break
            if phase_name == "minimal" and not all(
                result.get("success") for result in phase_results
            ):
                stop_reason = "minimal_gate_failed"
                break
            if phase_name == "probes" and not all(
                result.get("success") for result in phase_results
            ):
                stop_reason = "scaling_probe_gate_failed"
                break
        client.post(f"{api_base}/reset", json={}).raise_for_status()

    if results:
        write_csv(output / "results.csv", results)
    manifest.update(
        {
            "status": stop_reason or "complete",
            "completed_trials": len(results),
            "elapsed_wall_s": time.time() - started_at_s,
            "peak_rss_mb": runner_rss_mb(),
            "output_mb": directory_size_mb(output),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if stop_reason:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
