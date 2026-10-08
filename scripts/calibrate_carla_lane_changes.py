#!/usr/bin/env python3
"""Measure CARLA lane-change timing without changing controller thresholds.

The calibration separates wall-clock duration, CARLA simulation duration,
internal FSM outcome, and physical lane arrival. It is exploratory engineering
evidence used to choose a common executor for a later comparative study.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx


@dataclass(frozen=True)
class CalibrationBudget:
    trial_timeout_s: float = 20.0
    sweep_timeout_s: float = 900.0
    maximum_output_mb: float = 100.0
    maximum_rss_mb: float = 2048.0
    maximum_consecutive_failures: int = 3


@dataclass(frozen=True)
class CalibrationTrial:
    trial_id: str
    spawn_index: int
    direction: str
    target_speed_kmh: float
    seed: int


@dataclass
class PhysicalCompletionTracker:
    """Track target-lane centering using CARLA simulation time."""

    target_lane_id: int
    center_threshold_m: float = 0.5
    settling_time_s: float = 0.75
    centered_since_sim_s: Optional[float] = None
    completed_sim_s: Optional[float] = None
    completed_wall_s: Optional[float] = None

    def observe(
        self,
        *,
        lane_id: int,
        distance_to_center_m: float,
        sim_time_s: float,
        wall_time_s: float,
    ) -> bool:
        if self.completed_sim_s is not None:
            return True
        centered = (
            int(lane_id) == int(self.target_lane_id)
            and float(distance_to_center_m) <= self.center_threshold_m
        )
        if not centered:
            self.centered_since_sim_s = None
            return False
        if self.centered_since_sim_s is None:
            self.centered_since_sim_s = float(sim_time_s)
            return False
        if (
            float(sim_time_s) - self.centered_since_sim_s + 1e-9
            < self.settling_time_s
        ):
            return False
        self.completed_sim_s = float(sim_time_s)
        self.completed_wall_s = float(wall_time_s)
        return True


def latest_telemetry(client: httpx.Client, api_base: str) -> dict[str, dict]:
    response = client.get(f"{api_base}/telemetry.jsonl")
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return {}
    return rows[-1].get("vehicles") or {}


def wait_for_vehicle(
    client: httpx.Client,
    api_base: str,
    veh_id: int,
    *,
    minimum_speed_kmh: float = 10.5,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        telemetry = latest_telemetry(client, api_base)
        last = telemetry.get(str(veh_id)) or {}
        if (
            float(last.get("speed_kmh") or 0.0) >= minimum_speed_kmh
            and last.get("lane_id") is not None
            and last.get("sim_time_s") is not None
            and not last.get("is_junction")
        ):
            return last
        time.sleep(0.05)
    raise TimeoutError(
        "vehicle did not become calibration-ready: "
        f"speed={last.get('speed_kmh')} junction={last.get('is_junction')}"
    )


def select_spawn_indices(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    minimum_forward_m: float,
) -> list[int]:
    eligible = [
        row
        for row in rows
        if not row.get("is_junction")
        and row.get("left_lane_id") is not None
        and row.get("right_lane_id") is not None
        and float(row.get("forward_non_junction_m") or 0.0)
        >= minimum_forward_m
    ]
    eligible.sort(
        key=lambda row: (
            -float(row.get("forward_non_junction_m") or 0.0),
            int(row["index"]),
        )
    )
    selected: list[int] = []
    selected_roads: set[int] = set()
    for row in eligible:
        road_id = int(row["road_id"])
        if road_id in selected_roads:
            continue
        selected.append(int(row["index"]))
        selected_roads.add(road_id)
        if len(selected) == count:
            return selected
    for row in eligible:
        index = int(row["index"])
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"only {len(selected)} eligible spawn points; requested {count}"
    )


def build_schedule(
    spawn_indices: Iterable[int],
    speeds_kmh: Iterable[float],
    directions: Iterable[str],
    repetitions: int,
    base_seed: int,
) -> list[CalibrationTrial]:
    trials = []
    ordinal = 0
    for spawn_index in spawn_indices:
        for speed in speeds_kmh:
            for direction in directions:
                for repetition in range(repetitions):
                    ordinal += 1
                    trials.append(
                        CalibrationTrial(
                            trial_id=(
                                f"spawn-{spawn_index}-speed-{speed:g}-"
                                f"{direction}-r{repetition + 1:02d}"
                            ),
                            spawn_index=int(spawn_index),
                            direction=direction,
                            target_speed_kmh=float(speed),
                            seed=base_seed + ordinal,
                        )
                    )
    random.Random(base_seed).shuffle(trials)
    return trials


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024.0 * 1024.0
    )


def run_trial(
    client: httpx.Client,
    api_base: str,
    trial: CalibrationTrial,
    output_dir: Path,
    budget: CalibrationBudget,
    minimum_command_speed_kmh: float = 10.5,
    readiness_timeout_s: float = 15.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = output_dir / "trajectory.jsonl"
    client.post(f"{api_base}/reset", json={}).raise_for_status()
    response = client.post(
        f"{api_base}/config",
        json={
            "num_cars": 1,
            "spawn_indices": [trial.spawn_index],
            "initial_speeds": [trial.target_speed_kmh],
            "coordination_mode": "IA",
            "scenario_id": f"LANE_CHANGE_CALIBRATION_{trial.trial_id}",
            "seed": trial.seed,
            "persist_frames": False,
            "enable_cameras": False,
        },
    )
    response.raise_for_status()
    veh_id = int(response.json()["veh_ids"][0])
    initial = wait_for_vehicle(
        client,
        api_base,
        veh_id,
        minimum_speed_kmh=minimum_command_speed_kmh,
        timeout_s=readiness_timeout_s,
    )
    initial_lane_id = int(initial["lane_id"])
    initial_sequence = int(
        (initial.get("lane_change") or {}).get("request_sequence") or 0
    )
    command_wall_s = time.time()
    command_sim_s = float(initial["sim_time_s"])
    client.post(
        f"{api_base}/command",
        json={"cmd": "lane", "veh_id": veh_id, "dir": trial.direction},
    ).raise_for_status()

    deadline = command_wall_s + budget.trial_timeout_s
    target_lane_id: Optional[int] = None
    request_started_wall_s: Optional[float] = None
    request_started_sim_s: Optional[float] = None
    force_issued_wall_s: Optional[float] = None
    force_issued_sim_s: Optional[float] = None
    physical_arrival_wall_s: Optional[float] = None
    physical_arrival_sim_s: Optional[float] = None
    lane_id_arrival_wall_s: Optional[float] = None
    lane_id_arrival_sim_s: Optional[float] = None
    completion_tracker: Optional[PhysicalCompletionTracker] = None
    maximum_collision_count = 0
    last: dict[str, Any] = initial

    while time.time() < deadline:
        telemetry = latest_telemetry(client, api_base)
        row = telemetry.get(str(veh_id)) or {}
        if not row:
            time.sleep(0.05)
            continue
        last = row
        now_wall_s = time.time()
        append_jsonl(
            trajectory_path,
            {
                "wall_s": now_wall_s,
                "elapsed_wall_s": now_wall_s - command_wall_s,
                "vehicle": row,
            },
        )
        lane_change = row.get("lane_change") or {}
        sequence = int(lane_change.get("request_sequence") or 0)
        if sequence > initial_sequence:
            if target_lane_id is None and lane_change.get("target_lane_id") is not None:
                target_lane_id = int(lane_change["target_lane_id"])
                completion_tracker = PhysicalCompletionTracker(target_lane_id)
            request_started_wall_s = (
                request_started_wall_s or lane_change.get("request_started_at")
            )
            request_started_sim_s = (
                request_started_sim_s or lane_change.get("request_started_sim_s")
            )
            force_issued_wall_s = (
                force_issued_wall_s or lane_change.get("force_issued_at")
            )
            force_issued_sim_s = (
                force_issued_sim_s or lane_change.get("force_issued_sim_s")
            )
        maximum_collision_count = max(
            maximum_collision_count, int(row.get("collision_count") or 0)
        )
        if maximum_collision_count:
            break
        if target_lane_id is not None and int(row["lane_id"]) == target_lane_id:
            if lane_id_arrival_wall_s is None:
                lane_id_arrival_wall_s = now_wall_s
                lane_id_arrival_sim_s = float(row["sim_time_s"])
        if completion_tracker is not None and completion_tracker.observe(
            lane_id=int(row["lane_id"]),
            distance_to_center_m=float(row.get("distance_to_center") or 0.0),
            sim_time_s=float(row["sim_time_s"]),
            wall_time_s=now_wall_s,
        ):
            physical_arrival_wall_s = completion_tracker.completed_wall_s
            physical_arrival_sim_s = completion_tracker.completed_sim_s
            if lane_change.get("last_terminal_state") is not None:
                break
        time.sleep(0.05)

    lane_change = last.get("lane_change") or {}
    request_observed = int(lane_change.get("request_sequence") or 0) > initial_sequence
    physical_success = physical_arrival_wall_s is not None
    fsm_success = lane_change.get("last_terminal_state") == "DONE"
    status = (
        "collision"
        if maximum_collision_count
        else "completed"
        if physical_success
        else "not_started"
        if not request_observed
        else "timeout"
    )
    result = {
        **asdict(trial),
        "status": status,
        "veh_id": veh_id,
        "initial_lane_id": initial_lane_id,
        "target_lane_id": target_lane_id,
        "request_observed": request_observed,
        "physical_success": physical_success,
        "fsm_success": fsm_success,
        "fsm_terminal_state": lane_change.get("last_terminal_state"),
        "fsm_terminal_reason": lane_change.get("last_terminal_reason"),
        "fsm_reported_duration_wall_s": lane_change.get("last_duration_s"),
        "fsm_reported_duration_sim_s": lane_change.get("last_duration_sim_s"),
        "command_wall_s": command_wall_s,
        "command_sim_s": command_sim_s,
        "command_speed_kmh": float(initial["speed_kmh"]),
        "command_s_m": initial.get("s_m"),
        "request_started_wall_s": request_started_wall_s,
        "request_started_sim_s": request_started_sim_s,
        "force_issued_wall_s": force_issued_wall_s,
        "force_issued_sim_s": force_issued_sim_s,
        "physical_arrival_wall_s": physical_arrival_wall_s,
        "physical_arrival_sim_s": physical_arrival_sim_s,
        "lane_id_arrival_wall_s": lane_id_arrival_wall_s,
        "lane_id_arrival_sim_s": lane_id_arrival_sim_s,
        "physical_completion_definition": (
            "target lane and distance_to_center <= 0.5 m continuously for "
            "0.75 simulation seconds"
        ),
        "physical_duration_wall_s": (
            None
            if physical_arrival_wall_s is None
            else physical_arrival_wall_s - command_wall_s
        ),
        "physical_duration_sim_s": (
            None
            if physical_arrival_sim_s is None
            else physical_arrival_sim_s - command_sim_s
        ),
        "collision_count": maximum_collision_count,
        "final_lane_id": last.get("lane_id"),
        "final_speed_kmh": last.get("speed_kmh"),
        "final_road_id": last.get("road_id"),
        "final_is_junction": last.get("is_junction"),
        "trajectory_samples": (
            len(trajectory_path.read_text(encoding="utf-8").splitlines())
            if trajectory_path.exists()
            else 0
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        default="data/experiments/carla_lane_change_calibration_20260827",
    )
    parser.add_argument("--spawn-indices", default="auto")
    parser.add_argument("--spawn-count", type=int, default=3)
    parser.add_argument("--minimum-forward-m", type=float, default=80.0)
    parser.add_argument("--speeds-kmh", default="20,35,50")
    parser.add_argument("--directions", default="left,right")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=2026082800)
    parser.add_argument("--trial-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--minimum-command-speed-kmh",
        type=float,
        default=10.5,
        help="Wait for this observed speed before issuing the lane command.",
    )
    parser.add_argument("--readiness-timeout-s", type=float, default=15.0)
    parser.add_argument("--sweep-timeout-s", type=float, default=900.0)
    parser.add_argument("--maximum-output-mb", type=float, default=100.0)
    parser.add_argument("--maximum-rss-mb", type=float, default=2048.0)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    budget = CalibrationBudget(
        trial_timeout_s=args.trial_timeout_s,
        sweep_timeout_s=args.sweep_timeout_s,
        maximum_output_mb=args.maximum_output_mb,
        maximum_rss_mb=args.maximum_rss_mb,
        maximum_consecutive_failures=args.maximum_consecutive_failures,
    )
    api_base = args.api_base.rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        client.get(f"{api_base}/health").raise_for_status()
        spawn_rows = client.get(f"{api_base}/spawns").json()["spawns"]
        if args.spawn_indices == "auto":
            spawn_indices = select_spawn_indices(
                spawn_rows,
                count=args.spawn_count,
                minimum_forward_m=args.minimum_forward_m,
            )
        else:
            spawn_indices = [
                int(value)
                for value in args.spawn_indices.split(",")
                if value.strip()
            ]
        speeds = [float(value) for value in args.speeds_kmh.split(",")]
        directions = [value.strip() for value in args.directions.split(",")]
        if not set(directions).issubset({"left", "right"}):
            raise ValueError("directions must be left and/or right")
        schedule = build_schedule(
            spawn_indices,
            speeds,
            directions,
            args.repetitions,
            args.base_seed,
        )
        if args.limit is not None:
            schedule = schedule[: args.limit]
        manifest = {
            "schema_version": "1.1",
            "claim_scope": "exploratory CARLA executor calibration",
            "created_at_s": time.time(),
            "api_base": api_base,
            "spawn_indices": spawn_indices,
            "speeds_kmh": speeds,
            "directions": directions,
            "minimum_command_speed_kmh": args.minimum_command_speed_kmh,
            "readiness_timeout_s": args.readiness_timeout_s,
            "enable_cameras": False,
            "repetitions": args.repetitions,
            "base_seed": args.base_seed,
            "budget": asdict(budget),
            "persist_frames": False,
            "physical_completion": {
                "center_threshold_m": 0.5,
                "settling_time_sim_s": 0.75,
            },
            "schedule": [asdict(trial) for trial in schedule],
            "status": "dry_run" if args.dry_run else "running",
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.dry_run:
            print(
                f"dry-run: {len(schedule)} trials, spawns={spawn_indices}, "
                f"speeds={speeds}, directions={directions}"
            )
            return

        started = time.time()
        results = []
        consecutive_failures = 0
        for ordinal, trial in enumerate(schedule, start=1):
            if time.time() - started >= budget.sweep_timeout_s:
                manifest["status"] = "sweep_timeout"
                break
            if directory_size_mb(output) >= budget.maximum_output_mb:
                manifest["status"] = "output_budget_exceeded"
                break
            if (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                >= budget.maximum_rss_mb
            ):
                manifest["status"] = "memory_budget_exceeded"
                break
            result = run_trial(
                client,
                api_base,
                trial,
                output / trial.trial_id,
                budget,
                args.minimum_command_speed_kmh,
                args.readiness_timeout_s,
            )
            results.append(result)
            append_jsonl(output / "results.jsonl", result)
            if result["status"] == "completed":
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            print(
                f"[{ordinal}/{len(schedule)}] {trial.trial_id}: "
                f"{result['status']} wall={result['physical_duration_wall_s']} "
                f"sim={result['physical_duration_sim_s']} "
                f"fsm={result['fsm_terminal_reason']}"
            )
            if consecutive_failures >= budget.maximum_consecutive_failures:
                manifest["status"] = "consecutive_failure_limit"
                break
        else:
            manifest["status"] = "complete"
        client.post(f"{api_base}/reset", json={}).raise_for_status()
        if results:
            write_csv(output / "results.csv", results)
        manifest.update(
            {
                "completed_trials": len(results),
                "elapsed_wall_s": time.time() - started,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
                "output_mb": directory_size_mb(output),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
