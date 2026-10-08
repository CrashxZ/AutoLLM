#!/usr/bin/env python3
"""Prepare and validate the frozen CARLA paired-comparison lock."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from scripts.coordination_v2_freeze import (
    file_sha256,
    registration_freeze_commit,
    relative_path,
    verify_registration,
)
from scripts.plan_carla_paired_comparison import validate_schedule


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_STATUS = "frozen_confirmatory"
SOURCE_FILES = (
    "marl/fleet_mappo.py",
    "marl/highway_env.py",
    "marl/numpy_policy.py",
    "scripts/analyze_carla_paired_comparison.py",
    "scripts/analyze_carla_paired_coordination.py",
    "scripts/carla_paired_freeze.py",
    "scripts/plan_carla_paired_comparison.py",
    "scripts/run_carla_paired_coordination.py",
    "server/lane_change.py",
    "server/mec.py",
    "server/server.py",
)


def source_paths() -> list[Path]:
    paths = {REPO_ROOT / value for value in SOURCE_FILES}
    paths.update((REPO_ROOT / "server" / "coordination").glob("*.py"))
    selected = sorted(path for path in paths if path.is_file())
    if not selected:
        raise ValueError("CARLA paired source manifest is empty")
    return selected


def source_hashes(paths: Iterable[Path] | None = None) -> dict[str, str]:
    selected = source_paths() if paths is None else sorted(paths)
    return {relative_path(path): file_sha256(path) for path in selected}


def installed_packages() -> list[str]:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    return sorted(
        {line.strip() for line in output.splitlines() if line.strip()},
        key=str.casefold,
    )


def environment_packages(path: Path) -> list[str]:
    return sorted(
        {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        },
        key=str.casefold,
    )


def verify_environment(path: Path) -> None:
    expected = environment_packages(path)
    actual = installed_packages()
    if actual != expected:
        raise ValueError(
            "installed environment differs from frozen lock: "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )


def prepare_schedule(draft_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    schedule = json.loads(draft_path.read_text(encoding="utf-8"))
    validate_schedule(schedule)
    if schedule.get("claim_status") != "draft_not_frozen":
        raise ValueError("only an unfrozen draft can be prepared")
    final = deepcopy(schedule)
    final["schema_version"] = "1.0"
    final["claim_status"] = CONFIRMATORY_STATUS
    final["label"] = "carla_paired_coordination_confirmatory_current_executor"
    validate_schedule(final)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def build_lock(
    *, schedule: Path, registration: Path, environment_lock: Path
) -> dict:
    payload = json.loads(schedule.read_text(encoding="utf-8"))
    validate_schedule(payload)
    if payload.get("claim_status") != CONFIRMATORY_STATUS:
        raise ValueError("schedule is not frozen_confirmatory")
    actor = REPO_ROOT / payload["mappo_actor"]
    ranker = REPO_ROOT / payload["mind_ranker"]
    for path in (schedule, registration, environment_lock, actor, ranker):
        if not path.exists():
            raise FileNotFoundError(path)
    verify_environment(environment_lock)
    freeze_commit = registration_freeze_commit(registration)
    verify_registration(registration, freeze_commit)
    return {
        "label": "CARLA_PAIRED_CURRENT_EXECUTOR_FROZEN_LOCK",
        "claim_status": CONFIRMATORY_STATUS,
        "created_at_unix_s": time.time(),
        "registration": {
            "path": relative_path(registration),
            "freeze_commit": freeze_commit,
        },
        "schedule": {
            "path": relative_path(schedule),
            "sha256": file_sha256(schedule),
        },
        "environment": {
            "path": relative_path(environment_lock),
            "sha256": file_sha256(environment_lock),
        },
        "models": {
            "mappo_actor": {
                "path": relative_path(actor),
                "sha256": file_sha256(actor),
            },
            "mind_ranker": {
                "path": relative_path(ranker),
                "sha256": file_sha256(ranker),
            },
        },
        "sources": source_hashes(),
    }


def validate_lock(
    *,
    schedule: Path,
    registration: Path,
    environment_lock: Path,
    lock_path: Path,
) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("claim_status") != CONFIRMATORY_STATUS:
        raise ValueError("confirmatory lock has the wrong status")
    for label, path in (("schedule", schedule), ("environment", environment_lock)):
        record = lock[label]
        if relative_path(path) != record["path"]:
            raise ValueError(f"{label} path mismatch")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"{label} checksum mismatch")
    registration_record = lock["registration"]
    if relative_path(registration) != registration_record["path"]:
        raise ValueError("registration path mismatch")
    verify_registration(registration, registration_record["freeze_commit"])
    verify_environment(environment_lock)
    for label, record in lock["models"].items():
        path = REPO_ROOT / record["path"]
        if not path.exists() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"{label} checksum mismatch")
    actual_sources = source_hashes()
    if actual_sources != lock.get("sources"):
        expected = lock.get("sources", {})
        changed = sorted(
            set(actual_sources) ^ set(expected)
            | {
                path
                for path in set(actual_sources) & set(expected)
                if actual_sources[path] != expected[path]
            }
        )
        raise ValueError(f"frozen source mismatch: {changed}")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--draft", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--schedule", type=Path, required=True)
    lock_parser.add_argument("--registration", type=Path, required=True)
    lock_parser.add_argument("--environment-lock", type=Path, required=True)
    lock_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_schedule(args.draft, args.output)
        print(json.dumps({"output": str(args.output), "status": "prepared"}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = build_lock(
        schedule=args.schedule,
        registration=args.registration,
        environment_lock=args.environment_lock,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    validate_lock(
        schedule=args.schedule,
        registration=args.registration,
        environment_lock=args.environment_lock,
        lock_path=args.output,
    )
    print(json.dumps({"output": str(args.output), "status": "valid"}))


if __name__ == "__main__":
    main()
