#!/usr/bin/env python3
"""Build and verify the immutable lock for coordination comparison v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_STATUS = "preregistered-confirmatory"
SOURCE_FILES = (
    "marl/__init__.py",
    "marl/benchmark_v2.py",
    "marl/benchmark_v2_policies.py",
    "marl/fleet_highway_env.py",
    "marl/fleet_mappo.py",
    "marl/highway_env.py",
    "marl/numpy_policy.py",
    "marl/procedural_scenarios.py",
    "scripts/analyze_coordination_comparison_v2.py",
    "scripts/coordination_v2_freeze.py",
    "scripts/run_coordination_benchmark_v2.py",
    "scripts/run_mindcav_v2_episodes.py",
    "server/__init__.py",
    "server/coordination/__init__.py",
    "server/coordination/audit.py",
    "server/coordination/candidate_ranking.py",
    "server/coordination/conflict_graph.py",
    "server/coordination/models.py",
    "server/coordination/orchestrator.py",
    "server/coordination/proposer.py",
    "server/coordination/transaction.py",
    "server/coordination/validator.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def source_paths() -> list[Path]:
    paths = {REPO_ROOT / value for value in SOURCE_FILES}
    output = sorted(path for path in paths if path.is_file())
    if not output:
        raise ValueError("coordination source manifest is empty")
    return output


def source_hashes(paths: Iterable[Path] | None = None) -> dict[str, str]:
    selected = source_paths() if paths is None else sorted(paths)
    return {relative_path(path): file_sha256(path) for path in selected}


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def registration_freeze_commit(registration: Path) -> str:
    relative = relative_path(registration)
    commits = git_output("log", "--format=%H", "--", relative).splitlines()
    if not commits:
        raise ValueError("pre-registration has no freeze commit")
    return commits[-1]


def strip_stamp(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not line.startswith("**Frozen at commit:**")
    ]
    return "\n".join(lines).rstrip() + "\n"


def registration_stamp(text: str) -> str:
    stamp_prefix = "**Frozen at commit:**"
    return next(
        (
            line[len(stamp_prefix) :].strip()
            for line in text.splitlines()
            if line.startswith(stamp_prefix)
        ),
        "",
    )


def verify_registration(registration: Path, expected_commit: str) -> None:
    actual_commit = registration_freeze_commit(registration)
    if actual_commit != expected_commit:
        raise ValueError("pre-registration freeze commit mismatch")
    relative = relative_path(registration)
    frozen = git_output("show", f"{actual_commit}:{relative}")
    current = registration.read_text(encoding="utf-8")
    if strip_stamp(frozen) != strip_stamp(current):
        raise ValueError("pre-registration changed after freeze")
    stamp = registration_stamp(current)
    if not stamp or not actual_commit.startswith(stamp):
        raise ValueError("pre-registration stamp does not match freeze commit")


def build_lock(
    *,
    schedule: Path,
    registration: Path,
    environment_lock: Path,
) -> dict:
    payload = json.loads(schedule.read_text(encoding="utf-8"))
    if payload.get("claim_status") != CONFIRMATORY_STATUS:
        raise ValueError("final schedule is not labelled preregistered-confirmatory")
    actor = REPO_ROOT / payload["mappo_actor"]
    ranker = REPO_ROOT / payload["mind_ranker"]
    for path in (schedule, registration, environment_lock, actor, ranker):
        if not path.exists():
            raise FileNotFoundError(path)
    freeze_commit = registration_freeze_commit(registration)
    verify_registration(registration, freeze_commit)
    return {
        "label": "COORDINATION_COMPARISON_V2_FROZEN_LOCK",
        "claim_status": CONFIRMATORY_STATUS,
        "created_at_s": time.time(),
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
        raise ValueError("confirmatory lock has the wrong claim status")
    for label, path in (
        ("schedule", schedule),
        ("environment", environment_lock),
    ):
        expected = lock[label]
        if relative_path(path) != expected["path"]:
            raise ValueError(f"{label} path mismatch")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"{label} checksum mismatch")
    registration_record = lock["registration"]
    if relative_path(registration) != registration_record["path"]:
        raise ValueError("pre-registration path mismatch")
    verify_registration(registration, registration_record["freeze_commit"])
    for label, record in lock["models"].items():
        path = REPO_ROOT / record["path"]
        if not path.exists() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"{label} checksum mismatch")
    actual_sources = source_hashes()
    if actual_sources != lock.get("sources"):
        expected_sources = lock.get("sources", {})
        changed = sorted(
            set(actual_sources) ^ set(expected_sources)
            | {
                path
                for path in set(actual_sources) & set(expected_sources)
                if actual_sources[path] != expected_sources[path]
            }
        )
        raise ValueError(f"frozen source mismatch: {changed}")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite lock: {args.output}")
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
    print(json.dumps({"output": str(args.output), "status": "valid"}, indent=2))


if __name__ == "__main__":
    main()
