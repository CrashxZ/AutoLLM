#!/usr/bin/env python3
"""Run exploratory pilots or a frozen paired CARLA coordination schedule.

The runner keeps the physical executor and local maneuver requests common while
switching only the coordination method.  It is intentionally separate from the
dashboard and does not persist camera frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marl.carla_aligned_mappo import ALIGNED_MINIMUM_CLEARANCE_M
from marl.fleet_mappo import (
    EGO_FEATURE_NAMES,
    MAX_NEIGHBORS,
    NEIGHBOR_FEATURE_NAMES,
    OBSERVATION_DIM,
    PRIORITY_OBSERVATION_DIM,
)
from marl.highway_env import ACCELERATE, KEEP_LANE, LANE_LEFT, LANE_RIGHT, YIELD
from marl.numpy_policy import NumpyActorPolicy
from server.coordination.legacy_adapter import telemetry_to_states
from server.coordination.models import Action, JointPlan, PlanStep, VehiclePlan
from server.coordination.validator import DeterministicPlanValidator, ValidationConfig
from scripts.calibrate_carla_lane_changes import PhysicalCompletionTracker
from scripts.carla_paired_freeze import validate_lock as validate_frozen_lock
from scripts.plan_carla_paired_comparison import (
    FLEET_SIZES,
    METHODS,
    SCENARIO_FAMILIES,
    PlanningConfig,
    build_schedule,
    validate_schedule,
)
from scripts.run_carla_guarded_transfer import (
    decision_plans,
    first_steps,
    post_outcome,
    steps_complete,
)
from scripts.run_carla_multi_vehicle_executor import (
    PlannedCommand,
    TrialLayout,
    append_jsonl,
    build_layout,
    expected_target_lane,
    latest_telemetry,
    pair_distance_metrics,
)


PILOT_SEED_BASE = 6_100_000
FLOW_SPEED_KMH = 50.0
MAPPO_CONTROL_PERIOD_S = 0.5
MAPPO_ACCEL_MPS2 = 2.0
MAPPO_YIELD_DECEL_MPS2 = 3.0
MINIMUM_MERGE_CLEARANCE_M = 6.25
CONFLICT_DISTANCE_M = 50.0
TTC_THRESHOLD_S = 4.0
CENTER_THRESHOLD_M = 0.5
GOAL_PRIORITY_HORIZON_S = 8.0
MIND_RANKER_METHODS = {
    "MIND_CAV",
    "MIND_CAV_DETERMINISTIC",
    "MIND_CAV_LEARNED",
}


@dataclass(frozen=True)
class RunnerBudget:
    trial_timeout_wall_s: float = 210.0
    trial_timeout_sim_s: float = 60.0
    ready_timeout_wall_s: float = 90.0
    polling_s: float = 0.03
    action_period_sim_s: float = MAPPO_CONTROL_PERIOD_S
    request_retry_sim_s: float = 0.75
    no_progress_retry_sim_s: float = 3.0
    max_command_attempts: int = 10
    simulation_step_ticks: int = 10


def layout_for_block(block: Mapping[str, Any]) -> TrialLayout:
    """Resolve a legacy named layout or an explicit, schedule-owned layout."""
    spawn_indices = block.get("spawn_indices")
    command_rows = block.get("commands")
    if spawn_indices is None and command_rows is None:
        return build_layout(
            str(block["executor_pattern"]), int(block["fleet_size"])
        )
    if spawn_indices is None or command_rows is None:
        raise ValueError("explicit layouts require spawn_indices and commands")
    layout = TrialLayout(
        spawn_indices=tuple(int(value) for value in spawn_indices),
        commands=tuple(
            PlannedCommand(
                slot=int(row["slot"]),
                direction=str(row["direction"]),
                issue_offset_sim_s=float(row.get("issue_offset_sim_s") or 0.0),
            )
            for row in command_rows
        ),
    )
    if len(layout.spawn_indices) != int(block["fleet_size"]):
        raise ValueError("explicit layout fleet size does not match spawn count")
    offset_rows = block.get("spawn_longitudinal_offsets_m")
    if offset_rows is None:
        placements = [(spawn_index, 0.0) for spawn_index in layout.spawn_indices]
    else:
        if len(offset_rows) != len(layout.spawn_indices):
            raise ValueError("explicit layout spawn offset count does not match fleet")
        placements = [
            (spawn_index, float(offset_m))
            for spawn_index, offset_m in zip(layout.spawn_indices, offset_rows)
        ]
    if len(set(placements)) != len(placements):
        raise ValueError("explicit layout contains duplicate physical spawns")
    if any(
        command.slot < 0 or command.slot >= len(layout.spawn_indices)
        for command in layout.commands
    ):
        raise ValueError("explicit layout command slot is out of range")
    return layout


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_file_checksums() -> dict[str, str]:
    paths = {
        REPO_ROOT / "scripts" / "run_carla_paired_coordination.py",
        REPO_ROOT / "scripts" / "plan_carla_paired_comparison.py",
        REPO_ROOT / "scripts" / "analyze_carla_paired_coordination.py",
        REPO_ROOT / "scripts" / "analyze_carla_paired_comparison.py",
        REPO_ROOT / "server" / "server.py",
        REPO_ROOT / "server" / "lane_change.py",
        REPO_ROOT / "server" / "mec.py",
        REPO_ROOT / "marl" / "fleet_mappo.py",
        REPO_ROOT / "marl" / "highway_env.py",
        REPO_ROOT / "marl" / "numpy_policy.py",
    }
    paths.update((REPO_ROOT / "server" / "coordination").glob("*.py"))
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in sorted(paths)
        if path.exists()
    }


def build_provenance(actor_path: Path) -> dict[str, Any]:
    packages = sorted(
        {
            line.strip()
            for line in subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze", "--all"], text=True
            ).splitlines()
            if line.strip()
        },
        key=str.casefold,
    )
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    ranker_path = Path(
        os.environ.get(
            "MIND_CAV_RANKER_MODEL",
            REPO_ROOT / "data" / "models" / "candidate_ranker.npz",
        )
    )
    model_checksums = {str(actor_path.resolve()): sha256_file(actor_path)}
    if ranker_path.exists():
        model_checksums[str(ranker_path.resolve())] = sha256_file(ranker_path)
    return {
        "schema_version": "1.0",
        "created_at_unix_s": time.time(),
        "repository": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty": bool(status and status != "unavailable"),
            "tracked_diff_sha256": hashlib.sha256(
                _git_output("diff", "--binary", "HEAD").encode()
            ).hexdigest(),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "packages_sha256": sha256_json(packages),
        },
        "source_files_sha256": source_file_checksums(),
        "models_sha256": model_checksums,
    }


def lane_position(row: Mapping[str, Any], num_lanes: int = 4) -> float:
    """Map Town04 lanes to the MAPPO left-to-right continuous lane position."""
    lane_id = row.get("lane_id")
    if lane_id is None:
        return 0.0
    center_index = row.get("driving_lane_index")
    if center_index is None:
        if int(lane_id) > 0:
            raise ValueError(
                "positive OpenDRIVE lanes require driving_lane_index telemetry"
            )
        center_index = abs(int(lane_id)) - 1
    lateral_offset = float(row.get("d_m") or 0.0)
    return float(
        np.clip(float(center_index) - lateral_offset / 3.5, 0, num_lanes - 1)
    )


def lane_index(
    lane_id: int,
    driving_lane_ids: Optional[Iterable[int]] = None,
    num_lanes: int = 4,
) -> int:
    if driving_lane_ids is not None:
        ordered = [int(value) for value in driving_lane_ids]
        if int(lane_id) not in ordered:
            raise ValueError(f"lane {lane_id} is absent from driving_lane_ids")
        index = ordered.index(int(lane_id))
    elif int(lane_id) < 0:
        index = abs(int(lane_id)) - 1
    else:
        raise ValueError(
            "positive OpenDRIVE lanes require driving_lane_ids telemetry"
        )
    if index < 0 or index >= num_lanes:
        raise ValueError(f"driving lane index {index} is outside MAPPO domain")
    return index


def corridor_departures(
    telemetry: Mapping[str, Mapping[str, Any]],
    initial_lane_domains: Mapping[int, Iterable[int]],
) -> list[dict[str, Any]]:
    """Return vehicles that left their registered driving-lane domain."""
    departures: list[dict[str, Any]] = []
    for veh_id, expected_values in initial_lane_domains.items():
        row = telemetry.get(str(veh_id))
        if row is None:
            continue
        expected = [int(value) for value in expected_values]
        observed = [int(value) for value in row.get("driving_lane_ids") or []]
        lane_id = int(row.get("lane_id") or 0)
        if observed and set(observed) == set(expected) and lane_id in expected:
            continue
        departures.append(
            {
                "veh_id": int(veh_id),
                "road_id": row.get("road_id"),
                "lane_id": lane_id,
                "initial_driving_lane_ids": expected,
                "observed_driving_lane_ids": observed,
            }
        )
    return departures


def longitudinal_delta_m(
    ego: Mapping[str, Any], other: Mapping[str, Any]
) -> float:
    if (
        ego.get("road_id") == other.get("road_id")
        and ego.get("s_m") is not None
        and other.get("s_m") is not None
    ):
        raw_delta = float(other["s_m"]) - float(ego["s_m"])
        # Positive OpenDRIVE lanes travel opposite increasing road ``s``.
        # Return a vehicle-aligned delta: positive means ahead for either
        # carriageway direction.
        lane_id = int(ego.get("lane_id") or 0)
        return raw_delta if lane_id < 0 else -raw_delta
    try:
        dx = float(other["pose"]["x"]) - float(ego["pose"]["x"])
        dy = float(other["pose"]["y"]) - float(ego["pose"]["y"])
        yaw = math.radians(float(ego["pose"].get("yaw") or 0.0))
        return dx * math.cos(yaw) + dy * math.sin(yaw)
    except (KeyError, TypeError, ValueError):
        return 150.0


def occupied_lanes(row: Mapping[str, Any]) -> set[int]:
    raw = row.get("occupied_lane_ids") or (row.get("lane_change") or {}).get(
        "occupied_lane_ids"
    )
    if raw:
        return {int(value) for value in raw}
    lane = row.get("lane_id")
    return {int(lane)} if lane is not None else set()


def bumper_clearance_m(
    first: Mapping[str, Any], second: Mapping[str, Any], vehicle_length_m: float = 4.7
) -> float:
    return max(abs(longitudinal_delta_m(first, second)) - vehicle_length_m, 0.0)


def pair_ttc_s(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    delta = longitudinal_delta_m(first, second)
    first_speed = float(first.get("speed_kmh") or 0.0) / 3.6
    second_speed = float(second.get("speed_kmh") or 0.0) / 3.6
    if delta >= 0.0:
        closing = first_speed - second_speed
    else:
        closing = second_speed - first_speed
    if closing <= 0.05:
        return float("inf")
    return bumper_clearance_m(first, second) / closing


def target_corridor_threats(
    command: Mapping[str, Any],
    telemetry: Mapping[str, Mapping[str, Any]],
) -> list[int]:
    ego_id = int(command["veh_id"])
    ego = telemetry[str(ego_id)]
    target = int(command["target_lane_id"])
    threats = []
    for key, other in telemetry.items():
        other_id = int(other.get("veh_id") or key)
        if other_id == ego_id or target not in occupied_lanes(other):
            continue
        if (
            bumper_clearance_m(ego, other) < MINIMUM_MERGE_CLEARANCE_M
            or pair_ttc_s(ego, other) < TTC_THRESHOLD_S
        ):
            threats.append(other_id)
    return threats


def command_conflict(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    telemetry: Mapping[str, Mapping[str, Any]],
) -> bool:
    same_target = first["target_lane_id"] == second["target_lane_id"]
    reciprocal = (
        first["target_lane_id"] == second["initial_lane_id"]
        and second["target_lane_id"] == first["initial_lane_id"]
    )
    if not (same_target or reciprocal):
        return False
    a = telemetry[str(first["veh_id"])]
    b = telemetry[str(second["veh_id"])]
    return (
        abs(longitudinal_delta_m(a, b)) <= CONFLICT_DISTANCE_M
        or pair_ttc_s(a, b) < TTC_THRESHOLD_S
    )


def initial_conflict_evidence(
    commands: Iterable[Mapping[str, Any]],
    telemetry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe conflict-active command pairs and occupied target corridors."""
    command_rows = list(commands)
    evidence: list[dict[str, Any]] = []
    for index, first in enumerate(command_rows):
        for second in command_rows[index + 1 :]:
            if command_conflict(first, second, telemetry):
                evidence.append(
                    {
                        "kind": "command_pair",
                        "command_ids": [
                            str(first["command_id"]),
                            str(second["command_id"]),
                        ],
                        "vehicle_ids": [
                            int(first["veh_id"]),
                            int(second["veh_id"]),
                        ],
                    }
                )
    for command in command_rows:
        for threat_id in target_corridor_threats(command, telemetry):
            evidence.append(
                {
                    "kind": "occupied_target_corridor",
                    "command_ids": [str(command["command_id"])],
                    "vehicle_ids": [int(command["veh_id"]), int(threat_id)],
                    "target_lane_id": int(command["target_lane_id"]),
                }
            )
    return evidence


def carla_actor_observations(
    telemetry: Mapping[str, Mapping[str, Any]],
    vehicle_ids: Iterable[int],
    commands_by_vehicle: Mapping[int, Mapping[str, Any]],
    *,
    current_sim_s: float,
    priority_order: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Build a legacy or priority-aware MAPPO input from CARLA telemetry."""
    ids = [int(value) for value in vehicle_ids]
    max_lane = 3.0
    flow_mps = FLOW_SPEED_KMH / 3.6
    include_priority = priority_order is not None
    if include_priority:
        ordered_slots = [int(value) for value in priority_order]
        if sorted(ordered_slots) != list(range(len(ids))):
            raise ValueError("priority_order must contain every vehicle slot once")
        priority_rank = {
            slot: rank / max(len(ids) - 1, 1)
            for rank, slot in enumerate(ordered_slots)
        }
    else:
        priority_rank = {}
    observation_dim = (
        PRIORITY_OBSERVATION_DIM if include_priority else OBSERVATION_DIM
    )
    neighbor_base = len(EGO_FEATURE_NAMES) + int(include_priority)
    observations = np.zeros((len(ids), observation_dim), dtype=np.float32)
    for ego_index, ego_id in enumerate(ids):
        ego = telemetry[str(ego_id)]
        command = commands_by_vehicle.get(ego_id)
        active = bool(
            command
            and current_sim_s >= float(command["activation_sim_s"])
            and not command["completed"]
        )
        lane = lane_position(ego)
        goal = (
            float(
                lane_index(
                    int(command["target_lane_id"]),
                    ego.get("driving_lane_ids"),
                )
            )
            if command is not None and not command["completed"]
            else float(round(lane))
        )
        changing = bool((ego.get("lane_change") or {}).get("state") not in {None, "IDLE", "DONE", "ABORT"})
        destination = (
            float(
                lane_index(
                    int((ego.get("lane_change") or {}).get("target_lane_id")),
                    ego.get("driving_lane_ids"),
                )
            )
            if changing and (ego.get("lane_change") or {}).get("target_lane_id") is not None
            else float(round(lane))
        )
        ego_features = np.asarray(
            [
                lane / max_lane,
                (float(ego.get("speed_kmh") or 0.0) / 3.6) / flow_mps,
                goal / max_lane,
                (goal - lane) / max_lane,
                destination / max_lane,
                float(changing),
                float(active),
                0.0,
                float(bool(command and command["completed"])),
                1.0,
            ],
            dtype=np.float32,
        )
        observations[ego_index, : len(EGO_FEATURE_NAMES)] = ego_features
        if include_priority:
            observations[ego_index, len(EGO_FEATURE_NAMES)] = priority_rank[
                ego_index
            ]
        others = [other_id for other_id in ids if other_id != ego_id]
        others.sort(
            key=lambda other_id: (
                abs(longitudinal_delta_m(ego, telemetry[str(other_id)])),
                other_id,
            )
        )
        for slot, other_id in enumerate(others[:MAX_NEIGHBORS]):
            other = telemetry[str(other_id)]
            other_command = commands_by_vehicle.get(other_id)
            other_active = bool(
                other_command
                and current_sim_s >= float(other_command["activation_sim_s"])
                and not other_command["completed"]
            )
            other_lane = lane_position(other)
            other_changing = bool((other.get("lane_change") or {}).get("state") not in {None, "IDLE", "DONE", "ABORT"})
            other_destination = (
                float(
                    lane_index(
                        int((other.get("lane_change") or {}).get("target_lane_id")),
                        other.get("driving_lane_ids"),
                    )
                )
                if other_changing and (other.get("lane_change") or {}).get("target_lane_id") is not None
                else float(round(other_lane))
            )
            other_goal = (
                float(
                    lane_index(
                        int(other_command["target_lane_id"]),
                        other.get("driving_lane_ids"),
                    )
                )
                if other_command is not None and not other_command["completed"]
                else float(round(other_lane))
            )
            offset = neighbor_base + slot * len(NEIGHBOR_FEATURE_NAMES)
            features = np.asarray(
                [
                    np.clip(longitudinal_delta_m(ego, other), -150.0, 150.0) / 150.0,
                    (other_lane - lane) / max_lane,
                    np.clip(
                        (float(other.get("speed_kmh") or 0.0) - float(ego.get("speed_kmh") or 0.0)) / 3.6,
                        -20.0,
                        20.0,
                    )
                    / 20.0,
                    (other_destination - destination) / max_lane,
                    (other_goal - other_lane) / max_lane,
                    float(other_active),
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            observations[ego_index, offset : offset + len(features)] = features
    if observations.shape != (len(ids), observation_dim):
        raise AssertionError("invalid MAPPO CARLA observation shape")
    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("non-finite MAPPO CARLA observation")
    return observations


def carla_joint_plan_from_actions(
    actions: np.ndarray,
    telemetry: Mapping[str, Mapping[str, Any]],
    vehicle_ids: Iterable[int],
    *,
    transaction_id: str,
) -> JointPlan:
    """Encode a CARLA fleet action using the same typed validator contract."""
    ids = [int(value) for value in vehicle_ids]
    requested = np.asarray(actions, dtype=np.int64)
    if requested.shape != (len(ids),):
        raise ValueError("actions must contain one value per vehicle")
    protocol_actions = {
        KEEP_LANE: Action.KEEP_LANE,
        LANE_LEFT: Action.LANE_LEFT,
        LANE_RIGHT: Action.LANE_RIGHT,
        ACCELERATE: Action.ACCELERATE,
        YIELD: Action.YIELD,
    }
    plans = {}
    for slot, veh_id in enumerate(ids):
        row = telemetry[str(veh_id)]
        action_id = int(requested[slot])
        lane_id = int(row["lane_id"])
        lane_change = row.get("lane_change") or {}
        changing = lane_change.get("state") not in {None, "IDLE", "DONE", "ABORT"}
        if changing and lane_change.get("target_lane_id") is not None:
            target_lane_id = int(lane_change["target_lane_id"])
        elif action_id == LANE_LEFT:
            target_lane_id = expected_target_lane(lane_id, "left")
        elif action_id == LANE_RIGHT:
            target_lane_id = expected_target_lane(lane_id, "right")
        else:
            target_lane_id = lane_id
        speed_kmh = float(row.get("speed_kmh") or 0.0)
        target_speed_kmh = mappo_target_speed_kmh(action_id, speed_kmh)
        plans[veh_id] = VehiclePlan(
            veh_id=veh_id,
            summary="MAPPO adapted action",
            horizon_s=8.0,
            steps=[
                PlanStep(
                    action=protocol_actions[action_id],
                    target_lane_id=target_lane_id,
                    target_speed_kmh=target_speed_kmh,
                    duration_s=(
                        3.0
                        if action_id in {LANE_LEFT, LANE_RIGHT}
                        else MAPPO_CONTROL_PERIOD_S
                    ),
                )
            ],
        )
    return JointPlan(
        transaction_id=transaction_id,
        plans=plans,
        summary="MAPPO adapted joint action",
        proposer="mappo-adapted",
    )


def carla_validator_masked_argmax(
    logits: np.ndarray,
    telemetry: Mapping[str, Mapping[str, Any]],
    vehicle_ids: Iterable[int],
    commands_by_vehicle: Mapping[int, Mapping[str, Any]],
    priority_order: Iterable[int],
    validator: DeterministicPlanValidator,
    *,
    current_sim_s: float,
    allow_cooperative_support: bool = False,
    goal_directed_lane_actions: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the deterministic action mask used to train/evaluate MAPPO."""
    ids = [int(value) for value in vehicle_ids]
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (len(ids), 5):
        raise ValueError("MAPPO logits have an invalid shape")
    states = telemetry_to_states(dict(telemetry))
    now_s = max(state.observed_at_s for state in states.values())
    actions = np.full(len(ids), KEEP_LANE, dtype=np.int64)
    masks = np.zeros_like(scores, dtype=bool)
    for slot in (int(value) for value in priority_order):
        veh_id = ids[slot]
        row = telemetry[str(veh_id)]
        command = commands_by_vehicle.get(veh_id)
        active = bool(
            command
            and current_sim_s >= float(command["activation_sim_s"])
            and not command["completed"]
        )
        changing = (row.get("lane_change") or {}).get("state") not in {
            None,
            "IDLE",
            "DONE",
            "ABORT",
        }
        mask = np.zeros(5, dtype=bool)
        mask[KEEP_LANE] = True
        if not changing and (active or allow_cooperative_support):
            candidates = [ACCELERATE, YIELD]
            current_lane_index = lane_index(
                int(row["lane_id"]), row.get("driving_lane_ids")
            )
            if active and goal_directed_lane_actions:
                target_lane_index = lane_index(
                    int(command["target_lane_id"]),
                    row.get("driving_lane_ids"),
                )
                if target_lane_index < current_lane_index:
                    candidates.append(LANE_LEFT)
                elif target_lane_index > current_lane_index:
                    candidates.append(LANE_RIGHT)
            elif active:
                if current_lane_index > 0:
                    candidates.append(LANE_LEFT)
                if current_lane_index < 3:
                    candidates.append(LANE_RIGHT)
            for action_id in candidates:
                candidate = actions.copy()
                candidate[slot] = action_id
                plan = carla_joint_plan_from_actions(
                    candidate,
                    telemetry,
                    ids,
                    transaction_id=f"mappo-mask-{int(current_sim_s * 1000)}-{slot}-{action_id}",
                )
                mask[action_id] = validator.validate(
                    plan, states, now_s=now_s
                ).safe
        masks[slot] = mask
        actions[slot] = int(np.argmax(np.where(mask, scores[slot], -np.inf)))
    return actions, masks


def post_lane_command(
    client: httpx.Client, api_base: str, command: dict[str, Any]
) -> None:
    response = client.post(
        f"{api_base}/command",
        json={
            "cmd": "lane",
            "veh_id": int(command["veh_id"]),
            "dir": command["direction"],
        },
    )
    response.raise_for_status()
    command["attempts"] += 1


def post_speed(
    client: httpx.Client, api_base: str, veh_id: int, speed_kmh: float
) -> None:
    response = client.post(
        f"{api_base}/command",
        json={"cmd": "speed", "veh_id": int(veh_id), "kmh": float(speed_kmh)},
    )
    response.raise_for_status()


def mappo_target_speed_kmh(action: int, current_speed_kmh: float) -> float:
    """Translate one MAPPO action using its training-time acceleration bounds."""
    current = max(0.0, float(current_speed_kmh))
    if int(action) == ACCELERATE:
        delta = MAPPO_ACCEL_MPS2 * MAPPO_CONTROL_PERIOD_S * 3.6
        return min(90.0, current + delta)
    if int(action) == YIELD:
        delta = MAPPO_YIELD_DECEL_MPS2 * MAPPO_CONTROL_PERIOD_S * 3.6
        return max(0.0, current - delta)
    return FLOW_SPEED_KMH


def execute_protocol_step(
    client: httpx.Client,
    api_base: str,
    veh_id: int,
    step: Mapping[str, Any],
    telemetry: Mapping[str, Mapping[str, Any]],
) -> str:
    action = str(step.get("action") or "hold").lower()
    if action in {"lane_left", "lane_right"}:
        if step.get("target_speed_kmh") is not None:
            post_speed(
                client,
                api_base,
                veh_id,
                float(step["target_speed_kmh"]),
            )
        response = client.post(
            f"{api_base}/command",
            json={
                "cmd": "lane",
                "veh_id": int(veh_id),
                "dir": "left" if action == "lane_left" else "right",
            },
        )
        response.raise_for_status()
    elif action in {"accelerate", "speed_up"}:
        current = float((telemetry.get(str(veh_id)) or {}).get("speed_kmh") or 0.0)
        target = step.get("target_speed_kmh")
        post_speed(
            client,
            api_base,
            veh_id,
            float(target) if target is not None else min(90.0, current + 10.0),
        )
    elif action in {"yield", "speed_down"}:
        current = float((telemetry.get(str(veh_id)) or {}).get("speed_kmh") or 0.0)
        target = step.get("target_speed_kmh")
        post_speed(
            client,
            api_base,
            veh_id,
            float(target) if target is not None else max(0.0, current - 15.0),
        )
    elif step.get("target_speed_kmh") is not None:
        post_speed(client, api_base, veh_id, float(step["target_speed_kmh"]))
    return action


def diagnostic_executable_steps(
    steps: Mapping[int, dict[str, Any]],
    requesting_vehicle_id: int,
    *,
    suppress_joint_participants: bool,
) -> dict[int, dict[str, Any]]:
    """Optionally isolate the requesting vehicle for attribution experiments.

    The default preserves the protocol exactly.  Suppression deliberately
    violates joint-plan execution and must only be used for a labeled
    diagnostic run, never as a coordination result.
    """
    if not suppress_joint_participants:
        return dict(steps)
    ego_step = steps.get(int(requesting_vehicle_id))
    return {int(requesting_vehicle_id): ego_step} if ego_step is not None else {}


def diagnostic_step_sets(
    steps: Mapping[int, dict[str, Any]],
    requesting_vehicle_id: int,
    *,
    suppress_joint_participants: bool,
    suppress_joint_participant_actuation: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return execution and transaction-tracking steps for diagnostics."""
    if suppress_joint_participants and suppress_joint_participant_actuation:
        raise ValueError("diagnostic suppression modes are mutually exclusive")
    planned_steps = dict(steps)
    execution_steps = diagnostic_executable_steps(
        planned_steps,
        requesting_vehicle_id,
        suppress_joint_participants=(
            suppress_joint_participants or suppress_joint_participant_actuation
        ),
    )
    tracked_steps = execution_steps if suppress_joint_participants else planned_steps
    return execution_steps, tracked_steps


def protected_goal_vehicle_ids(
    commands: Iterable[Mapping[str, Any]],
    current_sim_s: float,
    horizon_s: float = GOAL_PRIORITY_HORIZON_S,
) -> list[int]:
    """Return goal owners whose pending maneuver falls in the plan horizon."""
    return sorted(
        {
            int(command["veh_id"])
            for command in commands
            if not bool(command.get("completed"))
            and float(command["activation_sim_s"])
            <= current_sim_s + horizon_s
        }
    )


def transaction_executor_failure(
    steps: Mapping[int, Mapping[str, Any]],
    telemetry: Mapping[str, Mapping[str, Any]],
    started_sim_s: float,
) -> Optional[dict[str, Any]]:
    """Return a fresh physical executor failure for a tracked lane step."""
    for veh_id, step in steps.items():
        action = str(step.get("action") or "hold").lower()
        if action not in {"lane_left", "lane_right"}:
            continue
        row = telemetry.get(str(veh_id)) or {}
        lane_change = row.get("lane_change") or {}
        terminal_sim_s = lane_change.get("last_terminal_sim_s")
        if (
            lane_change.get("last_terminal_state") == "ABORT"
            and terminal_sim_s is not None
            and float(terminal_sim_s) >= float(started_sim_s) - 1e-6
        ):
            return {
                "veh_id": int(veh_id),
                "reason": str(
                    lane_change.get("last_terminal_reason")
                    or "lane_change_abort"
                ),
                "terminal_sim_s": float(terminal_sim_s),
                "lateral_progress_m": float(
                    lane_change.get("lateral_progress_m") or 0.0
                ),
            }
    return None


def build_commands(
    block: Mapping[str, Any],
    vehicle_ids: list[int],
    ready: Mapping[str, Mapping[str, Any]],
    base_sim_s: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    layout = layout_for_block(block)
    commands = []
    for ordinal, planned in enumerate(layout.commands, start=1):
        veh_id = int(vehicle_ids[planned.slot])
        initial_lane = int(ready[str(veh_id)]["lane_id"])
        commands.append(
            {
                "command_id": f"{block['block_id']}-c{ordinal:02d}-v{veh_id}",
                "slot": int(planned.slot),
                "veh_id": veh_id,
                "direction": str(planned.direction),
                "initial_lane_id": initial_lane,
                "target_lane_id": expected_target_lane(initial_lane, planned.direction),
                "activation_sim_s": base_sim_s + float(planned.issue_offset_sim_s),
                "initial_request_sequence": int(
                    (ready[str(veh_id)].get("lane_change") or {}).get("request_sequence") or 0
                ),
                "attempts": 0,
                "review_attempts": 0,
                "last_attempt_sim_s": None,
                "next_retry_sim_s": None,
                "accepted_sequences": [],
                "completed": False,
                "terminal_done": False,
                "tracker": PhysicalCompletionTracker(
                    expected_target_lane(initial_lane, planned.direction)
                ),
            }
        )
    by_vehicle = {int(command["veh_id"]): command for command in commands}
    return commands, by_vehicle


def serializable_command(command: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in command.items()
        if key != "tracker"
    }


def observe_commands(
    commands: Iterable[dict[str, Any]],
    telemetry: Mapping[str, Mapping[str, Any]],
    now_wall_s: float,
) -> None:
    for command in commands:
        row = telemetry[str(command["veh_id"])]
        lane_change = row.get("lane_change") or {}
        sequence = int(lane_change.get("request_sequence") or 0)
        if sequence > int(command["initial_request_sequence"]) and sequence not in command["accepted_sequences"]:
            command["accepted_sequences"].append(sequence)
        if (
            not command["completed"]
            and command["accepted_sequences"]
            and command["tracker"].observe(
                lane_id=int(row["lane_id"]),
                distance_to_center_m=float(row.get("distance_to_center") or 0.0),
                sim_time_s=float(row["sim_time_s"]),
                wall_time_s=now_wall_s,
            )
        ):
            command["completed"] = True
        terminal_state = lane_change.get("last_terminal_state")
        terminal_sim = lane_change.get("last_terminal_sim_s")
        if (
            terminal_state == "DONE"
            and terminal_sim is not None
            and command["accepted_sequences"]
            and int(row["lane_id"]) == int(command["target_lane_id"])
            and float(row.get("distance_to_center") or 0.0)
            <= CENTER_THRESHOLD_M
        ):
            command["terminal_done"] = True


def due_commands(
    commands: Iterable[dict[str, Any]],
    current_sim_s: float,
    telemetry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    return [
        command
        for command in commands
        if not command["completed"]
        and current_sim_s >= float(command["activation_sim_s"])
        and not (
            telemetry is not None
            and int(telemetry[str(command["veh_id"])]["lane_id"])
            == int(command["target_lane_id"])
        )
    ]


def can_retry(command: Mapping[str, Any], current_sim_s: float, retry_s: float) -> bool:
    next_retry = command.get("next_retry_sim_s")
    if next_retry is not None and current_sim_s < float(next_retry):
        return False
    previous = command.get("last_attempt_sim_s")
    return previous is None or current_sim_s - float(previous) >= retry_s


def commands_at_retry_limit(
    commands: Iterable[Mapping[str, Any]],
    active_transactions: Mapping[str, Mapping[str, Any]],
    max_attempts: int,
) -> list[Mapping[str, Any]]:
    """Return unfinished commands that exhausted retries and are not executing."""
    active_command_ids = {
        str(transaction["command_id"])
        for transaction in active_transactions.values()
    }
    return [
        command
        for command in commands
        if not bool(command.get("completed"))
        and int(command.get("attempts") or 0) >= max_attempts
        and str(command.get("command_id")) not in active_command_ids
    ]


def lane_controller_idle(row: Mapping[str, Any]) -> bool:
    state = (row.get("lane_change") or {}).get("state")
    return state in {None, "IDLE", "DONE", "ABORT"}


def build_mec_payload(
    command: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    protected_vehicle_ids: Iterable[int] = (),
) -> dict:
    veh_id = int(command["veh_id"])
    threats = target_corridor_threats(command, telemetry)
    request = (
        {
            "to": [str(threats[0])],
            "ask": "yield",
            "reason": "create a validated merge gap",
        }
        if threats
        else {"to": [], "ask": "none"}
    )
    action = "lane_left" if command["direction"] == "left" else "lane_right"
    transaction_id = (
        f"carla-{command['command_id']}-r"
        f"{int(command.get('review_attempts') or 0) + 1}"
    )
    now = time.time()
    return {
        "veh_id": veh_id,
        "transaction_id": transaction_id,
        "created_at_s": now,
        "expires_at_s": now + 5.0,
        "goal": f"reach lane {command['target_lane_id']}",
        "intent": {
            "ego_veh_id": veh_id,
            "ego_action": action,
            "reason": "shared scripted CARLA maneuver request",
            "confidence": 1.0,
            "target_lane_id": command["target_lane_id"],
            "request": request,
        },
        "request": request,
        "plan": {
            "summary": "shared scripted CARLA maneuver request",
            "horizon_s": 8.0,
            "steps": [
                {
                    "id": f"{transaction_id}-step-1",
                    "action": action,
                    "target_lane_id": command["target_lane_id"],
                    "target_speed_kmh": FLOW_SPEED_KMH,
                    "duration_s": 3.0,
                }
            ],
        },
        "context": {
            "goal": f"reach lane {command['target_lane_id']}",
            "intent_ttl_s": 5.0,
            "protected_goal_vehicle_ids": sorted(
                {int(veh_id) for veh_id in protected_vehicle_ids}
            ),
        },
    }


def initial_state_signature(
    telemetry: Mapping[str, Mapping[str, Any]], vehicle_ids: Iterable[int]
) -> dict[str, Any]:
    return {
        str(slot): {
            "road_id": telemetry[str(veh_id)].get("road_id"),
            "lane_id": telemetry[str(veh_id)].get("lane_id"),
            "s_m": round(float(telemetry[str(veh_id)].get("s_m") or 0.0), 3),
            "speed_kmh": round(float(telemetry[str(veh_id)].get("speed_kmh") or 0.0), 3),
        }
        for slot, veh_id in enumerate(vehicle_ids)
    }


def run_method_episode(
    client: httpx.Client,
    api_base: str,
    block: Mapping[str, Any],
    method: str,
    output_dir: Path,
    actor: Optional[NumpyActorPolicy],
    budget: RunnerBudget,
    *,
    archive: bool,
    enable_cameras: bool = False,
    persist_frames: bool = False,
    suppress_mind_joint_participants: bool = False,
    suppress_mind_joint_participant_actuation: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    layout = layout_for_block(block)
    mind_method = method in MIND_RANKER_METHODS
    client.post(f"{api_base}/reset", json={}).raise_for_status()
    configured = client.post(
        f"{api_base}/config",
        json={
            "num_cars": int(block["fleet_size"]),
            "spawn_indices": list(layout.spawn_indices),
            "spawn_longitudinal_offsets_m": list(
                block.get("spawn_longitudinal_offsets_m")
                or [0.0] * int(block["fleet_size"])
            ),
            "initial_speeds": [FLOW_SPEED_KMH] * int(block["fleet_size"]),
            "coordination_mode": "MIND_CAVS" if mind_method else "IA",
            "scenario_id": f"CARLA_PAIRED_{block['block_id']}_{method}",
            "seed": int(block["seed"]),
            "persist_frames": persist_frames,
            "enable_cameras": enable_cameras,
            "tm_auto_lane_change": False,
            "tm_route": list(block.get("tm_route") or []),
            "start_paused": True,
        },
    )
    configured.raise_for_status()
    vehicle_ids = [int(value) for value in configured.json()["veh_ids"]]
    client.post(f"{api_base}/tm/unsafe", json={"unsafe": True}).raise_for_status()
    warmup = client.post(
        f"{api_base}/simulation/warmup",
        json={
            "vehicle_ids": vehicle_ids,
            "minimum_speed_kmh": 30.0,
            "require_non_junction": True,
            "max_ticks": min(
                2000,
                max(1, int(math.ceil(budget.ready_timeout_wall_s / 0.05))),
            ),
        },
    )
    warmup.raise_for_status()
    ready = warmup.json()["vehicles"]
    base_sim_s = max(float(ready[str(veh_id)]["sim_time_s"]) for veh_id in vehicle_ids)
    commands, commands_by_vehicle = build_commands(
        block, vehicle_ids, ready, base_sim_s
    )
    initial_lane_domains = {
        int(veh_id): tuple(
            int(value)
            for value in ready[str(veh_id)].get("driving_lane_ids")
            or [ready[str(veh_id)]["lane_id"]]
        )
        for veh_id in vehicle_ids
    }
    conflict_evidence = initial_conflict_evidence(commands, ready)
    if block.get("require_initial_conflict") and not conflict_evidence:
        raise RuntimeError(
            f"{block['block_id']}: expected a conflict-active initial state"
        )
    deadline_sim_s = base_sim_s + budget.trial_timeout_sim_s
    trajectory_path = output_dir / "trajectory.jsonl"
    events_path = output_dir / "events.jsonl"
    initial_signature = initial_state_signature(ready, vehicle_ids)
    started_wall_s = time.time()
    deadline_wall_s = started_wall_s + budget.trial_timeout_wall_s

    decisions: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    active_fcfs: list[dict[str, Any]] = []
    fcfs_arrival: dict[str, int] = {}
    active_transactions: dict[str, dict[str, Any]] = {}
    priority_aware_actor = bool(
        actor is not None
        and int(actor.weights[0].shape[1]) == PRIORITY_OBSERVATION_DIM
    )
    mappo_validator = DeterministicPlanValidator(
        ValidationConfig(minimum_clearance_m=ALIGNED_MINIMUM_CLEARANCE_M)
        if priority_aware_actor
        else None
    )
    last_policy_sim_s = -float("inf")
    mappo_policy_steps = 0
    mappo_validator_interventions = 0
    last_frame: Optional[int] = None
    maximum_collision_count = 0
    minimum_shared_corridor_distance_m: Optional[float] = None
    gap_violation_samples = 0
    last_telemetry = ready
    current_sim_s = base_sim_s
    status = "timeout"
    corridor_departure_events: list[dict[str, Any]] = []
    first_sample = True

    while time.time() < deadline_wall_s:
        if first_sample:
            telemetry = ready
            first_sample = False
        else:
            stepped = client.post(
                f"{api_base}/simulation/step",
                json={
                    "vehicle_ids": vehicle_ids,
                    "ticks": budget.simulation_step_ticks,
                },
            )
            stepped.raise_for_status()
            telemetry = stepped.json()["vehicles"]
        if not all(str(veh_id) in telemetry for veh_id in vehicle_ids):
            time.sleep(budget.polling_s)
            continue
        last_telemetry = {str(veh_id): telemetry[str(veh_id)] for veh_id in vehicle_ids}
        current_frame = max(int(row.get("sim_frame") or 0) for row in last_telemetry.values())
        if current_frame == last_frame:
            time.sleep(budget.polling_s)
            continue
        last_frame = current_frame
        current_sim_s = max(float(row["sim_time_s"]) for row in last_telemetry.values())
        now_wall_s = time.time()
        observe_commands(commands, last_telemetry, now_wall_s)
        _, shared_distance = pair_distance_metrics(last_telemetry)
        if shared_distance is not None:
            minimum_shared_corridor_distance_m = (
                shared_distance
                if minimum_shared_corridor_distance_m is None
                else min(minimum_shared_corridor_distance_m, shared_distance)
            )
            gap_violation_samples += int(shared_distance < 5.0)
        maximum_collision_count = max(
            maximum_collision_count,
            sum(int(row.get("collision_count") or 0) for row in last_telemetry.values()),
        )
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
        if maximum_collision_count:
            status = "collision"
            append_jsonl(events_path, {"event": "collision", "sim_time_s": current_sim_s})
            break
        departures = corridor_departures(last_telemetry, initial_lane_domains)
        if departures:
            status = "corridor_departure"
            corridor_departure_events.extend(departures)
            append_jsonl(
                events_path,
                {
                    "event": "corridor_departure",
                    "sim_time_s": current_sim_s,
                    "departures": departures,
                },
            )
            break
        if current_sim_s >= deadline_sim_s:
            status = "timeout"
            break

        pending = due_commands(commands, current_sim_s, last_telemetry)
        if method == "IA":
            for command in pending:
                row = last_telemetry[str(command["veh_id"])]
                if lane_controller_idle(row) and can_retry(command, current_sim_s, budget.request_retry_sim_s):
                    post_lane_command(client, api_base, command)
                    command["last_attempt_sim_s"] = current_sim_s
                    append_jsonl(events_path, {"event": "ia_command", "sim_time_s": current_sim_s, **serializable_command(command)})

        elif method in {"FCFS_QUEUE", "FCFS_GAP"}:
            active_fcfs = [command for command in active_fcfs if not command["completed"]]
            for command in pending:
                fcfs_arrival.setdefault(command["command_id"], len(fcfs_arrival))
            ordered = sorted(
                pending,
                key=lambda command: (
                    fcfs_arrival[command["command_id"]],
                    list(block["priority_order"]).index(int(command["slot"])),
                ),
            )
            for command in ordered:
                if command in active_fcfs:
                    continue
                blocker = next(
                    (
                        active
                        for active in active_fcfs
                        if command_conflict(command, active, last_telemetry)
                    ),
                    None,
                )
                if blocker is None:
                    active_fcfs.append(command)
                    append_jsonl(events_path, {"event": "fcfs_ack", "sim_time_s": current_sim_s, "command_id": command["command_id"]})
                else:
                    append_jsonl(events_path, {"event": "fcfs_nack", "sim_time_s": current_sim_s, "command_id": command["command_id"], "conflicting_command_id": blocker["command_id"]})
            for command in active_fcfs:
                if command["completed"]:
                    continue
                row = last_telemetry[str(command["veh_id"])]
                if not lane_controller_idle(row) or not can_retry(command, current_sim_s, budget.request_retry_sim_s):
                    continue
                threats = target_corridor_threats(command, last_telemetry) if method == "FCFS_GAP" else []
                if threats:
                    other = last_telemetry[str(threats[0])]
                    delta = longitudinal_delta_m(row, other)
                    target_speed = 30.0 if delta >= 0.0 else 55.0
                    post_speed(client, api_base, int(command["veh_id"]), target_speed)
                    command["last_attempt_sim_s"] = current_sim_s
                    append_jsonl(events_path, {"event": "fcfs_gap_action", "sim_time_s": current_sim_s, "command_id": command["command_id"], "action": "yield" if target_speed < FLOW_SPEED_KMH else "accelerate", "threat_vehicle_id": threats[0]})
                else:
                    post_lane_command(client, api_base, command)
                    command["last_attempt_sim_s"] = current_sim_s
                    append_jsonl(events_path, {"event": "fcfs_command", "sim_time_s": current_sim_s, **serializable_command(command)})

        elif method == "MAPPO_ADAPTED":
            if actor is None:
                raise RuntimeError("MAPPO_ADAPTED requires an actor")
            if current_sim_s - last_policy_sim_s >= budget.action_period_sim_s:
                actor_input_dim = int(actor.weights[0].shape[1])
                if actor_input_dim not in {
                    OBSERVATION_DIM,
                    PRIORITY_OBSERVATION_DIM,
                }:
                    raise ValueError(
                        f"unsupported MAPPO actor input dimension: {actor_input_dim}"
                    )
                priority_aware = actor_input_dim == PRIORITY_OBSERVATION_DIM
                observations = carla_actor_observations(
                    last_telemetry,
                    vehicle_ids,
                    commands_by_vehicle,
                    current_sim_s=current_sim_s,
                    priority_order=(
                        block["priority_order"] if priority_aware else None
                    ),
                )
                logits = actor.logits(observations)
                requested_actions = np.argmax(logits, axis=-1).astype(int)
                actions, action_masks = carla_validator_masked_argmax(
                    logits,
                    last_telemetry,
                    vehicle_ids,
                    commands_by_vehicle,
                    block["priority_order"],
                    mappo_validator,
                    current_sim_s=current_sim_s,
                    allow_cooperative_support=priority_aware,
                    goal_directed_lane_actions=priority_aware,
                )
                last_policy_sim_s = current_sim_s
                mappo_policy_steps += 1
                mappo_validator_interventions += int(
                    np.count_nonzero(requested_actions != actions)
                )
                append_jsonl(events_path, {"event": "mappo_actions", "sim_time_s": current_sim_s, "requested_actions": requested_actions.tolist(), "actions": actions.tolist(), "intervention_mask": (requested_actions != actions).tolist(), "action_masks": action_masks.tolist(), "observation_sha256": sha256_json(observations.round(7).tolist())})
                for slot, action in enumerate(actions):
                    veh_id = vehicle_ids[slot]
                    command = commands_by_vehicle.get(veh_id)
                    row = last_telemetry[str(veh_id)]
                    active_command = bool(
                        command
                        and current_sim_s >= float(command["activation_sim_s"])
                        and not command["completed"]
                    )
                    if not priority_aware and not active_command:
                        continue
                    post_speed(
                        client,
                        api_base,
                        veh_id,
                        mappo_target_speed_kmh(
                            int(action), float(row.get("speed_kmh") or 0.0)
                        ),
                    )
                    if (
                        active_command
                        and action in {LANE_LEFT, LANE_RIGHT}
                        and lane_controller_idle(row)
                        and can_retry(
                            command, current_sim_s, budget.request_retry_sim_s
                        )
                    ):
                        direction = "left" if action == LANE_LEFT else "right"
                        response = client.post(f"{api_base}/command", json={"cmd": "lane", "veh_id": veh_id, "dir": direction})
                        response.raise_for_status()
                        command["attempts"] += 1
                        command["last_attempt_sim_s"] = current_sim_s

        elif mind_method:
            for transaction_id, transaction in list(active_transactions.items()):
                elapsed_transaction_sim_s = current_sim_s - float(
                    transaction["started_sim_s"]
                )
                executor_failure = transaction_executor_failure(
                    transaction["steps"],
                    last_telemetry,
                    float(transaction["started_sim_s"]),
                )
                if executor_failure is not None:
                    post_outcome(
                        client,
                        api_base,
                        transaction_id,
                        "failed",
                        {
                            **executor_failure,
                            "elapsed_sim_s": elapsed_transaction_sim_s,
                        },
                    )
                    append_jsonl(
                        events_path,
                        {
                            "event": "transaction_failed",
                            "sim_time_s": current_sim_s,
                            "transaction_id": transaction_id,
                            **executor_failure,
                        },
                    )
                    failed_command = next(
                        (
                            command
                            for command in commands
                            if command["command_id"]
                            == transaction["command_id"]
                        ),
                        None,
                    )
                    if failed_command is not None:
                        failed_command["last_attempt_sim_s"] = current_sim_s
                        failed_command["next_retry_sim_s"] = (
                            current_sim_s + budget.no_progress_retry_sim_s
                        )
                    del active_transactions[transaction_id]
                elif steps_complete(
                    transaction["steps"],
                    last_telemetry,
                    elapsed_transaction_sim_s,
                ):
                    post_outcome(client, api_base, transaction_id, "completed", {"elapsed_sim_s": elapsed_transaction_sim_s})
                    append_jsonl(events_path, {"event": "transaction_completed", "sim_time_s": current_sim_s, "transaction_id": transaction_id})
                    del active_transactions[transaction_id]
                elif elapsed_transaction_sim_s > float(transaction["timeout_sim_s"]):
                    post_outcome(client, api_base, transaction_id, "failed", {"reason": "executor_timeout", "elapsed_sim_s": elapsed_transaction_sim_s})
                    append_jsonl(events_path, {"event": "transaction_failed", "sim_time_s": current_sim_s, "transaction_id": transaction_id})
                    del active_transactions[transaction_id]

            exhausted = commands_at_retry_limit(
                pending,
                active_transactions,
                budget.max_command_attempts,
            )
            if exhausted:
                status = "retry_limit"
                append_jsonl(
                    events_path,
                    {
                        "event": "retry_limit",
                        "sim_time_s": current_sim_s,
                        "max_command_attempts": budget.max_command_attempts,
                        "commands": [
                            {
                                "command_id": command["command_id"],
                                "attempts": command["attempts"],
                                "review_attempts": command.get(
                                    "review_attempts", 0
                                ),
                            }
                            for command in exhausted
                        ],
                    },
                )
                break

            priority_rank = {int(slot): rank for rank, slot in enumerate(block["priority_order"])}
            candidates = sorted(pending, key=lambda command: priority_rank[int(command["slot"])])
            active_vehicle_ids = {
                int(veh_id)
                for transaction in active_transactions.values()
                for veh_id in transaction["steps"]
            }
            for command in candidates:
                if int(command["veh_id"]) in active_vehicle_ids:
                    continue
                if (
                    lane_controller_idle(last_telemetry[str(command["veh_id"])])
                    and can_retry(command, current_sim_s, budget.request_retry_sim_s)
                ):
                    protected_ids = protected_goal_vehicle_ids(
                        commands,
                        current_sim_s,
                    )
                    payload = build_mec_payload(
                        command,
                        last_telemetry,
                        protected_vehicle_ids=protected_ids,
                    )
                    response = client.post(f"{api_base}/mec/review", json=payload)
                    response.raise_for_status()
                    decision = response.json()
                    if decision.get("decision_id") != payload["transaction_id"]:
                        raise RuntimeError("MIND-CAV changed transaction identity")
                    command["review_attempts"] += 1
                    command["last_attempt_sim_s"] = current_sim_s
                    command["next_retry_sim_s"] = None
                    decisions[str(decision.get("decision"))] += 1
                    reason_codes[str(decision.get("reason_code"))] += 1
                    plans = decision_plans(decision)
                    steps = (
                        first_steps(plans)
                        if decision.get("decision") in {"allow", "override"}
                        else {}
                    )
                    planned_steps = dict(steps)
                    execution_steps, tracked_steps = diagnostic_step_sets(
                        planned_steps,
                        int(command["veh_id"]),
                        suppress_joint_participants=suppress_mind_joint_participants,
                        suppress_joint_participant_actuation=(
                            suppress_mind_joint_participant_actuation
                        ),
                    )
                    executed = {
                        veh_id: execute_protocol_step(client, api_base, veh_id, step, last_telemetry)
                        for veh_id, step in execution_steps.items()
                    }
                    ego_step = execution_steps.get(int(command["veh_id"]))
                    if ego_step and str(ego_step.get("action") or "").lower() in {
                        "lane_left",
                        "lane_right",
                    }:
                        command["attempts"] += 1
                    suppressed_vehicle_ids = sorted(
                        set(planned_steps).difference(execution_steps)
                    )
                    append_jsonl(events_path, {"event": "mec_decision", "sim_time_s": current_sim_s, "command_id": command["command_id"], "decision": decision, "executed": executed, "protected_goal_vehicle_ids": protected_ids, "diagnostic_suppressed_vehicle_ids": suppressed_vehicle_ids})
                    if steps:
                        post_outcome(client, api_base, payload["transaction_id"], "executing", {"executed": executed})
                        active_transactions[payload["transaction_id"]] = {
                            "transaction_id": payload["transaction_id"],
                            "command_id": command["command_id"],
                            "steps": tracked_steps,
                            "started_sim_s": current_sim_s,
                            "timeout_sim_s": max(10.0, max(float(step.get("duration_s") or 0.0) for step in tracked_steps.values()) + 8.0),
                        }
                        active_vehicle_ids.update(int(veh_id) for veh_id in tracked_steps)

        else:
            raise ValueError(f"unsupported method: {method}")

        if commands and all(command["completed"] and command["terminal_done"] for command in commands) and not active_transactions:
            status = "completed"
            break
        time.sleep(budget.polling_s)

    for transaction_id, transaction in list(active_transactions.items()):
        elapsed_transaction_sim_s = current_sim_s - float(
            transaction["started_sim_s"]
        )
        post_outcome(
            client,
            api_base,
            transaction_id,
            "failed",
            {
                "reason": f"episode_{status}",
                "elapsed_sim_s": elapsed_transaction_sim_s,
            },
        )
        append_jsonl(
            events_path,
            {
                "event": "transaction_failed",
                "sim_time_s": current_sim_s,
                "transaction_id": transaction_id,
                "reason": f"episode_{status}",
            },
        )
        del active_transactions[transaction_id]

    ended_wall_s = time.time()
    completed_count = sum(bool(command["completed"]) for command in commands)
    terminal_done_count = sum(bool(command["terminal_done"]) for command in commands)
    safe_task_success = bool(
        status == "completed"
        and maximum_collision_count == 0
        and gap_violation_samples == 0
        and completed_count == len(commands)
        and terminal_done_count == len(commands)
    )
    archive_sha256 = None
    if archive:
        response = client.get(f"{api_base}/archive", timeout=120.0)
        response.raise_for_status()
        archive_path = output_dir / "server_archive.zip"
        archive_path.write_bytes(response.content)
        archive_sha256 = hashlib.sha256(response.content).hexdigest()
    result = {
        "schema_version": "1.0",
        "claim_status": str(
            block.get("claim_status")
            or ("exploratory_pilot" if int(block["seed"]) < 7_100_000 else "scheduled")
        ),
        "block_id": block["block_id"],
        "scenario_family": block["scenario_family"],
        "executor_pattern": block["executor_pattern"],
        "fleet_size": int(block["fleet_size"]),
        "seed": int(block["seed"]),
        "method": method,
        "method_order": list(block["method_order"]),
        "priority_order": list(block["priority_order"]),
        "spawn_indices": list(layout.spawn_indices),
        "vehicle_ids": vehicle_ids,
        "initial_state_signature": initial_signature,
        "initial_state_observed_sha256": sha256_json(initial_signature),
        "initial_conflict_required": bool(block.get("require_initial_conflict")),
        "initial_conflict_count": len(conflict_evidence),
        "initial_conflict_evidence": conflict_evidence,
        "paired_design_sha256": sha256_json(
            {
                "seed": int(block["seed"]),
                "spawn_indices": list(layout.spawn_indices),
                "initial_speeds_kmh": [FLOW_SPEED_KMH]
                * int(block["fleet_size"]),
                "commands": [
                    {
                        "slot": command["slot"],
                        "direction": command["direction"],
                        "target_lane_id": command["target_lane_id"],
                        "activation_offset_sim_s": (
                            float(command["activation_sim_s"]) - base_sim_s
                        ),
                    }
                    for command in commands
                ],
                "tm_route": list(block.get("tm_route") or []),
            }
        ),
        "status": status,
        "safe_task_success": safe_task_success,
        "planned_command_count": len(commands),
        "completed_command_count": completed_count,
        "terminal_done_count": terminal_done_count,
        "collision_count": maximum_collision_count,
        "corridor_departure_count": len(corridor_departure_events),
        "corridor_departures": corridor_departure_events,
        "gap_violation_samples": gap_violation_samples,
        "minimum_shared_corridor_distance_m": minimum_shared_corridor_distance_m,
        "elapsed_wall_s": ended_wall_s - started_wall_s,
        "elapsed_sim_s": max(float(row.get("sim_time_s") or base_sim_s) for row in last_telemetry.values()) - base_sim_s,
        "decision_counts": dict(decisions),
        "reason_code_counts": dict(reason_codes),
        "mappo_action_validator": method == "MAPPO_ADAPTED",
        "mappo_policy_steps": mappo_policy_steps,
        "mappo_validator_interventions": mappo_validator_interventions,
        "commands": [serializable_command(command) for command in commands],
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "archive_sha256": archive_sha256,
        "persist_frames": persist_frames,
        "camera_sensors_enabled": enable_cameras,
        "traffic_manager_auto_lane_change": False,
        "traffic_manager_collision_avoidance": False,
        "traffic_manager_route": list(block.get("tm_route") or []),
        "simulation_execution_mode": "fixed_step",
        "simulation_step_ticks": budget.simulation_step_ticks,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def pilot_schedule() -> dict:
    schedule = build_schedule(
        PlanningConfig(
            repetitions_per_cell=1,
            test_seed_base=PILOT_SEED_BASE,
            method_order_seed=2_026_082_703,
        )
    )
    schedule["claim_status"] = "exploratory_pilot"
    schedule["label"] = "carla_paired_coordination_integration_pilot"
    schedule["max_command_attempts"] = RunnerBudget().max_command_attempts
    return schedule


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "block_id",
        "scenario_family",
        "fleet_size",
        "seed",
        "method",
        "status",
        "safe_task_success",
        "planned_command_count",
        "completed_command_count",
        "terminal_done_count",
        "collision_count",
        "gap_violation_samples",
        "minimum_shared_corridor_distance_m",
        "initial_conflict_count",
        "elapsed_wall_s",
        "elapsed_sim_s",
        "initial_state_observed_sha256",
        "paired_design_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--confirmatory-lock", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/carla_paired_coordination_pilot_20260827"),
    )
    parser.add_argument(
        "--actor",
        type=Path,
        default=Path("data/experiments/adapted_mappo_full_20260826/seed_2026082601/actor.npz"),
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    parser.add_argument("--fleet-sizes", default=",".join(str(value) for value in FLEET_SIZES))
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--trial-timeout-wall-s", type=float, default=210.0)
    parser.add_argument("--trial-timeout-sim-s", type=float, default=60.0)
    parser.add_argument("--ready-timeout-wall-s", type=float, default=90.0)
    parser.add_argument("--max-command-attempts", type=int, default=10)
    parser.add_argument(
        "--diagnostic-suppress-mind-joint-participants",
        action="store_true",
        help=(
            "diagnostic only: execute only the requesting vehicle from each "
            "MIND-CAV joint plan"
        ),
    )
    parser.add_argument(
        "--diagnostic-suppress-mind-joint-participant-actuation",
        action="store_true",
        help=(
            "diagnostic only: retain MIND-CAV participant reservations but "
            "do not actuate their auxiliary actions"
        ),
    )
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument(
        "--enable-cameras",
        action="store_true",
        help="enable RGB sensors while keeping frame persistence disabled",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.trial_timeout_wall_s <= 0.0
        or args.trial_timeout_sim_s <= 0.0
        or args.ready_timeout_wall_s <= 0.0
        or args.max_command_attempts <= 0
    ):
        raise ValueError("runner limits must be positive")
    if (
        args.diagnostic_suppress_mind_joint_participants
        and args.diagnostic_suppress_mind_joint_participant_actuation
    ):
        raise ValueError("select at most one MIND-CAV diagnostic suppression mode")
    runner_budget = RunnerBudget(
        trial_timeout_wall_s=args.trial_timeout_wall_s,
        trial_timeout_sim_s=args.trial_timeout_sim_s,
        ready_timeout_wall_s=args.ready_timeout_wall_s,
        max_command_attempts=args.max_command_attempts,
    )

    if args.pilot == bool(args.schedule):
        raise ValueError("select exactly one of --pilot or --schedule")
    schedule = pilot_schedule() if args.pilot else json.loads(args.schedule.read_text(encoding="utf-8"))
    validate_schedule(schedule)
    expected_claim_status = "exploratory_pilot" if args.pilot else "frozen_confirmatory"
    if schedule.get("claim_status") != expected_claim_status:
        raise ValueError(
            f"expected {expected_claim_status} schedule, got "
            f"{schedule.get('claim_status')}"
        )
    if bool(args.enable_cameras) != bool(schedule["camera_sensors_enabled"]):
        raise ValueError("camera sensor mode differs from the frozen schedule")
    budget_checks = {
        "trial_timeout_wall_s": args.trial_timeout_wall_s,
        "trial_timeout_sim_s": args.trial_timeout_sim_s,
        "ready_timeout_wall_s": args.ready_timeout_wall_s,
        "max_command_attempts": args.max_command_attempts,
    }
    for key, observed in budget_checks.items():
        if abs(float(schedule[key]) - float(observed)) > 1e-9:
            raise ValueError(f"{key} differs from the frozen schedule")
    lock_arguments = (
        args.confirmatory_lock,
        args.registration,
        args.environment_lock,
    )
    if args.pilot:
        if any(value is not None for value in lock_arguments):
            raise ValueError("exploratory pilot cannot use a confirmatory lock")
        frozen_lock = None
    else:
        if any(value is None for value in lock_arguments):
            raise ValueError(
                "confirmatory execution requires --confirmatory-lock, "
                "--registration, and --environment-lock"
            )
        frozen_lock = validate_frozen_lock(
            schedule=args.schedule,
            registration=args.registration,
            environment_lock=args.environment_lock,
            lock_path=args.confirmatory_lock,
        )
        expected_actor = (REPO_ROOT / schedule["mappo_actor"]).resolve()
        if args.actor.resolve() != expected_actor:
            raise ValueError("MAPPO actor path differs from the frozen schedule")
    selected_methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    selected_families = {value.strip() for value in args.families.split(",") if value.strip()}
    selected_fleets = {int(value) for value in args.fleet_sizes.split(",") if value.strip()}
    if not set(selected_methods).issubset(METHODS):
        raise ValueError("unknown selected method")
    if not args.pilot:
        protocol_deviations = {
            "methods": selected_methods != list(schedule["methods"]),
            "families": selected_families != set(schedule["scenario_families"]),
            "fleet_sizes": selected_fleets != set(schedule["fleet_sizes"]),
            "max_blocks": args.max_blocks is not None,
            "diagnostic_suppress_participants": (
                args.diagnostic_suppress_mind_joint_participants
            ),
            "diagnostic_suppress_actuation": (
                args.diagnostic_suppress_mind_joint_participant_actuation
            ),
            "no_archive": args.no_archive,
            "overwrite": args.overwrite,
        }
        changed = sorted(key for key, value in protocol_deviations.items() if value)
        if changed:
            raise ValueError(f"confirmatory protocol deviation: {changed}")
    blocks = [
        block
        for block in schedule["blocks"]
        if block["scenario_family"] in selected_families
        and int(block["fleet_size"]) in selected_fleets
    ]
    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    (args.output / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True), encoding="utf-8"
    )

    provenance = build_provenance(args.actor)
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    actor = NumpyActorPolicy.load(str(args.actor))
    rows: list[dict[str, Any]] = []
    mec_preflight = None
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{args.api_base.rstrip('/')}/health")
        health.raise_for_status()
        if "MIND_CAV" in selected_methods:
            mec_preflight = client.get(
                f"{args.api_base.rstrip('/')}/mec/v2/status"
            ).json()
            expected_ranker = (REPO_ROOT / schedule["mind_ranker"]).resolve()
            observed_ranker = mec_preflight.get("ranker_model_path")
            if (
                mec_preflight.get("proposer") != "constrained-learned-ranker"
                or mec_preflight.get("ranker_model_loaded") is not True
                or mec_preflight.get("liveness_preparation") is not True
                or not observed_ranker
                or Path(observed_ranker).resolve() != expected_ranker
            ):
                raise ValueError(f"MIND-CAV API preflight mismatch: {mec_preflight}")
        for block in blocks:
            method_order = [
                method for method in block["method_order"] if method in selected_methods
            ]
            for method in method_order:
                run_dir = args.output / block["block_id"] / method
                print(f"[RUN] {block['block_id']} {method}", flush=True)
                row = run_method_episode(
                    client,
                    args.api_base.rstrip("/"),
                    block,
                    method,
                    run_dir,
                    actor if method == "MAPPO_ADAPTED" else None,
                    runner_budget,
                    archive=not args.no_archive,
                    enable_cameras=args.enable_cameras,
                    suppress_mind_joint_participants=(
                        args.diagnostic_suppress_mind_joint_participants
                    ),
                    suppress_mind_joint_participant_actuation=(
                        args.diagnostic_suppress_mind_joint_participant_actuation
                    ),
                )
                rows.append(row)
                write_summary(args.output / "summary.csv", rows)
                print(
                    f"[DONE] status={row['status']} safe={row['safe_task_success']} "
                    f"completed={row['completed_command_count']}/{row['planned_command_count']} "
                    f"collision={row['collision_count']}",
                    flush=True,
                )
    manifest = {
        "claim_status": schedule["claim_status"],
        "schedule_sha256": sha256_json(schedule),
        "actor": str(args.actor.resolve()),
        "actor_sha256": hashlib.sha256(args.actor.read_bytes()).hexdigest(),
        "provenance_sha256": sha256_file(provenance_path),
        "confirmatory_lock": (
            None
            if args.confirmatory_lock is None
            else {
                "path": str(args.confirmatory_lock.resolve()),
                "sha256": sha256_file(args.confirmatory_lock),
                "registration_freeze_commit": frozen_lock["registration"][
                    "freeze_commit"
                ],
            }
        ),
        "mec_preflight": mec_preflight,
        "rows": len(rows),
        "blocks": len(blocks),
        "methods": selected_methods,
        "persist_frames": False,
        "camera_sensors_enabled": args.enable_cameras,
        "diagnostic_suppress_mind_joint_participants": (
            args.diagnostic_suppress_mind_joint_participants
        ),
        "diagnostic_suppress_mind_joint_participant_actuation": (
            args.diagnostic_suppress_mind_joint_participant_actuation
        ),
        "runner_budget": {
            "trial_timeout_wall_s": runner_budget.trial_timeout_wall_s,
            "trial_timeout_sim_s": runner_budget.trial_timeout_sim_s,
            "ready_timeout_wall_s": runner_budget.ready_timeout_wall_s,
            "polling_s": runner_budget.polling_s,
            "action_period_sim_s": runner_budget.action_period_sim_s,
            "request_retry_sim_s": runner_budget.request_retry_sim_s,
            "no_progress_retry_sim_s": runner_budget.no_progress_retry_sim_s,
            "max_command_attempts": runner_budget.max_command_attempts,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
