#!/usr/bin/env python3
"""Run exploratory CARLA transfers of the frozen guarded MIND-CAV stack.

This runner uses the frozen V5 vehicle-intent checkpoint, submits its typed
proposal plus the current CARLA top-down frame to the FastAPI MEC endpoint,
executes only ACK/PLAN outputs, and records execution outcomes.  It is a
mechanism/transfer check, not a confirmatory safety experiment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marl.multimodal_intent import lane_index
from marl.multimodal_intent_runtime import FrozenMultimodalIntentPolicy
from server.coordination.legacy_adapter import telemetry_to_states
from server.coordination.models import IntentProposal


@dataclass(frozen=True)
class TransferScenario:
    scenario_id: str
    description: str
    spawn_indices: tuple[int, int]
    target_lane_id: int
    initial_speeds_kmh: tuple[float, float] = (40.0, 40.0)
    timeout_s: float = 35.0


SCENARIOS: dict[str, TransferScenario] = {
    "clear_change": TransferScenario(
        scenario_id="clear_change",
        description="Ego changes away from a close vehicle in the other adjacent lane.",
        spawn_indices=(212, 218),
        target_lane_id=-1,
    ),
    "contested_merge": TransferScenario(
        scenario_id="contested_merge",
        description=(
            "Ego requests the neighboring occupied lane and may require a "
            "joint revision."
        ),
        spawn_indices=(212, 218),
        target_lane_id=-3,
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def proposal_to_review_payload(
    proposal: IntentProposal,
    top_frame: bytes,
) -> dict[str, Any]:
    """Convert a protocol-native proposal without losing transaction identity."""
    first_step = proposal.plan.steps[0]
    request = None
    if proposal.request is not None:
        request = {
            "to": proposal.request.to_vehicle_ids,
            "ask": proposal.request.requested_action.value,
            "reason": proposal.request.reason,
            "expires_at_s": proposal.request.expires_at_s,
        }
    steps = [
        {
            "id": step.step_id,
            "action": step.action.value,
            "target_lane_id": step.target_lane_id,
            "target_speed_kmh": step.target_speed_kmh,
            "duration_s": step.duration_s,
            "completion_condition": step.completion_condition,
            "description": step.description,
        }
        for step in proposal.plan.steps
    ]
    goal_text = proposal.goal.semantic or proposal.goal.kind.value
    return {
        "veh_id": proposal.ego_veh_id,
        "transaction_id": proposal.transaction_id,
        "created_at_s": proposal.created_at_s,
        "expires_at_s": proposal.expires_at_s,
        "observation_ts_s": proposal.observation_ts_s,
        "source": proposal.source,
        "goal": goal_text,
        "intent": {
            "ego_veh_id": proposal.ego_veh_id,
            "ego_action": first_step.action.value,
            "reason": proposal.plan.summary,
            "confidence": proposal.confidence,
            "target_lane_id": first_step.target_lane_id,
            "request": request or {"to": [], "ask": "none"},
        },
        "request": request,
        "plan": {
            "summary": proposal.plan.summary,
            "horizon_s": proposal.plan.horizon_s,
            "steps": steps,
        },
        "context": {
            "goal": goal_text,
            "goal_kind": proposal.goal.kind.value,
            "exit_id": proposal.goal.exit_id,
            "intent_ttl_s": max(
                0.1,
                float(proposal.expires_at_s or time.time())
                - float(proposal.created_at_s),
            ),
        },
        "top_frame_b64": base64.b64encode(top_frame).decode("ascii"),
    }


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
    timeout_s: float = 15.0,
) -> dict[str, dict]:
    expected = {str(value) for value in vehicle_ids}
    deadline = time.time() + timeout_s
    last: dict[str, dict] = {}
    while time.time() < deadline:
        last = latest_telemetry(client, api_base)
        if expected.issubset(last) and all(
            float(last[veh_id].get("speed_kmh") or 0.0) >= minimum_speed_kmh
            and last[veh_id].get("lane_id") is not None
            and last[veh_id].get("road_id") is not None
            and last[veh_id].get("s_m") is not None
            for veh_id in expected
        ):
            return last
        time.sleep(0.1)
    speeds = {
        veh_id: (last.get(veh_id) or {}).get("speed_kmh") for veh_id in expected
    }
    raise TimeoutError(f"CARLA telemetry did not become ready; speeds={speeds}")


def wait_for_top_frame(
    client: httpx.Client,
    api_base: str,
    veh_id: int,
    timeout_s: float = 8.0,
) -> bytes:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"{api_base}/frame/top/{veh_id}.jpg")
        if response.status_code == 200 and response.content:
            return response.content
        if response.status_code != 404:
            response.raise_for_status()
        time.sleep(0.05)
    raise TimeoutError(f"top frame unavailable for vehicle {veh_id}")


def execute_step(
    client: httpx.Client,
    api_base: str,
    veh_id: int,
    step: Mapping[str, Any],
) -> str:
    action = str(step.get("action") or "hold").lower()
    target_speed = step.get("target_speed_kmh")
    if action in {"lane_left", "lane_right"}:
        response = client.post(
            f"{api_base}/command",
            json={
                "cmd": "intent",
                "veh_id": veh_id,
                "intent": {
                    "ego_veh_id": veh_id,
                    "ego_action": action.replace("_", " "),
                    "target_lane_id": step.get("target_lane_id"),
                    "target_speed_kmh": target_speed,
                    "reason": "authorized_guarded_carla_transfer",
                    "confidence": 1.0,
                    "request": {"to": [], "ask": "none"},
                },
            },
        )
    elif action == "brake":
        response = client.post(
            f"{api_base}/command", json={"cmd": "brake", "veh_id": veh_id}
        )
    elif target_speed is not None:
        response = client.post(
            f"{api_base}/command",
            json={"cmd": "speed", "veh_id": veh_id, "kmh": float(target_speed)},
        )
    else:
        return action
    response.raise_for_status()
    return action


def decision_plans(decision: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if decision.get("decision") in {"allow", "override"}:
        joint = decision.get("joint_plan") or {}
        raw_plans = joint.get("plans") or {}
        return {int(veh_id): dict(plan) for veh_id, plan in raw_plans.items()}
    fallback = decision.get("fallback_plan")
    if fallback and decision.get("veh_id") is not None:
        return {int(decision["veh_id"]): dict(fallback)}
    return {}


def first_steps(plans: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for veh_id, plan in plans.items():
        steps = plan.get("steps") or []
        if steps:
            selected[int(veh_id)] = dict(steps[0])
    return selected


def steps_complete(
    steps: Mapping[int, Mapping[str, Any]],
    telemetry: Mapping[str, Mapping[str, Any]],
    elapsed_s: float,
) -> bool:
    for veh_id, step in steps.items():
        action = str(step.get("action") or "hold").lower()
        row = telemetry.get(str(veh_id)) or {}
        if action in {"lane_left", "lane_right"}:
            if row.get("lane_id") != step.get("target_lane_id"):
                return False
            lane_state = (row.get("lane_change") or {}).get("state")
            if lane_state not in {None, "IDLE", "DONE"}:
                return False
        elif elapsed_s < float(step.get("duration_s") or 0.0):
            return False
    return True


def longitudinal_gap_m(telemetry: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    rows = list(telemetry.values())
    if len(rows) != 2:
        return None
    a, b = rows
    if (
        a.get("road_id") == b.get("road_id")
        and a.get("s_m") is not None
        and b.get("s_m") is not None
    ):
        return abs(float(a["s_m"]) - float(b["s_m"]))
    try:
        return math.hypot(
            float(a["pose"]["x"]) - float(b["pose"]["x"]),
            float(a["pose"]["y"]) - float(b["pose"]["y"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def download_archive(client: httpx.Client, api_base: str, output_dir: Path) -> None:
    response = client.get(f"{api_base}/archive", timeout=120.0)
    response.raise_for_status()
    archive_path = output_dir / "session_archive.zip"
    archive_path.write_bytes(response.content)
    safe_extract(archive_path, output_dir / "server_archive")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def post_outcome(
    client: httpx.Client,
    api_base: str,
    transaction_id: str,
    outcome: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    response = client.post(
        f"{api_base}/mec/v2/outcome",
        json={
            "transaction_id": transaction_id,
            "outcome": outcome,
            "details": details or {},
        },
    )
    response.raise_for_status()


def run_scenario(
    client: httpx.Client,
    api_base: str,
    scenario: TransferScenario,
    output_dir: Path,
    checkpoint: Path,
    calibration: Path,
    *,
    seed: int,
    archive: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    client.post(f"{api_base}/reset", json={}).raise_for_status()
    configured = client.post(
        f"{api_base}/config",
        json={
            "num_cars": 2,
            "spawn_indices": list(scenario.spawn_indices),
            "initial_speeds": list(scenario.initial_speeds_kmh),
            "coordination_mode": "MIND_CAVS",
            "scenario_id": f"CARLA_GUARDED_{scenario.scenario_id.upper()}",
            "seed": seed,
            "lane_goals": {"A": scenario.target_lane_id, "B": None},
        },
    )
    configured.raise_for_status()
    vehicle_ids = [int(value) for value in configured.json()["veh_ids"]]
    ego_id = vehicle_ids[0]
    wait_for_ready_telemetry(client, api_base, vehicle_ids)

    status = client.get(f"{api_base}/mec/v2/status").json()
    if not status.get("enabled"):
        raise RuntimeError("MIND-CAV v2 is disabled")
    if not status.get("ranker_model_loaded"):
        raise RuntimeError(f"candidate ranker is not loaded: {status}")
    if not status.get("liveness_preparation"):
        raise RuntimeError("server liveness preparation is not enabled")

    policy = FrozenMultimodalIntentPolicy(
        checkpoint,
        calibration,
        max_consecutive_holds=2,
        enforce_goal_direction=True,
    )
    started_at = time.time()
    deadline = started_at + scenario.timeout_s
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    frame_hashes: list[str] = []
    transaction_ids: list[str] = []
    active: Optional[dict[str, Any]] = None
    stable_goal_samples = 0
    minimum_gap_m: Optional[float] = None
    maximum_collision_count = 0
    success = False
    failure_reason: Optional[str] = None

    trajectory_path = output_dir / "trajectory.jsonl"
    events_path = output_dir / "events.jsonl"
    intents_path = output_dir / "vehicle_intents.jsonl"

    while time.time() < deadline:
        now = time.time()
        telemetry = latest_telemetry(client, api_base)
        if not {str(value) for value in vehicle_ids}.issubset(telemetry):
            time.sleep(0.1)
            continue
        append_jsonl(
            trajectory_path,
            {"ts": now, "elapsed_s": now - started_at, "vehicles": telemetry},
        )
        gap = longitudinal_gap_m(
            {str(value): telemetry[str(value)] for value in vehicle_ids}
        )
        if gap is not None:
            minimum_gap_m = gap if minimum_gap_m is None else min(minimum_gap_m, gap)

        ego = telemetry[str(ego_id)]
        maximum_collision_count = max(
            maximum_collision_count,
            sum(
                int(telemetry[str(veh_id)].get("collision_count") or 0)
                for veh_id in vehicle_ids
            ),
        )
        if maximum_collision_count:
            failure_reason = "collision"
            break
        lane_state = (ego.get("lane_change") or {}).get("state")
        if active is not None:
            elapsed = now - float(active["started_at"])
            complete = steps_complete(active["steps"], telemetry, elapsed)
            if complete:
                post_outcome(
                    client,
                    api_base,
                    active["transaction_id"],
                    "completed",
                    {"elapsed_s": elapsed, "steps": active["steps"]},
                )
                append_jsonl(
                    events_path,
                    {
                        "ts": now,
                        "event": "transaction_completed",
                        "transaction_id": active["transaction_id"],
                        "elapsed_s": elapsed,
                    },
                )
                active = None
            elif elapsed > float(active["timeout_s"]):
                post_outcome(
                    client,
                    api_base,
                    active["transaction_id"],
                    "failed",
                    {"elapsed_s": elapsed, "reason": "executor_timeout"},
                )
                append_jsonl(
                    events_path,
                    {
                        "ts": now,
                        "event": "transaction_failed",
                        "transaction_id": active["transaction_id"],
                        "elapsed_s": elapsed,
                    },
                )
                active = None
            time.sleep(0.2)
            continue

        if ego.get("lane_id") == scenario.target_lane_id and lane_state in {
            None,
            "IDLE",
            "DONE",
        }:
            stable_goal_samples += 1
            if stable_goal_samples >= 3:
                success = True
                break
            # The goal is already reached. Do not ask the vehicle policy for a
            # new maneuver while collecting the settling samples.
            time.sleep(0.2)
            continue
        stable_goal_samples = 0

        if lane_state not in {None, "IDLE", "DONE"}:
            time.sleep(0.2)
            continue

        states = telemetry_to_states(telemetry)
        if ego_id not in states:
            failure_reason = "typed_ego_state_missing"
            break
        intent_result = policy.propose(
            states=states,
            ego_veh_id=ego_id,
            goal_lane_index=lane_index(scenario.target_lane_id),
            route_exit=False,
            deadline_s=max(0.1, deadline - now),
            now_s=now,
            flow_speed_kmh=50.0,
            lane_change_duration_s=3.0,
        )
        frame = wait_for_top_frame(client, api_base, ego_id)
        frame_hash = sha256_bytes(frame)
        frame_hashes.append(frame_hash)
        payload = proposal_to_review_payload(intent_result.proposal, frame)
        append_jsonl(
            intents_path,
            {
                "ts": now,
                "trace": intent_result.trace,
                "proposal": intent_result.proposal.model_dump(
                    mode="json", exclude_none=True
                ),
                "carla_top_frame_sha256": frame_hash,
                "carla_top_frame_bytes": len(frame),
                "policy_image_source": "typed-telemetry-render",
            },
        )
        response = client.post(f"{api_base}/mec/review", json=payload)
        response.raise_for_status()
        decision = response.json()
        if decision.get("decision_id") != intent_result.proposal.transaction_id:
            raise RuntimeError("MEC did not preserve the proposal transaction ID")
        transaction_ids.append(str(decision["decision_id"]))
        decision_counts[str(decision.get("decision"))] += 1
        reason_counts[str(decision.get("reason_code"))] += 1
        plans = decision_plans(decision)
        steps = first_steps(plans)
        executed = {
            veh_id: execute_step(client, api_base, veh_id, step)
            for veh_id, step in steps.items()
        }
        append_jsonl(
            events_path,
            {
                "ts": time.time(),
                "event": "mec_decision",
                "transaction_id": decision.get("decision_id"),
                "decision": decision,
                "executed_actions": executed,
                "carla_top_frame_sha256": frame_hash,
            },
        )
        if decision.get("decision") in {"allow", "override"} and steps:
            post_outcome(
                client,
                api_base,
                str(decision["decision_id"]),
                "executing",
                {"executed_actions": executed},
            )
            maximum_duration = max(
                float(step.get("duration_s") or 0.0) for step in steps.values()
            )
            active = {
                "transaction_id": str(decision["decision_id"]),
                "steps": steps,
                "started_at": time.time(),
                "timeout_s": max(10.0, maximum_duration + 8.0),
            }
        else:
            time.sleep(0.5)

    if active is not None:
        try:
            post_outcome(
                client,
                api_base,
                active["transaction_id"],
                "failed",
                {"reason": "scenario_terminated"},
            )
        except httpx.HTTPError:
            pass
    ended_at = time.time()
    if not success and failure_reason is None:
        failure_reason = "scenario_timeout"
    if archive:
        download_archive(client, api_base, output_dir)

    metadata = {
        "schema_version": "1.0",
        "claim_scope": "exploratory CARLA mechanism/transfer validation only",
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "seed": seed,
        "spawn_indices": list(scenario.spawn_indices),
        "vehicle_ids": vehicle_ids,
        "target_lane_id": scenario.target_lane_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": ended_at - started_at,
        "success": success,
        "failure_reason": failure_reason,
        "decision_counts": dict(decision_counts),
        "reason_code_counts": dict(reason_counts),
        "intent_count": len(transaction_ids),
        "transaction_ids_unique": len(transaction_ids) == len(set(transaction_ids)),
        "carla_frames_attached": len(frame_hashes),
        "carla_frame_hashes_unique": len(set(frame_hashes)),
        "minimum_longitudinal_gap_m": minimum_gap_m,
        "collision_count": maximum_collision_count,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()),
        "calibration": str(calibration.resolve()),
        "calibration_sha256": sha256_bytes(calibration.read_bytes()),
        "mec_status": status,
        "policy_image_source": "typed-telemetry-render",
        "mec_image_attachment": "actual CARLA top-down JPEG",
        "collision_measurement": "CARLA sensor.other.collision event count",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output",
        default="data/experiments/carla_guarded_transfer_pilot_20260827",
    )
    parser.add_argument("--scenarios", default="clear_change,contested_merge")
    parser.add_argument(
        "--checkpoint",
        default="data/models/multimodal_intent_policy_v4/checkpoint.pt",
    )
    parser.add_argument(
        "--calibration",
        default="data/derived/multimodal_intent_v5_calibration/calibration.json",
    )
    parser.add_argument("--seed", type=int, default=2026082700)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    output_root = Path(args.output)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    summaries = []
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{args.api_base.rstrip('/')}/health")
        health.raise_for_status()
        for index, name in enumerate(selected):
            summary = run_scenario(
                client,
                args.api_base.rstrip("/"),
                SCENARIOS[name],
                output_root / name,
                Path(args.checkpoint),
                Path(args.calibration),
                seed=args.seed + index,
                archive=not args.no_archive,
            )
            summaries.append(summary)
            print(
                f"{name}: success={summary['success']} "
                f"decisions={summary['decision_counts']} "
                f"duration={summary['duration_s']:.2f}s"
            )
    (output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
