"""Compatibility adapter for the existing dashboard and `/mec/review` API."""

from __future__ import annotations

import base64
import hashlib
import time
from typing import Any, Dict, Optional, Tuple

from .models import (
    Action,
    ArbitrationDecision,
    CooperationRequest,
    DecisionKind,
    Goal,
    GoalKind,
    IntentProposal,
    PlanStep,
    VehiclePlan,
    VehicleState,
)


ACTION_ALIASES = {
    "keep": Action.KEEP_LANE,
    "keep_lane": Action.KEEP_LANE,
    "hold": Action.HOLD,
    "lane_left": Action.LANE_LEFT,
    "lane left": Action.LANE_LEFT,
    "left": Action.LANE_LEFT,
    "lane_right": Action.LANE_RIGHT,
    "lane right": Action.LANE_RIGHT,
    "right": Action.LANE_RIGHT,
    "set_speed": Action.SET_SPEED,
    "speed_up": Action.ACCELERATE,
    "accelerate": Action.ACCELERATE,
    "speed_down": Action.YIELD,
    "slow_down": Action.YIELD,
    "slow down": Action.YIELD,
    "yield": Action.YIELD,
    "brake": Action.BRAKE,
}


def normalize_action(value: Any) -> Action:
    key = str(value or "hold").strip().lower()
    return ACTION_ALIASES.get(key, Action.HOLD)


def telemetry_to_states(telemetry: Dict[str, Any]) -> Dict[int, VehicleState]:
    states: Dict[int, VehicleState] = {}
    for raw_id, row in telemetry.items():
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item.setdefault("veh_id", int(raw_id))
        try:
            state = VehicleState.from_telemetry(item)
        except (TypeError, ValueError):
            continue
        states[state.veh_id] = state
    return states


def payload_to_proposal(payload: Any, states: Dict[int, VehicleState]) -> IntentProposal:
    now = time.time()
    veh_id = int(payload.veh_id)
    intent = getattr(payload, "intent", None) or {}
    context = getattr(payload, "context", None) or {}
    raw_plan = getattr(payload, "plan", None) or {}
    raw_steps = raw_plan.get("steps") or raw_plan.get("plan_steps") or []
    steps = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        action = normalize_action(raw.get("action"))
        steps.append(
            PlanStep(
                step_id=str(raw.get("id") or raw.get("step_id") or f"legacy-{index + 1}"),
                action=action,
                target_lane_id=raw.get("target_lane_id"),
                target_speed_kmh=raw.get("target_speed_kmh"),
                duration_s=float(raw.get("duration_s") or (3.0 if action in {Action.LANE_LEFT, Action.LANE_RIGHT} else 2.0)),
                completion_condition=raw.get("completion_condition"),
                description=raw.get("description") or raw.get("label"),
            )
        )
    if not steps:
        action = normalize_action(intent.get("ego_action") or intent.get("action"))
        state = states.get(veh_id)
        target_speed = intent.get("target_speed_kmh")
        if action == Action.SET_SPEED and target_speed is None:
            target_speed = state.speed_mps * 3.6 if state else 0.0
        steps = [
            PlanStep(
                step_id="legacy-1",
                action=action,
                target_lane_id=intent.get("target_lane_id"),
                target_speed_kmh=target_speed,
                duration_s=3.0 if action in {Action.LANE_LEFT, Action.LANE_RIGHT} else 2.0,
                description=intent.get("reason"),
            )
        ]

    target_lane = next((step.target_lane_id for step in reversed(steps) if step.target_lane_id is not None), None)
    goal_text = str(getattr(payload, "goal", None) or context.get("goal") or intent.get("goal") or "")
    if "exit" in goal_text.lower():
        goal = Goal(
            kind=GoalKind.ROUTE_EXIT,
            exit_id=str(context.get("exit_id") or "legacy-exit"),
            target_lane_id=target_lane,
            semantic=goal_text,
        )
    elif target_lane is not None:
        goal = Goal(kind=GoalKind.TARGET_LANE, target_lane_id=target_lane, semantic=goal_text or None)
    else:
        target_speed = next(
            (step.target_speed_kmh for step in reversed(steps) if step.target_speed_kmh is not None),
            50.0,
        )
        goal = Goal(
            kind=GoalKind.MAINTAIN_FLOW,
            target_speed_kmh=target_speed,
            semantic=goal_text or None,
        )

    request = _parse_request(getattr(payload, "request", None) or intent.get("request"), now)
    top_b64 = getattr(payload, "top_frame_b64", None)
    image_sha = None
    if top_b64:
        try:
            image_sha = hashlib.sha256(base64.b64decode(top_b64, validate=True)).hexdigest()
        except (ValueError, TypeError):
            image_sha = "invalid-base64"

    state = states.get(veh_id)
    observation_ts = state.observed_at_s if state else now
    return IntentProposal(
        ego_veh_id=veh_id,
        created_at_s=now,
        expires_at_s=now + float(context.get("intent_ttl_s") or 5.0),
        observation_ts_s=observation_ts,
        goal=goal,
        plan=VehiclePlan(
            veh_id=veh_id,
            summary=str(raw_plan.get("summary") or intent.get("plan_summary") or "legacy proposal"),
            steps=steps,
            horizon_s=float(raw_plan.get("horizon_s") or 8.0),
        ),
        confidence=float(intent.get("confidence") if intent.get("confidence") is not None else 1.0),
        request=request,
        protected_goal_vehicle_ids=[
            int(value)
            for value in context.get("protected_goal_vehicle_ids", [])
        ],
        image_sha256=image_sha,
        top_frame_b64=top_b64,
        source="legacy-api",
    )


def _parse_request(raw: Any, now_s: float) -> Optional[CooperationRequest]:
    if not isinstance(raw, dict):
        return None
    targets = raw.get("to_vehicle_ids") or raw.get("to") or []
    if isinstance(targets, (str, int)):
        targets = [targets]
    parsed_targets = []
    for target in targets:
        try:
            parsed_targets.append(int(target))
        except (TypeError, ValueError):
            continue
    ask = str(raw.get("requested_action") or raw.get("ask") or "").strip().lower()
    if not parsed_targets or ask in {"", "none", "null"}:
        return None
    return CooperationRequest(
        to_vehicle_ids=parsed_targets,
        requested_action=normalize_action(ask),
        reason=str(raw.get("reason") or ask),
        expires_at_s=float(raw.get("expires_at_s") or now_s + 5.0),
    )


def decision_to_legacy(decision: ArbitrationDecision, ego_veh_id: int) -> Dict[str, Any]:
    legacy_kind = {
        DecisionKind.ACK: "allow",
        DecisionKind.PLAN: "override",
        DecisionKind.NACK: "reject",
    }[decision.decision]
    ego_plan = None
    if decision.joint_plan and ego_veh_id in decision.joint_plan.plans:
        ego_plan = _plan_to_legacy(decision.joint_plan.plans[ego_veh_id])
    fallback = decision.fallback_plans.get(ego_veh_id)
    return {
        "veh_id": ego_veh_id,
        "decision": legacy_kind,
        "reason": decision.reason,
        "reason_code": decision.reason_code,
        "plan": ego_plan,
        "joint_plan": decision.joint_plan.model_dump(mode="json", exclude_none=True)
        if decision.joint_plan
        else None,
        "fallback_plan": _plan_to_legacy(fallback) if fallback else None,
        "validation": decision.validation.model_dump(mode="json", exclude_none=True)
        if decision.validation
        else None,
        "initial_validation": decision.initial_validation.model_dump(
            mode="json", exclude_none=True
        )
        if decision.initial_validation
        else None,
        "candidate_evaluations": [
            item.model_dump(mode="json", exclude_none=True)
            for item in decision.candidate_evaluations
        ],
        "model": decision.proposer,
        "ts": decision.decided_at_s,
        "decision_id": decision.transaction_id,
        "latency_ms": decision.latency_ms,
    }


def _plan_to_legacy(plan: VehiclePlan) -> Dict[str, Any]:
    return {
        "summary": plan.summary,
        "horizon_s": plan.horizon_s,
        "steps": [
            {
                "id": step.step_id,
                "description": step.description or "",
                "action": step.action.value,
                "target_lane_id": step.target_lane_id,
                "target_speed_kmh": step.target_speed_kmh,
                "duration_s": step.duration_s,
                "completion_condition": step.completion_condition,
            }
            for step in plan.steps
        ],
    }
