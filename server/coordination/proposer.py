"""Semantic plan proposer interfaces.

The deterministic proposer is a reference implementation for tests and the
feasibility ladder. It is not presented as the final VLM component.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx

from .models import (
    Action,
    Conflict,
    IntentProposal,
    JointPlan,
    PlanStep,
    VehiclePlan,
    VehicleState,
)


class SemanticProposer(ABC):
    name = "semantic-proposer"

    @abstractmethod
    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        """Return one revised joint plan, or None when no revision is available."""


class DeterministicYieldProposer(SemanticProposer):
    """Reference proposer that delays the requester and honors explicit yields."""

    name = "deterministic-yield-reference"

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        ego = states.get(proposal.ego_veh_id)
        if ego is None:
            return None
        plans: Dict[int, VehiclePlan] = {
            proposal.ego_veh_id: VehiclePlan(
                veh_id=proposal.ego_veh_id,
                summary="Delay maneuver and preserve current lane",
                horizon_s=3.0,
                steps=[
                    PlanStep(
                        action=Action.HOLD,
                        target_lane_id=ego.lane_id,
                        target_speed_kmh=ego.speed_mps * 3.6,
                        duration_s=3.0,
                        description="Hold current lane before retrying the maneuver",
                    )
                ],
            )
        }
        if proposal.request:
            for target_id in proposal.request.to_vehicle_ids:
                target = states.get(target_id)
                if target is None:
                    return None
                requested = proposal.request.requested_action
                action = requested if requested in {Action.YIELD, Action.HOLD, Action.KEEP_LANE} else Action.YIELD
                speed = target.speed_mps * 3.6
                if action == Action.YIELD:
                    speed = max(0.0, speed - 10.0)
                plans[target_id] = VehiclePlan(
                    veh_id=target_id,
                    summary=f"Cooperate with vehicle {proposal.ego_veh_id}",
                    horizon_s=3.0,
                    steps=[
                        PlanStep(
                            action=action,
                            target_lane_id=target.lane_id,
                            target_speed_kmh=speed,
                            duration_s=3.0,
                            description=proposal.request.reason or "Create a safe coordination window",
                        )
                    ],
                )
        return JointPlan(
            transaction_id=proposal.transaction_id,
            plans=plans,
            summary="Deterministic conflict-delay revision",
            proposer=self.name,
        )


class UnavailableProposer(SemanticProposer):
    """Explicit unavailable proposer used to exercise fail-safe behavior."""

    name = "unavailable"

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        return None


class OpenAIJointPlanProposer(SemanticProposer):
    """One-shot multimodal joint-plan proposer.

    This component proposes semantics only. Every returned object remains subject
    to deterministic validation by the orchestrator.
    """

    name = "openai-joint-plan"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MEC_OPENAI_MODEL", "gpt-4o-mini")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.last_usage: Dict[str, int] = {}
        self.last_response_text: str = ""
        self.last_request_had_image: bool = False

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        prompt = self._prompt(proposal, states, conflicts, active_plans)
        content: List[dict] = [{"type": "text", "text": prompt}]
        self.last_request_had_image = bool(proposal.top_frame_b64)
        if proposal.top_frame_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{proposal.top_frame_b64}",
                        "detail": "low",
                    },
                }
            )
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You propose short, coordinated highway maneuver plans. "
                        "Use only listed vehicle IDs and actions. Do not claim that a plan is safe; "
                        "a deterministic validator makes that decision. Output only JSON."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 350,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        response.raise_for_status()
        data = response.json()
        self.last_usage = data.get("usage") or {}
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        self.last_response_text = text
        if not text:
            raise ValueError("empty proposer response")
        return self._parse(text, proposal, states)

    def _prompt(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
        *,
        include_schema_hint: bool = True,
    ) -> str:
        fleet = [
            {
                "veh_id": state.veh_id,
                "lane_id": state.lane_id,
                "s_m": round(state.longitudinal_position_m(), 2),
                "speed_kmh": round(state.speed_mps * 3.6, 2),
                "class": state.vehicle_class,
            }
            for state in sorted(states.values(), key=lambda item: item.veh_id)
        ]
        conflict_rows = [conflict.model_dump(mode="json", exclude_none=True) for conflict in conflicts]
        active = {
            str(veh_id): plan.model_dump(mode="json", exclude_none=True)
            for veh_id, plan in sorted(active_plans.items())
        }
        request = proposal.request.model_dump(mode="json") if proposal.request else None
        context = {
            "transaction_id": proposal.transaction_id,
            "ego_veh_id": proposal.ego_veh_id,
            "goal": proposal.goal.model_dump(mode="json", exclude_none=True),
            "proposed_plan": proposal.plan.model_dump(mode="json", exclude_none=True),
            "cooperation_request": request,
            "fleet": fleet,
            "detected_conflicts": conflict_rows,
            "active_plans": active,
        }
        schema = {
            "summary": "brief revision summary",
            "plans": {
                "101": {
                    "summary": "vehicle-specific plan",
                    "horizon_s": 3.0,
                    "steps": [
                        {
                            "action": "hold|keep_lane|lane_left|lane_right|set_speed|accelerate|yield|brake",
                            "target_lane_id": -1,
                            "target_speed_kmh": 40,
                            "duration_s": 3.0,
                            "description": "brief step",
                        }
                    ],
                }
            },
        }
        prompt = (
            "Revise the conflicting proposal into one compact joint plan. "
            "Return exactly one step per included vehicle. Every returned step starts now and "
            "executes simultaneously; descriptions such as 'after yielding' do not delay a step. "
            "If the ego target lane is occupied at the current longitudinal position, return only "
            "HOLD for the ego and YIELD for the blocker; do not include the lane change because "
            "the vehicle will replan after the gap opens. "
            "Prefer preserving 50 km/h traffic flow, but delay or yield when needed. "
            "Do not invent vehicles. If no useful revision exists, return {\"plans\":{}}.\n"
            f"Context:{json.dumps(context, separators=(',', ':'))}"
        )
        if include_schema_hint:
            prompt += f"\nOutput schema:{json.dumps(schema, separators=(',', ':'))}"
        return prompt

    def _parse(
        self,
        text: str,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
    ) -> Optional[JointPlan]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        raw_plans = parsed.get("plans")
        if not isinstance(raw_plans, (dict, list)) or not raw_plans:
            return None
        plans: Dict[int, VehiclePlan] = {}
        if isinstance(raw_plans, dict):
            plan_rows = raw_plans.items()
        else:
            plan_rows = (
                (raw_plan.get("veh_id"), raw_plan)
                for raw_plan in raw_plans
                if isinstance(raw_plan, dict)
            )
        for raw_id, raw_plan in plan_rows:
            if raw_id is None:
                raise ValueError("plan omitted veh_id")
            veh_id = int(raw_id)
            if veh_id not in states:
                raise ValueError(f"proposer invented vehicle:{veh_id}")
            if veh_id in plans:
                raise ValueError(f"duplicate plan for vehicle:{veh_id}")
            if not isinstance(raw_plan, dict):
                raise ValueError(f"invalid plan for vehicle:{veh_id}")
            normalized_steps = []
            for raw_step in raw_plan.get("steps") or []:
                if not isinstance(raw_step, dict):
                    continue
                step = dict(raw_step)
                if "id" in step and "step_id" not in step:
                    step["step_id"] = step.pop("id")
                action = str(step.get("action") or "hold").strip().lower().replace(" ", "_")
                action = {"speed_down": "yield", "slow_down": "yield", "speed_up": "accelerate"}.get(
                    action, action
                )
                step["action"] = action
                normalized_steps.append(PlanStep.model_validate(step))
            if not normalized_steps:
                raise ValueError(f"empty plan for vehicle:{veh_id}")
            plans[veh_id] = VehiclePlan(
                veh_id=veh_id,
                summary=str(raw_plan.get("summary") or "semantic revision"),
                horizon_s=float(raw_plan.get("horizon_s") or 3.0),
                steps=normalized_steps,
            )
        if proposal.ego_veh_id not in plans:
            raise ValueError("revision omitted ego vehicle")
        return JointPlan(
            transaction_id=proposal.transaction_id,
            plans=plans,
            summary=str(parsed.get("summary") or "semantic joint-plan revision"),
            proposer=f"{self.name}:{self.model}",
        )
