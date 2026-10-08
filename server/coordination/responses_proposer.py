"""Responses-API multimodal proposer used by the backbone evaluation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import httpx

from .models import Action, Conflict, IntentProposal, JointPlan, VehiclePlan, VehicleState
from .proposer import OpenAIJointPlanProposer


SYSTEM_INSTRUCTIONS = (
    "You propose short, coordinated highway maneuver plans. "
    "Use only listed vehicle IDs and actions. Do not claim that a plan is safe; "
    "a deterministic validator makes that decision."
)


def joint_plan_response_schema() -> dict:
    """Strict schema shared by every evaluated OpenAI backbone."""
    nullable_lane = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    nullable_speed = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    nullable_text = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "plans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "veh_id": {"type": "integer"},
                        "summary": {"type": "string"},
                        "horizon_s": {"type": "number", "minimum": 0.1, "maximum": 30.0},
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": [action.value for action in Action],
                                    },
                                    "target_lane_id": nullable_lane,
                                    "target_speed_kmh": nullable_speed,
                                    "duration_s": {
                                        "type": "number",
                                        "minimum": 0.1,
                                        "maximum": 30.0,
                                    },
                                    "description": nullable_text,
                                },
                                "required": [
                                    "action",
                                    "target_lane_id",
                                    "target_speed_kmh",
                                    "duration_s",
                                    "description",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["veh_id", "summary", "horizon_s", "steps"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "plans"],
        "additionalProperties": False,
    }


class OpenAIResponsesJointPlanProposer(OpenAIJointPlanProposer):
    """OpenAI proposer with image input and strict Responses structured output."""

    name = "openai-responses-joint-plan"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: str = "https://api.openai.com/v1/responses",
        timeout_s: float = 90.0,
        max_output_tokens: int = 500,
        max_response_bytes: int = 1024 * 1024,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            timeout_s=timeout_s,
        )
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self.transport = transport
        self.last_response_id: Optional[str] = None
        self.last_response_bytes = 0
        self.last_response_status: Optional[int] = None
        self.last_error_body = ""
        self.last_request_body: dict = {}

    def request_body(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> dict:
        prompt = self._prompt(
            proposal,
            states,
            conflicts,
            active_plans,
            include_schema_hint=False,
        )
        content: List[dict] = [{"type": "input_text", "text": prompt}]
        self.last_request_had_image = bool(proposal.top_frame_b64)
        if proposal.top_frame_b64:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{proposal.top_frame_b64}",
                    "detail": "low",
                }
            )
        return {
            "model": self.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "joint_plan_revision",
                    "strict": True,
                    "schema": joint_plan_response_schema(),
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        body = self.request_body(proposal, states, conflicts, active_plans)
        self.last_request_body = body
        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            transport=self.transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        self.last_response_bytes = len(response.content)
        self.last_response_status = response.status_code
        if response.is_error:
            self.last_error_body = response.text[:16384]
        if self.last_response_bytes > self.max_response_bytes:
            raise ValueError(
                "response exceeds byte limit:"
                f"{self.last_response_bytes}>{self.max_response_bytes}"
            )
        response.raise_for_status()
        data = response.json()
        self.last_response_id = data.get("id")
        self.last_usage = data.get("usage") or {}
        text = self._output_text(data).strip()
        self.last_response_text = text
        if not text:
            raise ValueError("empty proposer response")
        return self._parse(text, proposal, states)

    @staticmethod
    def _output_text(data: dict) -> str:
        chunks = []
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(str(content["text"]))
                elif content.get("type") == "refusal":
                    raise ValueError(f"model refusal: {content.get('refusal') or 'unspecified'}")
        if chunks:
            return "".join(chunks)
        output_text = data.get("output_text")
        return str(output_text) if output_text else ""
