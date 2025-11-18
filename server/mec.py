# server/mec.py
"""
MEC (Multi-access Edge Computing) safety/coordination layer.

Receives vehicle intent/plan proposals, aggregates global telemetry, and
consults OpenAI to approve, reject, or override with a new plan. Decisions are
recorded for later visualization.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Literal

import httpx

SAFETY_OPTIONS: Dict[str, str] = {
  "strict": "Prioritize zero-risk behaviour; reject maneuvers unless fully conflict-free.",
  "balanced": "Allow maneuvers with small gaps when mitigation exists; override only on clear conflicts.",
  "relaxed": "Approve most plans unless an imminent collision is likely; prefer keeping traffic flowing.",
}

class MecDecision:
  """Small struct representing a MEC decision."""

  def __init__(self, veh_id: int, decision: str, reason: str, plan: Optional[dict], model: str):
    self.veh_id = veh_id
    self.decision = decision
    self.reason = reason
    self.plan = plan
    self.model = model
    self.ts = time.time()
    self.decision_id = f"mec-{int(self.ts * 1000)}-{veh_id}"

  def to_dict(self) -> Dict[str, Any]:
    return {
      "veh_id": self.veh_id,
      "decision": self.decision,
      "reason": self.reason,
      "plan": self.plan,
      "model": self.model,
      "ts": self.ts,
      "decision_id": self.decision_id,
    }


class MECController:
  """
  Coordinates MEC decisions by invoking OpenAI with a holistic prompt that
  includes telemetry snapshot + the requested plan for a vehicle.
  """

  def __init__(self, model: str = "gpt-4o-mini", history_size: int = 200):
    self.api_key = os.getenv("OPENAI_API_KEY")
    self.model = os.getenv("MEC_OPENAI_MODEL", model)
    self.history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
    posture = (os.getenv("MEC_SAFETY_POSTURE") or "strict").lower()
    self.safety_posture = posture if posture in SAFETY_OPTIONS else "strict"

  async def review(self, payload, telemetry_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Review a single plan request. Returns a dict with decision + optional plan.
    """
    # If no API key set, pass-through.
    if not self.api_key:
      decision = MecDecision(payload.veh_id, "allow", "OPENAI_API_KEY missing — MEC bypassed.", None, "noop")
      self.history.appendleft(decision.to_dict())
      return decision.to_dict()

    prompt = self._build_prompt(payload, telemetry_snapshot)
    content = [{"type": "text", "text": prompt}]
    top_b64 = getattr(payload, "top_frame_b64", None)
    if top_b64:
      content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{top_b64}"},
      })

    body = {
      "model": self.model,
      "messages": [
        {
          "role": "system",
          "content": "You are the MEC controller for an autonomous driving fleet. "
          "You must approve or override requested plans based on fleet-wide safety.",
        },
        {
          "role": "user",
          "content": content,
        },
        {
          "role": "user",
          "content": (
            "Respond ONLY with JSON: "
            '{"decision":"allow|override|reject","reason":"...","plan":[{'
            '"description":"...", "action":"lane_left|lane_right|set_speed|speed_up|speed_down|hold|brake",'
            '"target_lane_id": number|null, "target_speed_kmh": number|null}]}'
          ),
        },
      ],
      "temperature": 0.2,
      "max_tokens": 500,
    }

    try:
      async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
          "https://api.openai.com/v1/chat/completions",
          headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
          },
          json=body,
        )
      resp.raise_for_status()
      data = resp.json()
      text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
      parsed = self._parse_response(text)
    except Exception as err:
      decision = MecDecision(payload.veh_id, "allow", f"MEC fallback: {err}", None, self.model)
      self.history.appendleft(decision.to_dict())
      return decision.to_dict()

    decision = MecDecision(
      payload.veh_id,
      parsed.get("decision") or "allow",
      parsed.get("reason") or "Allowed by MEC.",
      self._normalize_plan(parsed.get("plan")),
      self.model,
    )
    self.history.appendleft(decision.to_dict())
    return decision.to_dict()

  def get_history(self) -> List[Dict[str, Any]]:
    return list(self.history)

  def set_safety_posture(self, posture: Literal["strict", "balanced", "relaxed"]) -> None:
    val = posture.lower()
    if val not in SAFETY_OPTIONS:
      raise ValueError(f"Invalid safety posture: {posture}")
    self.safety_posture = val

  def get_safety_posture(self) -> str:
    return self.safety_posture

  def _build_prompt(self, payload, telemetry_snapshot: Dict[str, Any]) -> str:
    veh = payload.veh_id
    ctx = payload.context or {}
    goal = ctx.get("goal") or payload.plan.get("goal") or ctx.get("intent_goal") or ""
    plan_txt = self._format_plan(payload.plan)
    vehicles_txt = self._format_telemetry(telemetry_snapshot, highlight=veh)
    req = payload.intent or {}
    req_txt = json.dumps(req, separators=(",", ":"))
    posture_clause = (
      f"Safety posture: {self.safety_posture.upper()} - {SAFETY_OPTIONS[self.safety_posture]}"
    )
    return (
      f"MEC evaluation request:\nVehicle: {veh}\nGoal: {goal}\n"
      f"{posture_clause}\n"
      f"Proposed plan:\n{plan_txt}\n"
      f"Intent/request payload: {req_txt}\n\n"
      f"Fleet telemetry snapshot:\n{vehicles_txt}\n"
      "Decide if this plan keeps all vehicles safe. "
      "If unsafe, supply a safer multi-step plan."
    )

  def _format_plan(self, plan: Dict[str, Any]) -> str:
    if not plan:
      return "No plan provided."
    steps = plan.get("steps") or []
    lines = [f"- Summary: {plan.get('summary') or 'N/A'}"]
    for idx, step in enumerate(steps, start=1):
      action = step.get("action") or "hold"
      desc = step.get("description") or step.get("label") or ""
      lane = step.get("target_lane_id")
      speed = step.get("target_speed_kmh")
      extra = []
      if lane is not None:
        extra.append(f"lane {lane}")
      if speed is not None:
        extra.append(f"{speed} km/h")
      tail = f" ({', '.join(extra)})" if extra else ""
      lines.append(f"  {idx}. {action}{tail} - {desc}")
    return "\n".join(lines)

  def _format_telemetry(self, telemetry_snapshot: Dict[str, Any], highlight: int) -> str:
    entries = []
    for key, data in list(telemetry_snapshot.items())[:12]:
      veh_id = data.get("veh_id") or int(key)
      prefix = "*" if veh_id == highlight else "-"
      entries.append(
        f"{prefix} veh={veh_id} speed={data.get('speed_kmh','?')}km/h lane={data.get('lane_id','?')}"
        f" d2c={round(data.get('distance_to_center', 0.0), 2)} lc={data.get('lane_change', {}).get('state','?')}"
      )
    return "\n".join(entries) or "No telemetry."

  def _parse_response(self, text: str) -> Dict[str, Any]:
    if not text:
      return {"decision": "allow", "reason": "Empty response from MEC model."}
    cleaned = text.strip()
    if cleaned.startswith("```"):
      cleaned = cleaned.strip("`")
      cleaned = cleaned.replace("json", "", 1).strip()
    try:
      return json.loads(cleaned)
    except json.JSONDecodeError:
      return {"decision": "allow", "reason": f"MEC parse fallback: {cleaned[:120]}"}

  def _normalize_plan(self, plan_obj: Any) -> Optional[dict]:
    if not plan_obj:
      return None
    if isinstance(plan_obj, list):
      summary = "MEC override plan"
      steps = plan_obj
    else:
      steps = plan_obj.get("steps")
      summary = plan_obj.get("summary") or "MEC override plan"
    if not isinstance(steps, list):
      return None
    normalized_steps = []
    for idx, step in enumerate(steps, start=1):
      if not isinstance(step, dict):
        continue
      normalized_steps.append({
        "id": step.get("id") or f"mec-step-{idx}",
        "description": step.get("description") or step.get("label") or "",
        "action": step.get("action") or "hold",
        "target_lane_id": step.get("target_lane_id"),
        "target_speed_kmh": step.get("target_speed_kmh"),
      })
    if not normalized_steps:
      return None
    return {
      "summary": summary,
      "steps": normalized_steps,
    }
