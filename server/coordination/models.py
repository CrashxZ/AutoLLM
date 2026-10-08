"""Versioned data models for MIND-CAV coordination transactions."""

from __future__ import annotations

import math
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROTOCOL_VERSION = "2.0"


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Action(str, Enum):
    KEEP_LANE = "keep_lane"
    LANE_LEFT = "lane_left"
    LANE_RIGHT = "lane_right"
    SET_SPEED = "set_speed"
    ACCELERATE = "accelerate"
    YIELD = "yield"
    HOLD = "hold"
    BRAKE = "brake"


class GoalKind(str, Enum):
    TARGET_LANE = "target_lane"
    ROUTE_EXIT = "route_exit"
    MERGE = "merge"
    YIELD = "yield"
    MAINTAIN_FLOW = "maintain_flow"


class DecisionKind(str, Enum):
    ACK = "ACK"
    PLAN = "PLAN"
    NACK = "NACK"


class TransactionState(str, Enum):
    PROPOSED = "PROPOSED"
    REVIEWING = "REVIEWING"
    ACK = "ACK"
    PLAN = "PLAN"
    NACK = "NACK"
    COMMITTED = "COMMITTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class Goal(ProtocolModel):
    kind: GoalKind
    target_lane_id: Optional[int] = None
    exit_id: Optional[str] = None
    route_segment_id: Optional[str] = None
    target_speed_kmh: Optional[float] = Field(default=None, ge=0.0, le=160.0)
    deadline_s: Optional[float] = Field(default=None, gt=0.0)
    semantic: Optional[str] = None

    @model_validator(mode="after")
    def require_typed_target(self) -> "Goal":
        if self.kind in {GoalKind.TARGET_LANE, GoalKind.MERGE} and self.target_lane_id is None:
            raise ValueError(f"{self.kind.value} requires target_lane_id")
        if self.kind == GoalKind.ROUTE_EXIT and not (self.exit_id or self.route_segment_id):
            raise ValueError("route_exit requires exit_id or route_segment_id")
        return self


class PlanStep(ProtocolModel):
    step_id: str = Field(default_factory=lambda: f"step-{uuid.uuid4().hex[:10]}")
    action: Action
    target_lane_id: Optional[int] = None
    target_speed_kmh: Optional[float] = Field(default=None, ge=0.0, le=160.0)
    duration_s: float = Field(default=3.0, gt=0.0, le=30.0)
    completion_condition: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_target(self) -> "PlanStep":
        if self.action == Action.SET_SPEED and self.target_speed_kmh is None:
            raise ValueError("set_speed requires target_speed_kmh")
        return self


class VehiclePlan(ProtocolModel):
    veh_id: int = Field(ge=0)
    summary: str = ""
    steps: List[PlanStep] = Field(min_length=1, max_length=12)
    horizon_s: float = Field(default=8.0, gt=0.0, le=30.0)


class CooperationRequest(ProtocolModel):
    to_vehicle_ids: List[int] = Field(min_length=1, max_length=32)
    requested_action: Action
    reason: str = ""
    expires_at_s: float

    @model_validator(mode="after")
    def unique_targets(self) -> "CooperationRequest":
        if len(self.to_vehicle_ids) != len(set(self.to_vehicle_ids)):
            raise ValueError("cooperation request targets must be unique")
        return self


class VehicleState(ProtocolModel):
    veh_id: int = Field(ge=0)
    observed_at_s: float
    x_m: float
    y_m: float
    z_m: float = 0.0
    yaw_deg: float = 0.0
    speed_mps: float = Field(ge=0.0)
    desired_speed_mps: Optional[float] = Field(default=None, ge=0.0)
    acceleration_mps2: float = 0.0
    lane_id: int
    road_id: Optional[int] = None
    s_m: Optional[float] = None
    d_m: Optional[float] = None
    occupied_lane_ids: Optional[List[int]] = None
    lane_change_remaining_s: float = Field(default=0.0, ge=0.0, le=30.0)
    length_m: float = Field(default=4.7, gt=0.0)
    width_m: float = Field(default=1.9, gt=0.0)
    localization_error_m: float = Field(default=0.25, ge=0.0, le=10.0)
    vehicle_class: str = "standard"

    @classmethod
    def from_telemetry(cls, data: Dict[str, Any], observed_at_s: Optional[float] = None) -> "VehicleState":
        pose = data.get("pose") or {}
        if data.get("veh_id") is None or data.get("lane_id") is None:
            raise ValueError("telemetry requires veh_id and lane_id")
        speed_mps = float(data.get("speed_kmh") or 0.0) / 3.6
        return cls(
            veh_id=int(data["veh_id"]),
            observed_at_s=float(observed_at_s if observed_at_s is not None else data.get("ts", time.time())),
            x_m=float(pose.get("x", 0.0)),
            y_m=float(pose.get("y", 0.0)),
            z_m=float(pose.get("z", 0.0)),
            yaw_deg=float(pose.get("yaw", 0.0)),
            speed_mps=max(0.0, speed_mps),
            desired_speed_mps=(
                max(0.0, float(data["desired_speed_kmh"]) / 3.6)
                if data.get("desired_speed_kmh") is not None
                else None
            ),
            acceleration_mps2=float(data.get("acceleration_mps2") or 0.0),
            lane_id=int(data["lane_id"]),
            road_id=data.get("road_id"),
            s_m=data.get("s_m"),
            d_m=data.get("d_m"),
            occupied_lane_ids=data.get("occupied_lane_ids"),
            lane_change_remaining_s=float(
                data.get("lane_change_remaining_s") or 0.0
            ),
            length_m=float(data.get("length_m") or 4.7),
            width_m=float(data.get("width_m") or 1.9),
            localization_error_m=float(data.get("localization_error_m") or 0.25),
            vehicle_class=str(data.get("vehicle_class") or "standard"),
        )

    def longitudinal_position_m(self) -> float:
        if self.s_m is not None:
            # OpenDRIVE lanes with negative IDs follow the reference-line
            # direction, while positive-ID lanes travel against it. Convert
            # raw road ``s`` into a route-aligned coordinate that always
            # increases in the vehicle's direction of travel.
            return float(self.s_m) if self.lane_id < 0 else -float(self.s_m)
        yaw = math.radians(self.yaw_deg)
        return self.x_m * math.cos(yaw) + self.y_m * math.sin(yaw)

    def occupied_lanes(self) -> set[int]:
        if self.occupied_lane_ids:
            return {int(lane_id) for lane_id in self.occupied_lane_ids}
        return {self.lane_id}


class IntentProposal(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    transaction_id: str = Field(default_factory=lambda: f"txn-{uuid.uuid4().hex}")
    ego_veh_id: int = Field(ge=0)
    created_at_s: float = Field(default_factory=time.time)
    expires_at_s: Optional[float] = None
    observation_ts_s: float
    goal: Goal
    plan: VehiclePlan
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    request: Optional[CooperationRequest] = None
    protected_goal_vehicle_ids: List[int] = Field(
        default_factory=list,
        max_length=64,
    )
    image_sha256: Optional[str] = None
    top_frame_b64: Optional[str] = Field(default=None, exclude=True, repr=False)
    source: str = "vehicle"

    @model_validator(mode="after")
    def validate_identity_and_expiry(self) -> "IntentProposal":
        if self.expires_at_s is None:
            self.expires_at_s = self.created_at_s + 5.0
        if self.expires_at_s <= self.created_at_s:
            raise ValueError("expires_at_s must be after created_at_s")
        if self.plan.veh_id != self.ego_veh_id:
            raise ValueError("plan veh_id must match ego_veh_id")
        if self.request and self.ego_veh_id in self.request.to_vehicle_ids:
            raise ValueError("a vehicle cannot request itself")
        if len(self.protected_goal_vehicle_ids) != len(
            set(self.protected_goal_vehicle_ids)
        ):
            raise ValueError("protected goal vehicle IDs must be unique")
        return self

    def is_expired(self, now_s: Optional[float] = None) -> bool:
        now = time.time() if now_s is None else now_s
        return now >= float(self.expires_at_s)


class JointPlan(ProtocolModel):
    transaction_id: str
    plans: Dict[int, VehiclePlan] = Field(min_length=1, max_length=64)
    summary: str = ""
    proposer: str = "deterministic"

    @model_validator(mode="after")
    def validate_plan_keys(self) -> "JointPlan":
        for veh_id, plan in self.plans.items():
            if int(veh_id) != plan.veh_id:
                raise ValueError("joint plan key must match plan veh_id")
        return self


class Conflict(ProtocolModel):
    veh_a: int
    veh_b: int
    kind: str
    predicted_at_s: float = Field(ge=0.0)
    separation_m: Optional[float] = None
    threshold_m: Optional[float] = None
    ttc_s: Optional[float] = None
    details: str = ""


class ValidationResult(ProtocolModel):
    safe: bool
    checked_at_s: float = Field(default_factory=time.time)
    validator: str = "deterministic-v2"
    conflicts: List[Conflict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    checked_vehicle_ids: List[int] = Field(default_factory=list)
    total_pair_count: int = 0
    candidate_pair_count: int = 0
    sample_count: int = 0


class CandidateEvaluation(ProtocolModel):
    candidate_id: str
    template: str
    valid: bool
    expert_cost: Optional[float] = None
    predicted_cost: Optional[float] = None
    selected: bool = False
    rank: Optional[int] = None
    features: Dict[str, float] = Field(default_factory=dict)
    validation: ValidationResult


class ArbitrationDecision(ProtocolModel):
    transaction_id: str
    decision: DecisionKind
    reason_code: str
    reason: str
    decided_at_s: float = Field(default_factory=time.time)
    joint_plan: Optional[JointPlan] = None
    fallback_plans: Dict[int, VehiclePlan] = Field(default_factory=dict)
    initial_validation: Optional[ValidationResult] = None
    validation: Optional[ValidationResult] = None
    candidate_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    proposer: str = "none"


class AuditEvent(ProtocolModel):
    transaction_id: str
    state: TransactionState
    ts_s: float = Field(default_factory=time.time)
    vehicle_ids: List[int] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
