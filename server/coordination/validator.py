"""Deterministic finite-horizon validation for proposed joint plans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

from .conflict_graph import (
    PredictionConfig,
    build_conflict_graph,
    target_lane_for,
)
from .models import Action, JointPlan, PlanStep, ValidationResult, VehiclePlan, VehicleState


@dataclass(frozen=True)
class ValidationConfig(PredictionConfig):
    max_observation_age_s: float = 2.0
    max_plan_steps: int = 12
    allowed_lane_ids: Optional[Set[int]] = None


class DeterministicPlanValidator:
    """Validate plans without trusting semantic-model safety judgments."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()

    def validate(
        self,
        joint_plan: JointPlan,
        states: Dict[int, VehicleState],
        active_plans: Optional[Dict[int, VehiclePlan]] = None,
        now_s: Optional[float] = None,
    ) -> ValidationResult:
        now = time.time() if now_s is None else now_s
        errors = []
        active = active_plans or {}

        for veh_id, state in states.items():
            age = now - state.observed_at_s
            if age < -0.5:
                errors.append(f"future_observation:{veh_id}")
            elif age > self.config.max_observation_age_s:
                errors.append(f"stale_observation:{veh_id}:{age:.3f}")

        for veh_id, plan in joint_plan.plans.items():
            state = states.get(veh_id)
            if state is None:
                errors.append(f"missing_state:{veh_id}")
                continue
            if len(plan.steps) > self.config.max_plan_steps:
                errors.append(f"too_many_steps:{veh_id}")
            self._validate_steps(plan, errors)
            active_plan = active.get(veh_id)
            if active_plan is not None:
                active_target = target_lane_for(active_plan, state.lane_id)
                revised_target = target_lane_for(plan, state.lane_id)
                if (
                    active_target != state.lane_id
                    and revised_target != active_target
                ):
                    errors.append(
                        "active_lateral_commitment_mismatch:"
                        f"{veh_id}:{active_target}:{revised_target}"
                    )

        merged = {veh_id: plan for veh_id, plan in active.items() if veh_id not in joint_plan.plans}
        merged.update(joint_plan.plans)
        for veh_id, state in states.items():
            if veh_id not in merged:
                target_speed_mps = (
                    state.desired_speed_mps
                    if state.desired_speed_mps is not None
                    else state.speed_mps
                )
                merged[veh_id] = VehiclePlan(
                    veh_id=veh_id,
                    summary="implicit constant-lane prediction",
                    horizon_s=self.config.horizon_s,
                    steps=[
                        PlanStep(
                            action=Action.KEEP_LANE,
                            target_lane_id=state.lane_id,
                            target_speed_kmh=target_speed_mps * 3.6,
                            duration_s=self.config.horizon_s,
                        )
                    ],
                )
        missing_active_states = sorted(set(merged) - set(states))
        errors.extend(f"missing_state:{veh_id}" for veh_id in missing_active_states)

        conflicts = []
        total_pair_count = 0
        candidate_pair_count = 0
        sample_count = 0
        if not errors:
            complete_plan = JointPlan(
                transaction_id=joint_plan.transaction_id,
                plans=merged,
                summary=joint_plan.summary,
                proposer=joint_plan.proposer,
            )
            graph = build_conflict_graph(complete_plan, states, self.config)
            conflicts = graph.conflicts
            total_pair_count = graph.total_pair_count
            candidate_pair_count = graph.candidate_pair_count
            sample_count = graph.sample_count

        return ValidationResult(
            safe=not errors and not conflicts,
            conflicts=conflicts,
            errors=errors,
            checked_vehicle_ids=sorted(joint_plan.plans),
            total_pair_count=total_pair_count,
            candidate_pair_count=candidate_pair_count,
            sample_count=sample_count,
        )

    def _validate_steps(self, plan: VehiclePlan, errors: list[str]) -> None:
        for index, step in enumerate(plan.steps):
            prefix = f"invalid_step:{plan.veh_id}:{index}"
            if self.config.allowed_lane_ids is not None and step.target_lane_id is not None:
                if step.target_lane_id not in self.config.allowed_lane_ids:
                    errors.append(f"{prefix}:lane")
            if step.target_speed_kmh is not None:
                if step.target_speed_kmh > self.config.max_speed_mps * 3.6:
                    errors.append(f"{prefix}:speed")
            if step.action in {Action.LANE_LEFT, Action.LANE_RIGHT}:
                if step.duration_s < self.config.lane_change_duration_s * 0.5:
                    errors.append(f"{prefix}:lane_change_duration")
