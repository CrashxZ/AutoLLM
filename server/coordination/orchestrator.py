"""Validator-gated orchestration for MIND-CAV v2."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

from .models import (
    Action,
    ArbitrationDecision,
    DecisionKind,
    IntentProposal,
    JointPlan,
    PlanStep,
    TransactionState,
    ValidationResult,
    VehiclePlan,
    VehicleState,
)
from .proposer import SemanticProposer, UnavailableProposer
from .transaction import ActivePlanStore
from .validator import DeterministicPlanValidator


class CoordinationOrchestrator:
    def __init__(
        self,
        validator: Optional[DeterministicPlanValidator] = None,
        proposer: Optional[SemanticProposer] = None,
        store: Optional[ActivePlanStore] = None,
        proposer_timeout_s: float = 30.0,
        semantic_call_cap: int = 500,
    ) -> None:
        self.validator = validator or DeterministicPlanValidator()
        self.proposer = proposer or UnavailableProposer()
        self.store = store or ActivePlanStore()
        self.proposer_timeout_s = proposer_timeout_s
        self.semantic_call_cap = semantic_call_cap
        self.semantic_calls = 0

    def reset(self) -> None:
        """Clear session-scoped transactions while preserving configured components."""
        self.store = ActivePlanStore()
        self.semantic_calls = 0

    async def review(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        now_s: Optional[float] = None,
    ) -> ArbitrationDecision:
        started = time.perf_counter()
        if hasattr(self.proposer, "last_evaluations"):
            self.proposer.last_evaluations = []
        now = time.time() if now_s is None else now_s
        self.store.register(proposal)
        self.store.transition(proposal.transaction_id, TransactionState.REVIEWING)

        preflight_errors = self._preflight_errors(proposal, states, now)
        if preflight_errors:
            validation = ValidationResult(
                safe=False,
                errors=preflight_errors,
                checked_vehicle_ids=[proposal.ego_veh_id],
            )
            return self._nack(
                proposal,
                states,
                validation,
                "PRECHECK_FAILED",
                "; ".join(preflight_errors),
                started,
                initial_validation=validation,
            )

        active = self.store.active_plans(now)
        original = JointPlan(
            transaction_id=proposal.transaction_id,
            plans={proposal.ego_veh_id: proposal.plan},
            summary=proposal.plan.summary,
            proposer="vehicle",
        )
        initial_validation = self.validator.validate(original, states, active, now)
        if initial_validation.safe:
            return self._approve(
                proposal,
                original,
                initial_validation,
                DecisionKind.ACK,
                "FAST_PATH_SAFE",
                "Proposed plan passed deterministic validation.",
                started,
                initial_validation=initial_validation,
                now_s=now,
            )

        if self.semantic_calls >= self.semantic_call_cap:
            return self._nack(
                proposal,
                states,
                initial_validation,
                "SEMANTIC_CALL_CAP",
                "Semantic proposal cap reached; safe fallback selected.",
                started,
                initial_validation=initial_validation,
            )

        self.semantic_calls += 1
        try:
            revised = await asyncio.wait_for(
                self.proposer.propose(
                    proposal,
                    states,
                    initial_validation.conflicts,
                    active,
                ),
                timeout=self.proposer_timeout_s,
            )
        except asyncio.TimeoutError:
            return self._nack(
                proposal,
                states,
                initial_validation,
                "PROPOSER_TIMEOUT",
                "Semantic proposer timed out; safe fallback selected.",
                started,
                initial_validation=initial_validation,
            )
        except Exception as exc:
            return self._nack(
                proposal,
                states,
                initial_validation,
                "PROPOSER_ERROR",
                f"Semantic proposer failed ({type(exc).__name__}); safe fallback selected.",
                started,
                initial_validation=initial_validation,
            )

        if revised is None:
            return self._nack(
                proposal,
                states,
                initial_validation,
                "NO_SAFE_REVISION",
                "Semantic proposer returned no revision; safe fallback selected.",
                started,
                initial_validation=initial_validation,
            )
        if revised.transaction_id != proposal.transaction_id:
            return self._nack(
                proposal,
                states,
                ValidationResult(safe=False, errors=["transaction_id_mismatch"]),
                "INVALID_REVISION",
                "Revised plan transaction ID did not match the request.",
                started,
                initial_validation=initial_validation,
            )

        revised_validation = self.validator.validate(revised, states, active, now)
        if not revised_validation.safe:
            return self._nack(
                proposal,
                states,
                revised_validation,
                "REVISION_UNSAFE",
                "Revised plan failed deterministic validation.",
                started,
                initial_validation=initial_validation,
            )
        return self._approve(
            proposal,
            revised,
            revised_validation,
            DecisionKind.PLAN,
            "VALIDATED_REVISION",
            "Revised joint plan passed deterministic validation.",
            started,
            initial_validation=initial_validation,
            now_s=now,
        )

    def _preflight_errors(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        now_s: float,
    ) -> list[str]:
        errors = []
        if proposal.is_expired(now_s):
            errors.append("expired_proposal")
        if proposal.ego_veh_id not in states:
            errors.append(f"missing_ego_state:{proposal.ego_veh_id}")
        if proposal.request:
            missing = sorted(set(proposal.request.to_vehicle_ids) - set(states))
            errors.extend(f"unknown_request_target:{veh_id}" for veh_id in missing)
            if proposal.request.expires_at_s <= now_s:
                errors.append("expired_cooperation_request")
        return errors

    def _approve(
        self,
        proposal: IntentProposal,
        plan: JointPlan,
        validation: ValidationResult,
        kind: DecisionKind,
        reason_code: str,
        reason: str,
        started: float,
        initial_validation: Optional[ValidationResult] = None,
        now_s: Optional[float] = None,
    ) -> ArbitrationDecision:
        decision = ArbitrationDecision(
            transaction_id=proposal.transaction_id,
            decision=kind,
            reason_code=reason_code,
            reason=reason,
            joint_plan=plan,
            initial_validation=initial_validation,
            validation=validation,
            candidate_evaluations=list(getattr(self.proposer, "last_evaluations", [])),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            proposer=plan.proposer,
        )
        self.store.set_decision(decision)
        self.store.commit(proposal.transaction_id, plan, validation, now_s=now_s)
        return decision

    def _nack(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        validation: ValidationResult,
        reason_code: str,
        reason: str,
        started: float,
        initial_validation: Optional[ValidationResult] = None,
    ) -> ArbitrationDecision:
        fallback = self._fallback_plans(proposal, states)
        decision = ArbitrationDecision(
            transaction_id=proposal.transaction_id,
            decision=DecisionKind.NACK,
            reason_code=reason_code,
            reason=reason,
            fallback_plans=fallback,
            initial_validation=initial_validation,
            validation=validation,
            candidate_evaluations=list(getattr(self.proposer, "last_evaluations", [])),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            proposer=self.proposer.name,
        )
        self.store.set_decision(decision)
        self.store.close_nack(proposal.transaction_id)
        return decision

    @staticmethod
    def _fallback_plans(
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
    ) -> Dict[int, VehiclePlan]:
        state = states.get(proposal.ego_veh_id)
        if state is None:
            return {}
        return {
            proposal.ego_veh_id: VehiclePlan(
                veh_id=proposal.ego_veh_id,
                summary="Safe fallback: preserve lane and bounded speed",
                horizon_s=2.0,
                steps=[
                    PlanStep(
                        action=Action.HOLD,
                        target_lane_id=state.lane_id,
                        target_speed_kmh=state.speed_mps * 3.6,
                        duration_s=2.0,
                        description="No new maneuver authorized",
                    )
                ],
            )
        }
