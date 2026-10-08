"""Thread-safe transaction lifecycle and atomic active-plan storage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .audit import AuditLog
from .models import (
    ArbitrationDecision,
    AuditEvent,
    IntentProposal,
    JointPlan,
    TransactionState,
    ValidationResult,
    VehiclePlan,
)


ALLOWED_TRANSITIONS = {
    TransactionState.PROPOSED: {TransactionState.REVIEWING, TransactionState.EXPIRED},
    TransactionState.REVIEWING: {
        TransactionState.ACK,
        TransactionState.PLAN,
        TransactionState.NACK,
        TransactionState.EXPIRED,
        TransactionState.FAILED,
    },
    TransactionState.ACK: {TransactionState.COMMITTED, TransactionState.FAILED},
    TransactionState.PLAN: {TransactionState.COMMITTED, TransactionState.FAILED},
    TransactionState.NACK: {TransactionState.CLOSED},
    TransactionState.COMMITTED: {TransactionState.EXECUTING, TransactionState.EXPIRED},
    TransactionState.EXECUTING: {
        TransactionState.COMPLETED,
        TransactionState.FAILED,
        TransactionState.EXPIRED,
    },
    TransactionState.COMPLETED: {TransactionState.CLOSED},
    TransactionState.FAILED: {TransactionState.CLOSED},
    TransactionState.EXPIRED: {TransactionState.CLOSED},
    TransactionState.CLOSED: set(),
}


@dataclass
class TransactionRecord:
    proposal: IntentProposal
    state: TransactionState
    updated_at_s: float
    decision: Optional[ArbitrationDecision] = None


@dataclass
class ActivePlan:
    transaction_id: str
    plan: VehiclePlan
    committed_at_s: float
    expires_at_s: float


class ActivePlanStore:
    def __init__(self, audit: Optional[AuditLog] = None) -> None:
        self.audit = audit or AuditLog()
        self._transactions: Dict[str, TransactionRecord] = {}
        self._active: Dict[int, ActivePlan] = {}
        self._lock = threading.RLock()

    def register(self, proposal: IntentProposal) -> TransactionRecord:
        with self._lock:
            if proposal.transaction_id in self._transactions:
                raise ValueError(f"duplicate transaction_id:{proposal.transaction_id}")
            record = TransactionRecord(
                proposal=proposal,
                state=TransactionState.PROPOSED,
                updated_at_s=time.time(),
            )
            self._transactions[proposal.transaction_id] = record
            self._log(record, payload={"proposal": proposal.model_dump(mode="json")})
            return record

    def get(self, transaction_id: str) -> TransactionRecord:
        with self._lock:
            try:
                return self._transactions[transaction_id]
            except KeyError as exc:
                raise KeyError(f"unknown transaction_id:{transaction_id}") from exc

    def transition(
        self,
        transaction_id: str,
        new_state: TransactionState,
        payload: Optional[dict] = None,
    ) -> TransactionRecord:
        with self._lock:
            record = self.get(transaction_id)
            if new_state not in ALLOWED_TRANSITIONS[record.state]:
                raise ValueError(f"invalid transition:{record.state.value}->{new_state.value}")
            record.state = new_state
            record.updated_at_s = time.time()
            self._log(record, payload=payload or {})
            return record

    def set_decision(self, decision: ArbitrationDecision) -> TransactionRecord:
        state = {
            "ACK": TransactionState.ACK,
            "PLAN": TransactionState.PLAN,
            "NACK": TransactionState.NACK,
        }[decision.decision.value]
        with self._lock:
            record = self.transition(
                decision.transaction_id,
                state,
                payload={
                    "decision": decision.decision.value,
                    "reason_code": decision.reason_code,
                    "validation_safe": bool(decision.validation and decision.validation.safe),
                    "candidate_count": len(decision.candidate_evaluations),
                    "selected_candidate_id": next(
                        (
                            item.candidate_id
                            for item in decision.candidate_evaluations
                            if item.selected
                        ),
                        None,
                    ),
                },
            )
            record.decision = decision
            return record

    def commit(
        self,
        transaction_id: str,
        joint_plan: JointPlan,
        validation: ValidationResult,
        now_s: Optional[float] = None,
    ) -> None:
        now = time.time() if now_s is None else now_s
        with self._lock:
            record = self.get(transaction_id)
            if record.state not in {TransactionState.ACK, TransactionState.PLAN}:
                raise ValueError(f"cannot commit transaction in state:{record.state.value}")
            if record.proposal.is_expired(now):
                raise ValueError("cannot commit expired transaction")
            if not validation.safe:
                raise ValueError("cannot commit plan without safe validation")

            staged = dict(self._active)
            for veh_id, plan in joint_plan.plans.items():
                staged[veh_id] = ActivePlan(
                    transaction_id=transaction_id,
                    plan=plan,
                    committed_at_s=now,
                    expires_at_s=now + plan.horizon_s,
                )
            self._active = staged
            self.transition(
                transaction_id,
                TransactionState.COMMITTED,
                payload={
                    "validation_safe": True,
                    "vehicle_ids": sorted(joint_plan.plans),
                    "joint_plan": joint_plan.model_dump(mode="json"),
                },
            )

    def close_nack(self, transaction_id: str) -> None:
        self.transition(transaction_id, TransactionState.CLOSED, payload={"outcome": "not_executed"})

    def active_plans(self, now_s: Optional[float] = None) -> Dict[int, VehiclePlan]:
        now = time.time() if now_s is None else now_s
        with self._lock:
            self._active = {
                veh_id: entry for veh_id, entry in self._active.items() if entry.expires_at_s > now
            }
            return {veh_id: entry.plan for veh_id, entry in self._active.items()}

    def active_transaction_ids(self, now_s: Optional[float] = None) -> Dict[int, str]:
        self.active_plans(now_s)
        with self._lock:
            return {veh_id: entry.transaction_id for veh_id, entry in self._active.items()}

    def supersede_vehicle_plan(
        self,
        veh_id: int,
        *,
        reason: str = "goal_revised",
    ) -> tuple[Optional[str], tuple[int, ...]]:
        """Atomically cancel the joint commitment containing ``veh_id``.

        A revised goal invalidates the complete joint transaction, not only one
        vehicle's row, because cooperating actions were validated together.
        The affected vehicles are returned so an executor can clear scheduled
        commands before replanning.
        """
        with self._lock:
            active = self._active.get(int(veh_id))
            if active is None:
                return None, ()
            transaction_id = active.transaction_id
            affected = tuple(
                sorted(
                    vehicle_id
                    for vehicle_id, entry in self._active.items()
                    if entry.transaction_id == transaction_id
                )
            )
            self._active = {
                vehicle_id: entry
                for vehicle_id, entry in self._active.items()
                if entry.transaction_id != transaction_id
            }
            record = self.get(transaction_id)
            if record.state in {TransactionState.COMMITTED, TransactionState.EXECUTING}:
                self.transition(
                    transaction_id,
                    TransactionState.FAILED,
                    payload={"reason": reason, "superseded_vehicle_id": int(veh_id)},
                )
                self.transition(
                    transaction_id,
                    TransactionState.CLOSED,
                    payload={"outcome": "superseded", "reason": reason},
                )
            return transaction_id, affected

    def mark_executing(self, transaction_id: str) -> None:
        self.transition(transaction_id, TransactionState.EXECUTING)

    def mark_outcome(
        self,
        transaction_id: str,
        success: bool,
        payload: Optional[dict] = None,
    ) -> None:
        terminal = TransactionState.COMPLETED if success else TransactionState.FAILED
        with self._lock:
            self.transition(transaction_id, terminal, payload=payload or {})
            self.transition(
                transaction_id,
                TransactionState.CLOSED,
                payload={"success": success},
            )
            self._active = {
                veh_id: entry
                for veh_id, entry in self._active.items()
                if entry.transaction_id != transaction_id
            }

    def _log(self, record: TransactionRecord, payload: dict) -> None:
        vehicle_ids = [record.proposal.ego_veh_id]
        if record.proposal.request:
            vehicle_ids.extend(record.proposal.request.to_vehicle_ids)
        self.audit.append(
            AuditEvent(
                transaction_id=record.proposal.transaction_id,
                state=record.state,
                vehicle_ids=sorted(set(vehicle_ids)),
                payload=payload,
            )
        )
