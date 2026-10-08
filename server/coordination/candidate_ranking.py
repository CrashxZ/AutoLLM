"""Validator-constrained joint-plan generation and local candidate ranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .conflict_graph import (
    predict_plan,
    robust_center_separation_m,
    target_lane_for,
    target_speed_for,
)
from .models import (
    Action,
    CandidateEvaluation,
    Conflict,
    IntentProposal,
    JointPlan,
    PlanStep,
    VehiclePlan,
    VehicleState,
)
from .proposer import SemanticProposer
from .validator import DeterministicPlanValidator


FLOW_SPEED_KMH = 50.0
FEATURE_NAMES: Tuple[str, ...] = (
    "fleet_count",
    "participant_count",
    "conflict_count",
    "request_target_count",
    "emergency_count",
    "ego_speed_kmh",
    "ego_flow_error_kmh",
    "lane_change_count",
    "hold_count",
    "yield_count",
    "accelerate_count",
    "ego_speed_loss_kmh",
    "non_ego_speed_loss_kmh",
    "max_speed_loss_kmh",
    "total_flow_error_kmh",
    "initial_min_separation_m",
    "initial_min_ttc_s",
    "final_min_separation_m",
    "final_min_ttc_s",
    "robust_clearance_margin_m",
    "mean_gap_gain_m",
    "min_gap_gain_m",
    "request_satisfaction",
    "goal_progress",
    "validator_candidate_pairs",
    "validator_samples",
)


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_id: str
    template: str
    joint_plan: JointPlan


class CandidatePlanGenerator:
    """Generate a deterministic, bounded plan set from typed state."""

    def __init__(
        self,
        max_candidates: int = 16,
        flow_speed_kmh: float = FLOW_SPEED_KMH,
        enable_liveness_preparation: bool = False,
        enable_global_conflict_recovery: bool = False,
    ) -> None:
        self.max_candidates = max_candidates
        self.flow_speed_kmh = flow_speed_kmh
        self.enable_liveness_preparation = enable_liveness_preparation
        self.enable_global_conflict_recovery = enable_global_conflict_recovery

    def generate(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: Sequence[Conflict],
        active_plans: Optional[Dict[int, VehiclePlan]] = None,
    ) -> List[GeneratedCandidate]:
        active = active_plans or {}
        protected_goal_ids = {
            int(veh_id) for veh_id in proposal.protected_goal_vehicle_ids
        } - {proposal.ego_veh_id}

        def candidate_plan(state: VehicleState, action: Action) -> VehiclePlan:
            return self._plan(state, action, active.get(state.veh_id))

        def protect_goal(state: VehicleState, action: Action) -> Action:
            """Keep optional cooperation from delaying another pending goal."""
            if (
                state.veh_id in protected_goal_ids
                and action == Action.ACCELERATE
            ):
                return Action.HOLD
            return action

        ego = states[proposal.ego_veh_id]
        adjacency: Dict[int, set[int]] = {}
        for conflict in conflicts:
            adjacency.setdefault(conflict.veh_a, set()).add(conflict.veh_b)
            adjacency.setdefault(conflict.veh_b, set()).add(conflict.veh_a)
        connected = {proposal.ego_veh_id}
        frontier = [proposal.ego_veh_id]
        while frontier:
            veh_id = frontier.pop()
            for neighbor in adjacency.get(veh_id, set()):
                if neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)
        conflict_ids = sorted(connected - {proposal.ego_veh_id})
        request_ids = list(proposal.request.to_vehicle_ids) if proposal.request else []
        involved = sorted(set(conflict_ids + request_ids))
        raw: List[Tuple[str, Dict[int, VehiclePlan]]] = [
            ("original", {ego.veh_id: proposal.plan}),
            ("ego_hold", {ego.veh_id: candidate_plan(ego, Action.HOLD)}),
            ("ego_yield", {ego.veh_id: candidate_plan(ego, Action.YIELD)}),
        ]
        if self.enable_liveness_preparation:
            raw.append(
                (
                    "ego_accelerate",
                    {ego.veh_id: candidate_plan(ego, Action.ACCELERATE)},
                )
            )

        if self.enable_global_conflict_recovery and conflicts:
            # Full-snapshot validation can detect another conflict component
            # that a requester-local revision cannot repair. Include a bounded
            # candidate that restores flow for completed/non-protected agents
            # while holding other pending goal owners at their observed speed.
            # Active lateral commitments are retained by candidate_plan().
            global_ids = {
                veh_id
                for conflict in conflicts
                for veh_id in (conflict.veh_a, conflict.veh_b)
            }
            recovery = {ego.veh_id: proposal.plan}
            for veh_id in sorted(global_ids - {ego.veh_id}):
                if veh_id in states:
                    recovery[veh_id] = candidate_plan(
                        states[veh_id],
                        Action.HOLD if veh_id in protected_goal_ids else Action.ACCELERATE,
                    )
            raw.insert(1, ("global_flow_recovery", recovery))

        if request_ids:
            plans = {ego.veh_id: candidate_plan(ego, Action.HOLD)}
            requested = proposal.request.requested_action
            if requested not in {Action.YIELD, Action.HOLD, Action.KEEP_LANE, Action.ACCELERATE}:
                requested = Action.YIELD
            for veh_id in request_ids:
                if veh_id in states:
                    state = states[veh_id]
                    plans[veh_id] = candidate_plan(
                        state,
                        protect_goal(state, requested),
                    )
            raw.append(("requested_cooperation", plans))

        if involved:
            open_gap = {ego.veh_id: candidate_plan(ego, Action.HOLD)}
            all_yield = {ego.veh_id: candidate_plan(ego, Action.HOLD)}
            guarded = {ego.veh_id: proposal.plan}
            ego_s = ego.longitudinal_position_m()
            for veh_id in involved:
                state = states.get(veh_id)
                if state is None:
                    continue
                gap_action = (
                    Action.ACCELERATE
                    if state.longitudinal_position_m() >= ego_s
                    else Action.YIELD
                )
                gap_action = protect_goal(state, gap_action)
                open_gap[veh_id] = candidate_plan(state, gap_action)
                all_yield[veh_id] = candidate_plan(state, Action.YIELD)
                guarded[veh_id] = candidate_plan(state, gap_action)
            raw.extend(
                [
                    ("open_gap", open_gap),
                    ("all_conflicts_yield", all_yield),
                    ("guarded_original", guarded),
                ]
            )

        proposed_target_lane = target_lane_for(proposal.plan, ego.lane_id)
        if (
            self.enable_liveness_preparation
            and proposed_target_lane != ego.lane_id
        ):
            target_lane_states = [
                state
                for state in states.values()
                if state.veh_id != ego.veh_id
                and state.lane_id == proposed_target_lane
            ]
            if target_lane_states:
                ego_s = ego.longitudinal_position_m()
                target_lane_gap = {
                    ego.veh_id: self._plan(ego, Action.HOLD),
                }
                for state in target_lane_states:
                    gap_action = (
                        Action.ACCELERATE
                        if state.longitudinal_position_m() >= ego_s
                        else Action.YIELD
                    )
                    # This template is selected only when immediate goal
                    # progress is validator-inadmissible. Acceleration here is
                    # required cooperation, not optional goal displacement;
                    # the other vehicle's lane commitment remains unchanged.
                    target_lane_gap[state.veh_id] = candidate_plan(
                        state,
                        gap_action,
                    )
                raw.append(("target_lane_open_gap", target_lane_gap))

        output: List[GeneratedCandidate] = []
        seen = set()
        for template, plans in raw:
            canonical = json.dumps(
                {
                    str(veh_id): [
                        {
                            "action": step.action.value,
                            "lane": step.target_lane_id,
                            "speed": step.target_speed_kmh,
                            "duration": step.duration_s,
                        }
                        for step in plan.steps
                    ]
                    for veh_id, plan in sorted(plans.items())
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical in seen and template != "target_lane_open_gap":
                continue
            seen.add(canonical)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
            output.append(
                GeneratedCandidate(
                    candidate_id=f"{template}-{digest}",
                    template=template,
                    joint_plan=JointPlan(
                        transaction_id=proposal.transaction_id,
                        plans=plans,
                        summary=f"Constrained candidate: {template}",
                        proposer="candidate-generator-v1",
                    ),
                )
            )
            if len(output) >= self.max_candidates:
                break
        return output

    def _plan(
        self,
        state: VehicleState,
        action: Action,
        active_plan: Optional[VehiclePlan] = None,
    ) -> VehiclePlan:
        speed_kmh = state.speed_mps * 3.6
        target_lane_id = state.lane_id
        if active_plan is not None:
            target_lane_id = target_lane_for(active_plan, state.lane_id)
        if action == Action.YIELD:
            target_speed = max(0.0, speed_kmh - 15.0)
        elif action == Action.ACCELERATE:
            target_speed = min(self.flow_speed_kmh + 10.0, max(self.flow_speed_kmh, speed_kmh + 10.0))
        else:
            target_speed = speed_kmh
        return VehiclePlan(
            veh_id=state.veh_id,
            summary=f"{action.value} candidate",
            horizon_s=3.0,
            steps=[
                PlanStep(
                    action=action,
                    target_lane_id=target_lane_id,
                    target_speed_kmh=target_speed,
                    duration_s=3.0,
                    description="Validator-constrained candidate action",
                )
            ],
        )


class CandidateFeatureExtractor:
    def __init__(self, validator: DeterministicPlanValidator, flow_speed_kmh: float = FLOW_SPEED_KMH) -> None:
        self.validator = validator
        self.flow_speed_kmh = flow_speed_kmh

    def evaluate(
        self,
        candidate: GeneratedCandidate,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: Sequence[Conflict],
        active_plans: Dict[int, VehiclePlan],
        now_s: float,
    ) -> CandidateEvaluation:
        validation = self.validator.validate(candidate.joint_plan, states, active_plans, now_s)
        features = self._features(candidate, proposal, states, conflicts, validation)
        expert_cost = self.expert_cost(features) if validation.safe else None
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            template=candidate.template,
            valid=validation.safe,
            expert_cost=expert_cost,
            features=features,
            validation=validation,
        )

    @staticmethod
    def expert_cost(features: Dict[str, float]) -> float:
        goal_delay = 1.0 - features["goal_progress"]
        clearance_penalty = max(0.0, 2.0 - features["robust_clearance_margin_m"])
        ttc_penalty = max(0.0, 8.0 - features["final_min_ttc_s"])
        request_penalty = (
            1.0 - features["request_satisfaction"]
            if features["request_target_count"] > 0
            else 0.0
        )
        return float(
            3.0 * goal_delay
            + 0.06 * features["ego_speed_loss_kmh"]
            + 0.04 * features["non_ego_speed_loss_kmh"]
            + 0.08 * features["max_speed_loss_kmh"]
            + 0.015 * features["total_flow_error_kmh"]
            + 0.12 * max(0.0, features["participant_count"] - 1.0)
            + 1.25 * request_penalty
            + 0.35 * clearance_penalty
            + 0.08 * ttc_penalty
            - 0.18 * max(0.0, features["mean_gap_gain_m"])
        )

    def _features(
        self,
        candidate: GeneratedCandidate,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: Sequence[Conflict],
        validation,
    ) -> Dict[str, float]:
        plans = candidate.joint_plan.plans
        ego = states[proposal.ego_veh_id]
        actions = [step.action for plan in plans.values() for step in plan.steps]
        conflict_pairs = sorted({tuple(sorted((c.veh_a, c.veh_b))) for c in conflicts})
        if not conflict_pairs:
            conflict_pairs = [
                (proposal.ego_veh_id, veh_id)
                for veh_id in sorted(states)
                if veh_id != proposal.ego_veh_id
            ]

        predictions = {}
        speed_losses = {}
        flow_errors = []
        for veh_id, state in states.items():
            plan = plans.get(veh_id) or VehiclePlan(
                veh_id=veh_id,
                summary="implicit keep lane",
                horizon_s=self.validator.config.horizon_s,
                steps=[
                    PlanStep(
                        action=Action.KEEP_LANE,
                        target_lane_id=state.lane_id,
                        target_speed_kmh=(
                            state.desired_speed_mps
                            if state.desired_speed_mps is not None
                            else state.speed_mps
                        )
                        * 3.6,
                        duration_s=self.validator.config.horizon_s,
                    )
                ],
            )
            predictions[veh_id] = predict_plan(state, plan, self.validator.config)
            target_kmh = target_speed_for(plan, state.speed_mps) * 3.6
            current_kmh = state.speed_mps * 3.6
            speed_losses[veh_id] = max(0.0, current_kmh - target_kmh)
            flow_errors.append(abs(self.flow_speed_kmh - target_kmh))

        initial_separations = []
        final_separations = []
        initial_ttcs = []
        final_ttcs = []
        margins = []
        gains = []
        for veh_a, veh_b in conflict_pairs:
            if veh_a not in states or veh_b not in states:
                continue
            state_a, state_b = states[veh_a], states[veh_b]
            pred_a, pred_b = predictions[veh_a], predictions[veh_b]
            synchronized = list(zip(pred_a, pred_b))
            if not synchronized:
                continue
            initial_a, initial_b = synchronized[0]
            final_a, final_b = synchronized[-1]
            initial = abs(initial_a.s_m - initial_b.s_m)
            final = abs(final_a.s_m - final_b.s_m)
            threshold = robust_center_separation_m(state_a, state_b, self.validator.config)
            initial_separations.append(initial)
            final_separations.append(final)
            margins.append(final - threshold)
            gains.append(final - initial)
            initial_ttcs.append(self._ttc(initial_a, initial_b, state_a, state_b))
            final_ttcs.append(self._ttc(final_a, final_b, state_a, state_b))

        requested_ids = proposal.request.to_vehicle_ids if proposal.request else []
        satisfied = 0
        for veh_id in requested_ids:
            plan = plans.get(veh_id)
            if plan and plan.steps and plan.steps[0].action == proposal.request.requested_action:
                satisfied += 1
        request_satisfaction = satisfied / len(requested_ids) if requested_ids else 1.0

        ego_plan = plans.get(proposal.ego_veh_id)
        goal_progress = 0.0
        if ego_plan:
            target_lane = ego.lane_id
            for step in ego_plan.steps:
                if step.action not in {Action.LANE_LEFT, Action.LANE_RIGHT}:
                    continue
                if step.target_lane_id is not None:
                    target_lane = step.target_lane_id
                elif step.action == Action.LANE_LEFT:
                    target_lane += 1
                else:
                    target_lane -= 1
            goal_lane = proposal.goal.target_lane_id
            if goal_lane is not None:
                before = abs(goal_lane - ego.lane_id)
                after = abs(goal_lane - target_lane)
                goal_progress = max(-1.0, min(1.0, float(before - after)))

        finite_initial_ttc = [value for value in initial_ttcs if np.isfinite(value)]
        finite_final_ttc = [value for value in final_ttcs if np.isfinite(value)]
        non_ego_losses = [loss for veh_id, loss in speed_losses.items() if veh_id != proposal.ego_veh_id]
        values = {
            "fleet_count": len(states),
            "participant_count": len(plans),
            "conflict_count": len(conflicts),
            "request_target_count": len(requested_ids),
            "emergency_count": sum(state.vehicle_class.lower() == "emergency" for state in states.values()),
            "ego_speed_kmh": ego.speed_mps * 3.6,
            "ego_flow_error_kmh": abs(self.flow_speed_kmh - ego.speed_mps * 3.6),
            "lane_change_count": sum(action in {Action.LANE_LEFT, Action.LANE_RIGHT} for action in actions),
            "hold_count": sum(action in {Action.HOLD, Action.KEEP_LANE} for action in actions),
            "yield_count": sum(action == Action.YIELD for action in actions),
            "accelerate_count": sum(action == Action.ACCELERATE for action in actions),
            "ego_speed_loss_kmh": speed_losses.get(proposal.ego_veh_id, 0.0),
            "non_ego_speed_loss_kmh": sum(non_ego_losses),
            "max_speed_loss_kmh": max(speed_losses.values(), default=0.0),
            "total_flow_error_kmh": sum(flow_errors),
            "initial_min_separation_m": min(initial_separations, default=200.0),
            "initial_min_ttc_s": min(finite_initial_ttc, default=20.0),
            "final_min_separation_m": min(final_separations, default=200.0),
            "final_min_ttc_s": min(finite_final_ttc, default=20.0),
            "robust_clearance_margin_m": min(margins, default=100.0),
            "mean_gap_gain_m": float(np.mean(gains)) if gains else 0.0,
            "min_gap_gain_m": min(gains, default=0.0),
            "request_satisfaction": request_satisfaction,
            "goal_progress": goal_progress,
            "validator_candidate_pairs": validation.candidate_pair_count,
            "validator_samples": validation.sample_count,
        }
        return {name: float(values[name]) for name in FEATURE_NAMES}

    @staticmethod
    def _ttc(point_a, point_b, state_a: VehicleState, state_b: VehicleState) -> float:
        if point_a.s_m <= point_b.s_m:
            closing = point_a.speed_mps - point_b.speed_mps
        else:
            closing = point_b.speed_mps - point_a.speed_mps
        if closing <= 0.05 or not point_a.occupied_lanes.intersection(point_b.occupied_lanes):
            return float("inf")
        clearance = max(0.0, abs(point_a.s_m - point_b.s_m) - 0.5 * (state_a.length_m + state_b.length_m))
        return clearance / closing


class NumpyMLPRanker:
    """Portable inference for a StandardScaler + sklearn MLPRegressor export."""

    def __init__(self, model_path: str) -> None:
        data = np.load(model_path, allow_pickle=False)
        feature_names = tuple(str(value) for value in data["feature_names"].tolist())
        if feature_names != FEATURE_NAMES:
            raise ValueError("candidate ranker feature schema mismatch")
        self.mean = data["mean"].astype(np.float64)
        self.scale = data["scale"].astype(np.float64)
        self.layer_count = int(data["layer_count"][0])
        self.coefs = [data[f"coef_{index}"].astype(np.float64) for index in range(self.layer_count)]
        self.intercepts = [
            data[f"intercept_{index}"].astype(np.float64) for index in range(self.layer_count)
        ]

    def predict(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        matrix = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
        hidden = (matrix - self.mean) / np.where(self.scale == 0.0, 1.0, self.scale)
        for index, (coef, intercept) in enumerate(zip(self.coefs, self.intercepts)):
            hidden = hidden @ coef + intercept
            if index < self.layer_count - 1:
                hidden = np.maximum(hidden, 0.0)
        return hidden.reshape(-1)


class RankedCandidateProposer(SemanticProposer):
    """Generate, validate, and rank a finite candidate set locally."""

    name = "constrained-deterministic-ranker"

    def __init__(
        self,
        validator: Optional[DeterministicPlanValidator] = None,
        model_path: Optional[str] = None,
        enable_liveness_preparation: bool = False,
        enable_global_conflict_recovery: bool = False,
    ) -> None:
        self.validator = validator or DeterministicPlanValidator()
        self.enable_liveness_preparation = enable_liveness_preparation
        self.generator = CandidatePlanGenerator(
            enable_liveness_preparation=enable_liveness_preparation,
            enable_global_conflict_recovery=enable_global_conflict_recovery,
        )
        self.extractor = CandidateFeatureExtractor(self.validator)
        self.model_path = model_path
        self.model: Optional[NumpyMLPRanker] = None
        self.model_error: Optional[str] = None
        if model_path:
            try:
                self.model = NumpyMLPRanker(model_path)
                self.name = "constrained-learned-ranker"
            except Exception as exc:
                self.model_error = type(exc).__name__
        self.last_evaluations: List[CandidateEvaluation] = []
        self.last_selected_candidate_id: Optional[str] = None

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        self.last_evaluations = []
        self.last_selected_candidate_id = None
        now_s = max((state.observed_at_s for state in states.values()), default=proposal.created_at_s)
        candidates = self.generator.generate(
            proposal,
            states,
            conflicts,
            active_plans=active_plans,
        )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        evaluations = [
            self.extractor.evaluate(candidate, proposal, states, conflicts, active_plans, now_s)
            for candidate in candidates
        ]
        admissible = [evaluation for evaluation in evaluations if evaluation.valid]
        if not admissible:
            self.last_evaluations = evaluations
            return None

        if self.model is not None:
            predictions = self.model.predict([evaluation.features for evaluation in admissible])
            for evaluation, prediction in zip(admissible, predictions):
                evaluation.predicted_cost = float(prediction)
            ordered = sorted(admissible, key=lambda item: (item.predicted_cost, item.candidate_id))
        else:
            ordered = sorted(admissible, key=lambda item: (item.expert_cost, item.candidate_id))
        has_immediate_progress = any(
            evaluation.features["goal_progress"] > 0.0
            for evaluation in admissible
        )
        if has_immediate_progress:
            ordered = sorted(
                ordered,
                key=lambda item: (
                    item.features["goal_progress"] <= 0.0,
                    item.predicted_cost
                    if item.predicted_cost is not None
                    else item.expert_cost,
                    item.candidate_id,
                ),
            )
        elif self.enable_liveness_preparation:
            ordered = sorted(
                ordered,
                key=lambda item: (
                    item.template != "target_lane_open_gap",
                    item.predicted_cost
                    if item.predicted_cost is not None
                    else item.expert_cost,
                    item.candidate_id,
                ),
            )
        for rank, evaluation in enumerate(ordered, start=1):
            evaluation.rank = rank
        selected = ordered[0]
        selected.selected = True
        self.last_selected_candidate_id = selected.candidate_id
        self.last_evaluations = evaluations
        plan = by_id[selected.candidate_id].joint_plan.model_copy(deep=True)
        plan.proposer = self.name
        return plan
