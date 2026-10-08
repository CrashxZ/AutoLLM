"""Cheap candidate prioritization before deterministic trajectory validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .candidate_ranking import (
    FLOW_SPEED_KMH,
    CandidateFeatureExtractor,
    CandidatePlanGenerator,
    GeneratedCandidate,
)
from .models import (
    Action,
    CandidateEvaluation,
    Conflict,
    IntentProposal,
    JointPlan,
    VehiclePlan,
    VehicleState,
)
from .proposer import SemanticProposer
from .validator import DeterministicPlanValidator


TEMPLATE_NAMES = (
    "original",
    "ego_hold",
    "ego_yield",
    "ego_accelerate",
    "requested_cooperation",
    "open_gap",
    "all_conflicts_yield",
    "guarded_original",
    "target_lane_open_gap",
)

CHEAP_FEATURE_NAMES = (
    "fleet_count",
    "participant_count",
    "initial_conflict_count",
    "request_target_count",
    "emergency_count",
    "ego_speed_kmh",
    "ego_flow_error_kmh",
    "candidate_step_count",
    "lane_change_count",
    "hold_count",
    "yield_count",
    "accelerate_count",
    "ego_speed_loss_kmh",
    "non_ego_speed_loss_kmh",
    "max_speed_loss_kmh",
    "goal_progress",
    "request_satisfaction",
    "target_lane_vehicle_count",
    "target_lane_min_abs_gap_m",
    "target_lane_min_ahead_gap_m",
    "target_lane_min_rear_gap_m",
    "max_localization_error_m",
    *(f"template_{name}" for name in TEMPLATE_NAMES),
)


def candidate_priority_group(features: Dict[str, float]) -> int:
    """Apply the same hard progress/liveness precedence as production ranking."""
    if features["goal_progress"] > 0.0:
        return 0
    if features["template_target_lane_open_gap"] > 0.5:
        return 1
    return 2


class CheapCandidateFeatureExtractor:
    """Extract state/plan features without trajectory prediction or validation."""

    def __init__(self, flow_speed_kmh: float = FLOW_SPEED_KMH) -> None:
        self.flow_speed_kmh = flow_speed_kmh

    def extract(
        self,
        candidate: GeneratedCandidate,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: Sequence[Conflict],
    ) -> Dict[str, float]:
        ego = states[proposal.ego_veh_id]
        plans = candidate.joint_plan.plans
        actions = [step.action for plan in plans.values() for step in plan.steps]
        speed_losses: Dict[int, float] = {}
        for veh_id, plan in plans.items():
            state = states[veh_id]
            current_kmh = state.speed_mps * 3.6
            target_kmh = current_kmh
            for step in plan.steps:
                if step.target_speed_kmh is not None:
                    target_kmh = float(step.target_speed_kmh)
            speed_losses[veh_id] = max(0.0, current_kmh - target_kmh)

        ego_plan = plans.get(proposal.ego_veh_id)
        goal_progress = 0.0
        if ego_plan and proposal.goal.target_lane_id is not None:
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
            before = abs(proposal.goal.target_lane_id - ego.lane_id)
            after = abs(proposal.goal.target_lane_id - target_lane)
            goal_progress = max(-1.0, min(1.0, float(before - after)))

        requested_ids = proposal.request.to_vehicle_ids if proposal.request else []
        requested_action = (
            proposal.request.requested_action if proposal.request else None
        )
        satisfied = sum(
            bool(
                plans.get(veh_id)
                and plans[veh_id].steps
                and plans[veh_id].steps[0].action == requested_action
            )
            for veh_id in requested_ids
        )
        request_satisfaction = (
            satisfied / len(requested_ids) if requested_ids else 1.0
        )

        target_lane = proposal.goal.target_lane_id
        target_states = [
            state
            for state in states.values()
            if state.veh_id != ego.veh_id
            and target_lane is not None
            and state.lane_id == target_lane
        ]
        ego_s = ego.longitudinal_position_m()
        signed_gaps = [
            state.longitudinal_position_m() - ego_s for state in target_states
        ]
        ahead = [gap for gap in signed_gaps if gap >= 0.0]
        rear = [-gap for gap in signed_gaps if gap < 0.0]
        non_ego_losses = [
            loss for veh_id, loss in speed_losses.items() if veh_id != ego.veh_id
        ]
        values = {
            "fleet_count": len(states),
            "participant_count": len(plans),
            "initial_conflict_count": len(conflicts),
            "request_target_count": len(requested_ids),
            "emergency_count": sum(
                state.vehicle_class.lower() == "emergency"
                for state in states.values()
            ),
            "ego_speed_kmh": ego.speed_mps * 3.6,
            "ego_flow_error_kmh": abs(
                self.flow_speed_kmh - ego.speed_mps * 3.6
            ),
            "candidate_step_count": sum(len(plan.steps) for plan in plans.values()),
            "lane_change_count": sum(
                action in {Action.LANE_LEFT, Action.LANE_RIGHT}
                for action in actions
            ),
            "hold_count": sum(
                action in {Action.HOLD, Action.KEEP_LANE} for action in actions
            ),
            "yield_count": sum(action == Action.YIELD for action in actions),
            "accelerate_count": sum(
                action == Action.ACCELERATE for action in actions
            ),
            "ego_speed_loss_kmh": speed_losses.get(ego.veh_id, 0.0),
            "non_ego_speed_loss_kmh": sum(non_ego_losses),
            "max_speed_loss_kmh": max(speed_losses.values(), default=0.0),
            "goal_progress": goal_progress,
            "request_satisfaction": request_satisfaction,
            "target_lane_vehicle_count": len(target_states),
            "target_lane_min_abs_gap_m": min(
                (abs(gap) for gap in signed_gaps), default=500.0
            ),
            "target_lane_min_ahead_gap_m": min(ahead, default=500.0),
            "target_lane_min_rear_gap_m": min(rear, default=500.0),
            "max_localization_error_m": max(
                (state.localization_error_m for state in states.values()),
                default=0.0,
            ),
        }
        for template in TEMPLATE_NAMES:
            values[f"template_{template}"] = float(candidate.template == template)
        return {name: float(values[name]) for name in CHEAP_FEATURE_NAMES}


class NumpyCandidatePrioritizer:
    """Portable StandardScaler + MLP regressor for cheap candidate features."""

    def __init__(self, model_path: str) -> None:
        data = np.load(model_path, allow_pickle=False)
        feature_names = tuple(str(value) for value in data["feature_names"].tolist())
        if feature_names != CHEAP_FEATURE_NAMES:
            raise ValueError("candidate prioritizer feature schema mismatch")
        self.mean = data["mean"].astype(np.float64)
        self.scale = data["scale"].astype(np.float64)
        self.layer_count = int(data["layer_count"][0])
        self.coefs = [
            data[f"coef_{index}"].astype(np.float64)
            for index in range(self.layer_count)
        ]
        self.intercepts = [
            data[f"intercept_{index}"].astype(np.float64)
            for index in range(self.layer_count)
        ]

    def predict(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        matrix = np.asarray(
            [[row[name] for name in CHEAP_FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )
        hidden = (matrix - self.mean) / np.where(
            self.scale == 0.0, 1.0, self.scale
        )
        for index, (coef, intercept) in enumerate(
            zip(self.coefs, self.intercepts)
        ):
            hidden = hidden @ coef + intercept
            if index < self.layer_count - 1:
                hidden = np.maximum(hidden, 0.0)
        return hidden.reshape(-1)


@dataclass
class PrioritizationStats:
    generated_candidates: int = 0
    validated_candidates: int = 0
    validator_total_pairs: int = 0
    validator_candidate_pairs: int = 0
    validator_samples: int = 0
    fallback_batches: int = 0
    selected_candidate_id: Optional[str] = None


class PrioritizedCandidateProposer(SemanticProposer):
    """Order candidates cheaply, then validate bounded batches sequentially."""

    name = "sequential-heuristic-prioritizer"

    def __init__(
        self,
        validator: Optional[DeterministicPlanValidator] = None,
        model_path: Optional[str] = None,
        batch_size: int = 2,
        enable_liveness_preparation: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.validator = validator or DeterministicPlanValidator()
        self.generator = CandidatePlanGenerator(
            enable_liveness_preparation=enable_liveness_preparation
        )
        self.cheap_extractor = CheapCandidateFeatureExtractor()
        self.full_extractor = CandidateFeatureExtractor(self.validator)
        self.batch_size = batch_size
        self.model_path = model_path
        self.model: Optional[NumpyCandidatePrioritizer] = None
        self.model_error: Optional[str] = None
        if model_path:
            try:
                self.model = NumpyCandidatePrioritizer(model_path)
                self.name = "sequential-learned-prioritizer"
            except Exception as exc:
                self.model_error = type(exc).__name__
        self.last_evaluations: List[CandidateEvaluation] = []
        self.last_stats = PrioritizationStats()
        self.last_order: List[str] = []

    @staticmethod
    def heuristic_score(candidate: GeneratedCandidate) -> float:
        order = {
            "original": 0.0,
            "target_lane_open_gap": 0.5,
            "requested_cooperation": 1.0,
            "open_gap": 2.0,
            "ego_hold": 3.0,
            "ego_accelerate": 4.0,
            "ego_yield": 5.0,
            "all_conflicts_yield": 6.0,
            "guarded_original": 7.0,
        }
        return order.get(candidate.template, 10.0)

    def order_candidates(
        self,
        candidates: Sequence[GeneratedCandidate],
        features: Sequence[Dict[str, float]],
    ) -> List[GeneratedCandidate]:
        if self.model is not None:
            scores = self.model.predict(features)
        else:
            scores = np.asarray(
                [self.heuristic_score(candidate) for candidate in candidates],
                dtype=np.float64,
            )
        indexed = list(zip(candidates, features, scores))
        indexed.sort(
            key=lambda item: (
                candidate_priority_group(item[1]),
                float(item[2]),
                item[0].candidate_id,
            )
        )
        return [item[0] for item in indexed]

    async def propose(
        self,
        proposal: IntentProposal,
        states: Dict[int, VehicleState],
        conflicts: List[Conflict],
        active_plans: Dict[int, VehiclePlan],
    ) -> Optional[JointPlan]:
        self.last_evaluations = []
        self.last_stats = PrioritizationStats()
        self.last_order = []
        now_s = max(
            (state.observed_at_s for state in states.values()),
            default=proposal.created_at_s,
        )
        candidates = self.generator.generate(
            proposal,
            states,
            conflicts,
            active_plans=active_plans,
        )
        features = [
            self.cheap_extractor.extract(candidate, proposal, states, conflicts)
            for candidate in candidates
        ]
        ordered = self.order_candidates(candidates, features)
        features_by_id = {
            candidate.candidate_id: candidate_features
            for candidate, candidate_features in zip(candidates, features)
        }
        self.last_order = [candidate.candidate_id for candidate in ordered]
        self.last_stats.generated_candidates = len(ordered)

        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start : start + self.batch_size]
            evaluations = [
                self.full_extractor.evaluate(
                    candidate,
                    proposal,
                    states,
                    conflicts,
                    active_plans,
                    now_s,
                )
                for candidate in batch
            ]
            self.last_evaluations.extend(evaluations)
            self.last_stats.validated_candidates += len(evaluations)
            for evaluation in evaluations:
                validation = evaluation.validation
                self.last_stats.validator_total_pairs += validation.total_pair_count
                self.last_stats.validator_candidate_pairs += (
                    validation.candidate_pair_count
                )
                self.last_stats.validator_samples += validation.sample_count
            admissible = [evaluation for evaluation in evaluations if evaluation.valid]
            if admissible:
                earliest_group = min(
                    candidate_priority_group(features_by_id[evaluation.candidate_id])
                    for evaluation in admissible
                )
                admissible = [
                    evaluation
                    for evaluation in admissible
                    if candidate_priority_group(
                        features_by_id[evaluation.candidate_id]
                    )
                    == earliest_group
                ]
                selected = min(
                    admissible,
                    key=lambda item: (float(item.expert_cost), item.candidate_id),
                )
                selected.selected = True
                selected.rank = 1
                self.last_stats.selected_candidate_id = selected.candidate_id
                by_id = {
                    candidate.candidate_id: candidate for candidate in candidates
                }
                plan = by_id[selected.candidate_id].joint_plan.model_copy(deep=True)
                plan.proposer = self.name
                return plan
            if start == 0 and len(ordered) > self.batch_size:
                self.last_stats.fallback_batches += 1
        return None
