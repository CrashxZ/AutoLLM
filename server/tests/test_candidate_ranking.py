import asyncio
import time

import pytest

from server.coordination.candidate_ranking import (
    CandidateFeatureExtractor,
    CandidatePlanGenerator,
    RankedCandidateProposer,
)
from server.coordination.models import (
    Action,
    CooperationRequest,
    Conflict,
    DecisionKind,
    Goal,
    GoalKind,
    IntentProposal,
    JointPlan,
    PlanStep,
    VehiclePlan,
    VehicleState,
)
from server.coordination.orchestrator import CoordinationOrchestrator
from server.coordination.validator import DeterministicPlanValidator


def make_state(veh_id: int, lane_id: int, s_m: float, speed_kmh: float = 40.0):
    return VehicleState(
        veh_id=veh_id,
        observed_at_s=time.time(),
        x_m=s_m,
        y_m=0.0,
        s_m=s_m,
        speed_mps=speed_kmh / 3.6,
        lane_id=lane_id,
    )


def make_proposal(request: bool = False):
    now = time.time()
    cooperation = (
        CooperationRequest(
            to_vehicle_ids=[1001],
            requested_action=Action.YIELD,
            expires_at_s=now + 5.0,
        )
        if request
        else None
    )
    return IntentProposal(
        ego_veh_id=1000,
        created_at_s=now,
        expires_at_s=now + 5.0,
        observation_ts_s=now,
        goal=Goal(kind=GoalKind.TARGET_LANE, target_lane_id=-2),
        plan=VehiclePlan(
            veh_id=1000,
            summary="change right",
            horizon_s=3.0,
            steps=[
                PlanStep(
                    action=Action.LANE_RIGHT,
                    target_lane_id=-2,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        ),
        request=cooperation,
    )


def test_positive_lane_gap_preparation_accelerates_vehicle_ahead() -> None:
    now = time.time()
    proposal = IntentProposal(
        ego_veh_id=1000,
        created_at_s=now,
        expires_at_s=now + 5.0,
        observation_ts_s=now,
        goal=Goal(kind=GoalKind.TARGET_LANE, target_lane_id=5),
        plan=VehiclePlan(
            veh_id=1000,
            summary="change right",
            horizon_s=3.0,
            steps=[
                PlanStep(
                    action=Action.LANE_RIGHT,
                    target_lane_id=5,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        ),
    )
    ego = make_state(1000, 4, 100.0)
    ahead = make_state(1001, 5, 80.0)

    assert ego.longitudinal_position_m() == -100.0
    assert ahead.longitudinal_position_m() == -80.0
    candidates = CandidatePlanGenerator(
        enable_liveness_preparation=True
    ).generate(proposal, {1000: ego, 1001: ahead}, [])
    gap = next(
        candidate
        for candidate in candidates
        if candidate.template == "target_lane_open_gap"
    )
    assert gap.joint_plan.plans[1001].steps[0].action == Action.ACCELERATE


def test_synchronized_features_do_not_invent_hold_gap_gain():
    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 0.0),
        1001: make_state(1001, -2, 0.0),
    }
    validator = DeterministicPlanValidator()
    original = JointPlan(transaction_id=proposal.transaction_id, plans={1000: proposal.plan})
    conflicts = validator.validate(original, states).conflicts
    candidate = next(
        item
        for item in CandidatePlanGenerator().generate(proposal, states, conflicts)
        if item.template == "ego_hold"
    )
    result = CandidateFeatureExtractor(validator).evaluate(
        candidate,
        proposal,
        states,
        conflicts,
        {},
        max(state.observed_at_s for state in states.values()),
    )
    assert result.features["mean_gap_gain_m"] == pytest.approx(0.0, abs=1e-6)


def test_hold_with_inherited_target_lane_does_not_count_as_goal_progress():
    proposal = make_proposal()
    states = {1000: make_state(1000, -1, 0.0)}
    active_plans = {1000: proposal.plan}
    candidate = next(
        item
        for item in CandidatePlanGenerator().generate(
            proposal,
            states,
            [],
            active_plans=active_plans,
        )
        if item.template == "ego_hold"
    )
    result = CandidateFeatureExtractor(DeterministicPlanValidator()).evaluate(
        candidate,
        proposal,
        states,
        [],
        active_plans,
        max(state.observed_at_s for state in states.values()),
    )

    assert candidate.joint_plan.plans[1000].steps[0].action is Action.HOLD
    assert candidate.joint_plan.plans[1000].steps[0].target_lane_id == -2
    assert result.features["goal_progress"] == 0.0


def test_validator_admissible_goal_progress_has_priority_over_learned_hold():
    class HoldBiasedModel:
        @staticmethod
        def predict(rows):
            return [
                0.0 if row["goal_progress"] == 0.0 else 100.0
                for row in rows
            ]

    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 0.0, 50.0),
        1001: make_state(1001, -2, 100.0, 50.0),
    }
    proposer = RankedCandidateProposer(enable_liveness_preparation=True)
    proposer.model = HoldBiasedModel()

    selected_plan = asyncio.run(proposer.propose(proposal, states, [], {}))
    selected = next(item for item in proposer.last_evaluations if item.selected)

    assert selected_plan is not None
    assert selected.features["goal_progress"] > 0.0
    assert selected_plan.plans[1000].steps[0].action is Action.LANE_RIGHT


def test_candidate_generation_includes_bounded_ego_acceleration():
    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 7.0, 50.0),
        1001: make_state(1001, -2, 0.0, 50.0),
    }
    validator = DeterministicPlanValidator()
    original = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={1000: proposal.plan},
    )
    conflicts = validator.validate(original, states).conflicts

    candidate = next(
        item
        for item in CandidatePlanGenerator(
            enable_liveness_preparation=True
        ).generate(
            proposal,
            states,
            conflicts,
        )
        if item.template == "ego_accelerate"
    )
    step = candidate.joint_plan.plans[1000].steps[0]

    assert step.action is Action.ACCELERATE
    assert step.target_speed_kmh == pytest.approx(60.0)
    assert validator.validate(candidate.joint_plan, states).safe


def test_target_lane_gap_candidate_splits_platoon_around_ego():
    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 0.0, 50.0),
        1001: make_state(1001, -2, -8.0, 50.0),
        1002: make_state(1002, -2, 8.0, 50.0),
        1003: make_state(1003, -3, 0.0, 50.0),
    }
    validator = DeterministicPlanValidator()
    original = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={1000: proposal.plan},
    )
    conflicts = validator.validate(original, states).conflicts

    candidate = next(
        item
        for item in CandidatePlanGenerator(
            enable_liveness_preparation=True
        ).generate(
            proposal,
            states,
            conflicts,
        )
        if item.template == "target_lane_open_gap"
    )

    assert set(candidate.joint_plan.plans) == {1000, 1001, 1002}
    assert candidate.joint_plan.plans[1000].steps[0].action is Action.HOLD
    assert candidate.joint_plan.plans[1001].steps[0].action is Action.YIELD
    assert candidate.joint_plan.plans[1002].steps[0].action is Action.ACCELERATE
    assert validator.validate(candidate.joint_plan, states).safe


def test_ranker_uses_safe_target_lane_preparation_when_progress_is_blocked():
    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 0.0, 50.0),
        1001: make_state(1001, -2, -8.0, 50.0),
        1002: make_state(1002, -2, 8.0, 50.0),
    }
    validator = DeterministicPlanValidator()
    original = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={1000: proposal.plan},
    )
    conflicts = validator.validate(original, states).conflicts
    proposer = RankedCandidateProposer(
        validator=validator,
        enable_liveness_preparation=True,
    )

    selected_plan = asyncio.run(
        proposer.propose(proposal, states, conflicts, {})
    )
    selected = [
        item for item in proposer.last_evaluations if item.selected
    ]

    assert selected_plan is not None
    assert len(selected) == 1
    assert selected[0].template == "target_lane_open_gap"
    assert selected[0].valid


def test_ranked_proposer_selects_only_valid_candidate_and_audits_set():
    proposal = make_proposal(request=True)
    states = {
        1000: make_state(1000, -1, 0.0),
        1001: make_state(1001, -2, 0.0),
    }
    result = asyncio.run(
        CoordinationOrchestrator(proposer=RankedCandidateProposer()).review(
            proposal,
            states,
        )
    )
    assert result.decision == DecisionKind.PLAN
    assert result.validation and result.validation.safe
    assert result.candidate_evaluations
    selected = [item for item in result.candidate_evaluations if item.selected]
    assert len(selected) == 1
    assert selected[0].valid
    assert any(not item.valid for item in result.candidate_evaluations)


def test_missing_learned_model_falls_back_to_deterministic_ranking():
    proposer = RankedCandidateProposer(model_path="/does/not/exist/ranker.npz")
    assert proposer.model is None
    assert proposer.model_error == "FileNotFoundError"
    assert proposer.name == "constrained-deterministic-ranker"


def test_corrupt_learned_model_falls_back_to_deterministic_ranking(tmp_path):
    corrupt = tmp_path / "ranker.npz"
    corrupt.write_bytes(b"not a numpy archive")
    proposer = RankedCandidateProposer(model_path=str(corrupt))
    assert proposer.model is None
    assert proposer.model_error is not None
    assert proposer.name == "constrained-deterministic-ranker"


def test_candidate_generation_covers_transitive_conflict_component():
    proposal = make_proposal()
    states = {
        1000: make_state(1000, -1, 0.0, 45.0),
        1001: make_state(1001, -2, 8.0, 35.0),
        1002: make_state(1002, -2, -45.0, 60.0),
    }
    validator = DeterministicPlanValidator()
    original = JointPlan(transaction_id=proposal.transaction_id, plans={1000: proposal.plan})
    conflicts = validator.validate(original, states).conflicts
    assert any({conflict.veh_a, conflict.veh_b} == {1000, 1001} for conflict in conflicts)
    assert any({conflict.veh_a, conflict.veh_b} == {1001, 1002} for conflict in conflicts)
    open_gap = next(
        item
        for item in CandidatePlanGenerator().generate(proposal, states, conflicts)
        if item.template == "open_gap"
    )
    assert set(open_gap.joint_plan.plans) == {1000, 1001, 1002}


def test_cooperative_yield_preserves_an_active_lateral_commitment():
    proposal = make_proposal(request=True)
    states = {
        1000: make_state(1000, -1, 0.0),
        # The rounded observation still reports lane -3 while this vehicle is
        # physically moving toward the committed destination lane -2.
        1001: make_state(1001, -3, -8.0),
    }
    active_plans = {
        1001: VehiclePlan(
            veh_id=1001,
            summary="active lane change",
            horizon_s=3.0,
            steps=[
                PlanStep(
                    action=Action.LANE_LEFT,
                    target_lane_id=-2,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        )
    }
    conflicts = [
        # Only graph connectivity matters to candidate generation here.
        Conflict(
            veh_a=1000,
            veh_b=1001,
            kind="predicted_separation",
            predicted_at_s=1.0,
            separation_m=6.0,
            threshold_m=10.0,
        )
    ]

    candidate = next(
        item
        for item in CandidatePlanGenerator().generate(
            proposal,
            states,
            conflicts,
            active_plans=active_plans,
        )
        if item.template == "guarded_original"
    )

    yield_step = candidate.joint_plan.plans[1001].steps[0]
    assert yield_step.action == Action.YIELD
    assert yield_step.target_lane_id == -2


def test_pending_goal_owner_is_not_used_for_optional_acceleration():
    proposal = make_proposal().model_copy(
        update={"protected_goal_vehicle_ids": [1001]}
    )
    states = {
        1000: make_state(1000, -1, 0.0, 40.0),
        1001: make_state(1001, -2, 8.0, 40.0),
    }
    conflicts = [
        Conflict(
            veh_a=1000,
            veh_b=1001,
            kind="predicted_separation",
            predicted_at_s=1.0,
            separation_m=6.0,
            threshold_m=10.0,
        )
    ]

    guarded = next(
        candidate
        for candidate in CandidatePlanGenerator().generate(
            proposal,
            states,
            conflicts,
        )
        if candidate.template == "guarded_original"
    )

    assert guarded.joint_plan.plans[1001].steps[0].action is Action.HOLD


def test_liveness_gap_can_accelerate_a_pending_goal_owner():
    proposal = make_proposal().model_copy(
        update={"protected_goal_vehicle_ids": [1001]}
    )
    states = {
        1000: make_state(1000, -1, 0.0, 40.0),
        1001: make_state(1001, -2, 8.0, 40.0),
    }
    conflicts = [
        Conflict(
            veh_a=1000,
            veh_b=1001,
            kind="predicted_separation",
            predicted_at_s=1.0,
            separation_m=6.0,
            threshold_m=10.0,
        )
    ]

    candidate = next(
        item
        for item in CandidatePlanGenerator(
            enable_liveness_preparation=True
        ).generate(proposal, states, conflicts)
        if item.template == "target_lane_open_gap"
    )

    step = candidate.joint_plan.plans[1001].steps[0]
    assert step.action is Action.ACCELERATE
    assert step.target_lane_id == -2
