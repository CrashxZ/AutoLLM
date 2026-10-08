import asyncio
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from server.coordination.audit import AuditLog
from server.coordination.legacy_adapter import payload_to_proposal
from server.coordination.models import (
    Action,
    CooperationRequest,
    DecisionKind,
    Goal,
    GoalKind,
    IntentProposal,
    JointPlan,
    PlanStep,
    TransactionState,
    ValidationResult,
    VehiclePlan,
    VehicleState,
)
from server.coordination.orchestrator import CoordinationOrchestrator
from server.coordination.proposer import (
    DeterministicYieldProposer,
    OpenAIJointPlanProposer,
    SemanticProposer,
    UnavailableProposer,
)
from server.coordination.transaction import ActivePlanStore
from server.coordination.validator import DeterministicPlanValidator, ValidationConfig


def state(veh_id, lane_id, s_m, speed_kmh=40.0, age_s=0.0, road_id=None):
    return VehicleState(
        veh_id=veh_id,
        observed_at_s=time.time() - age_s,
        x_m=s_m,
        y_m=0.0,
        yaw_deg=0.0,
        s_m=s_m,
        speed_mps=speed_kmh / 3.6,
        lane_id=lane_id,
        road_id=road_id,
    )


def lane_proposal(veh_id, current_lane, target_lane, request=None):
    action = Action.LANE_RIGHT if target_lane < current_lane else Action.LANE_LEFT
    now = time.time()
    return IntentProposal(
        ego_veh_id=veh_id,
        created_at_s=now,
        expires_at_s=now + 5.0,
        observation_ts_s=now,
        goal=Goal(kind=GoalKind.TARGET_LANE, target_lane_id=target_lane),
        plan=VehiclePlan(
            veh_id=veh_id,
            summary=f"move to lane {target_lane}",
            steps=[
                PlanStep(
                    action=action,
                    target_lane_id=target_lane,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        ),
        request=request,
    )


def test_self_request_is_invalid():
    now = time.time()
    request = CooperationRequest(
        to_vehicle_ids=[101],
        requested_action=Action.YIELD,
        expires_at_s=now + 5.0,
    )
    with pytest.raises(ValidationError, match="cannot request itself"):
        lane_proposal(101, -1, -2, request=request)


def test_legacy_payload_preserves_protected_goal_owners():
    states = {101: state(101, -1, 0.0), 102: state(102, -2, 8.0)}
    payload = SimpleNamespace(
        veh_id=101,
        intent={"ego_action": "lane_right", "target_lane_id": -2},
        context={
            "goal": "reach lane -2",
            "protected_goal_vehicle_ids": [101, 102],
        },
        plan={
            "steps": [
                {
                    "action": "lane_right",
                    "target_lane_id": -2,
                    "duration_s": 3.0,
                }
            ]
        },
        request=None,
        goal="reach lane -2",
        top_frame_b64=None,
    )

    proposal = payload_to_proposal(payload, states)

    assert proposal.protected_goal_vehicle_ids == [101, 102]


def test_safe_fast_path_ack_and_audit_validation_before_commit():
    audit = AuditLog()
    store = ActivePlanStore(audit)
    orchestrator = CoordinationOrchestrator(
        proposer=UnavailableProposer(),
        store=store,
    )
    proposal = lane_proposal(101, -1, -2)
    states = {
        101: state(101, -1, 0.0),
        102: state(102, -3, 1.0),
    }

    result = asyncio.run(orchestrator.review(proposal, states))

    assert result.decision == DecisionKind.ACK
    assert result.validation and result.validation.safe
    assert store.active_transaction_ids()[101] == proposal.transaction_id
    assert audit.has_validation_before_commit(proposal.transaction_id)


@pytest.mark.parametrize("success", [True, False])
def test_terminal_outcome_removes_every_joint_active_plan(success):
    store = ActivePlanStore()
    proposal = lane_proposal(111, -1, -2)
    cooperating_plan = VehiclePlan(
        veh_id=112,
        summary="cooperate",
        steps=[
            PlanStep(
                action=Action.YIELD,
                target_lane_id=-2,
                target_speed_kmh=25.0,
                duration_s=3.0,
            )
        ],
    )
    joint = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={111: proposal.plan, 112: cooperating_plan},
    )
    store.register(proposal)
    store.transition(proposal.transaction_id, TransactionState.REVIEWING)
    store.transition(proposal.transaction_id, TransactionState.ACK)
    store.commit(
        proposal.transaction_id,
        joint,
        ValidationResult(safe=True),
    )
    store.mark_executing(proposal.transaction_id)

    store.mark_outcome(proposal.transaction_id, success=success)

    assert store.get(proposal.transaction_id).state is TransactionState.CLOSED
    assert store.active_transaction_ids() == {}


def test_unconnected_vehicle_is_included_in_conflict_check():
    orchestrator = CoordinationOrchestrator(proposer=UnavailableProposer())
    proposal = lane_proposal(201, -1, -2)
    states = {
        201: state(201, -1, 0.0),
        202: state(202, -2, 0.0),
    }

    result = asyncio.run(orchestrator.review(proposal, states))

    assert result.decision == DecisionKind.NACK
    assert result.reason_code == "NO_SAFE_REVISION"
    assert result.validation and result.validation.conflicts


def test_vertically_separated_roads_are_not_conflict_candidates():
    orchestrator = CoordinationOrchestrator(proposer=UnavailableProposer())
    proposal = lane_proposal(211, -1, -2)
    states = {
        211: state(211, -1, 10.0, road_id=35),
        212: state(212, -2, 10.0, road_id=36),
    }
    states[212].z_m = 6.0

    result = asyncio.run(orchestrator.review(proposal, states))

    assert result.decision == DecisionKind.ACK
    assert result.validation and result.validation.safe
    assert result.validation.total_pair_count == 1
    assert result.validation.candidate_pair_count == 0


def test_validator_preserves_mid_lane_change_corridor_occupancy():
    now = time.time()
    states = {
        221: VehicleState(
            veh_id=221,
            observed_at_s=now,
            x_m=0.0,
            y_m=3.5,
            s_m=0.0,
            d_m=3.5,
            speed_mps=58.5 / 3.6,
            lane_id=-2,
            occupied_lane_ids=[-2],
        ),
        222: VehicleState(
            veh_id=222,
            observed_at_s=now,
            x_m=13.97,
            y_m=6.0,
            s_m=13.97,
            d_m=6.0,
            speed_mps=50.0 / 3.6,
            lane_id=-3,
            occupied_lane_ids=[-3, -2],
            lane_change_remaining_s=1.4,
        ),
    }
    plan = JointPlan(
        transaction_id="txn-mid-transition",
        plans={
            221: VehiclePlan(
                veh_id=221,
                steps=[
                    PlanStep(
                        action=Action.ACCELERATE,
                        target_lane_id=-2,
                        target_speed_kmh=60.0,
                    )
                ],
            ),
            222: VehiclePlan(
                veh_id=222,
                steps=[
                    PlanStep(
                        action=Action.KEEP_LANE,
                        target_lane_id=-3,
                        target_speed_kmh=50.0,
                    )
                ],
            ),
        },
    )

    result = DeterministicPlanValidator().validate(
        plan,
        states,
        now_s=now,
    )

    assert not result.safe
    assert any(conflict.kind == "ttc" for conflict in result.conflicts)


def test_conflict_can_be_revised_to_validated_delay_plan():
    orchestrator = CoordinationOrchestrator(proposer=DeterministicYieldProposer())
    proposal = lane_proposal(301, -1, -2)
    states = {
        301: state(301, -1, 0.0),
        302: state(302, -2, 0.0),
    }

    result = asyncio.run(orchestrator.review(proposal, states))

    assert result.decision == DecisionKind.PLAN
    assert result.initial_validation and not result.initial_validation.safe
    assert result.initial_validation.candidate_pair_count == 1
    assert result.initial_validation.sample_count > 0
    assert result.validation and result.validation.safe
    assert result.joint_plan is not None
    assert result.joint_plan.plans[301].steps[0].action == Action.HOLD


def test_stale_telemetry_cannot_execute():
    orchestrator = CoordinationOrchestrator(proposer=DeterministicYieldProposer())
    proposal = lane_proposal(401, -1, -2)
    states = {
        401: state(401, -1, 0.0, age_s=10.0),
        402: state(402, -3, 50.0),
    }

    result = asyncio.run(orchestrator.review(proposal, states))

    assert result.decision == DecisionKind.NACK
    assert any(error.startswith("stale_observation:401") for error in result.validation.errors)


def test_unknown_request_target_is_rejected():
    now = time.time()
    request = CooperationRequest(
        to_vehicle_ids=[999],
        requested_action=Action.YIELD,
        expires_at_s=now + 5.0,
    )
    proposal = lane_proposal(501, -1, -2, request=request)
    result = asyncio.run(
        CoordinationOrchestrator(proposer=DeterministicYieldProposer()).review(
            proposal,
            {501: state(501, -1, 0.0)},
        )
    )
    assert result.decision == DecisionKind.NACK
    assert result.reason_code == "PRECHECK_FAILED"


class ExplodingProposer(SemanticProposer):
    name = "exploding"

    async def propose(self, proposal, states, conflicts, active_plans):
        raise RuntimeError("model unavailable")


class SlowProposer(SemanticProposer):
    name = "slow"

    async def propose(self, proposal, states, conflicts, active_plans):
        await asyncio.sleep(0.05)
        return None


def test_proposer_error_fails_closed():
    proposal = lane_proposal(601, -1, -2)
    states = {601: state(601, -1, 0.0), 602: state(602, -2, 0.0)}
    result = asyncio.run(CoordinationOrchestrator(proposer=ExplodingProposer()).review(proposal, states))
    assert result.decision == DecisionKind.NACK
    assert result.reason_code == "PROPOSER_ERROR"
    assert result.fallback_plans[601].steps[0].action == Action.HOLD


def test_proposer_timeout_fires_and_records_fallback():
    proposal = lane_proposal(701, -1, -2)
    states = {701: state(701, -1, 0.0), 702: state(702, -2, 0.0)}
    result = asyncio.run(
        CoordinationOrchestrator(proposer=SlowProposer(), proposer_timeout_s=0.001).review(
            proposal, states
        )
    )
    assert result.decision == DecisionKind.NACK
    assert result.reason_code == "PROPOSER_TIMEOUT"
    assert result.fallback_plans


def test_unsafe_joint_plan_cannot_be_committed():
    validator = DeterministicPlanValidator()
    store = ActivePlanStore()
    proposal = lane_proposal(801, -1, -2)
    store.register(proposal)
    store.transition(proposal.transaction_id, TransactionState.REVIEWING)
    store.transition(proposal.transaction_id, TransactionState.ACK)
    states = {801: state(801, -1, 0.0), 802: state(802, -2, 0.0)}
    joint = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={801: proposal.plan},
    )
    validation = validator.validate(joint, states)
    assert not validation.safe
    with pytest.raises(ValueError, match="without safe validation"):
        store.commit(proposal.transaction_id, joint, validation)


def test_validator_rejects_revision_that_hides_active_lateral_motion():
    validator = DeterministicPlanValidator()
    states = {
        811: state(811, -3, 0.0),
        812: state(812, -2, 20.0),
    }
    active = {
        811: VehiclePlan(
            veh_id=811,
            summary="already changing toward lane -2",
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
    fictional_revision = JointPlan(
        transaction_id="txn-active-lateral-regression",
        plans={
            811: VehiclePlan(
                veh_id=811,
                summary="yield without representing lateral motion",
                horizon_s=3.0,
                steps=[
                    PlanStep(
                        action=Action.YIELD,
                        target_lane_id=-3,
                        target_speed_kmh=25.0,
                        duration_s=3.0,
                    )
                ],
            )
        }
    )

    result = validator.validate(fictional_revision, states, active)

    assert not result.safe
    assert "active_lateral_commitment_mismatch:811:-2:-3" in result.errors


@pytest.mark.parametrize("vehicle_count", [2, 4, 8])
def test_validator_supports_variable_vehicle_count(vehicle_count):
    now = time.time()
    states = {}
    plans = {}
    for index in range(vehicle_count):
        veh_id = 900 + index
        lane = -1 - (index % 4)
        states[veh_id] = state(veh_id, lane, index * 150.0)
        plans[veh_id] = VehiclePlan(
            veh_id=veh_id,
            summary="maintain sparse flow",
            steps=[
                PlanStep(
                    action=Action.KEEP_LANE,
                    target_lane_id=lane,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        )
    result = DeterministicPlanValidator(
        ValidationConfig(max_observation_age_s=2.0)
    ).validate(JointPlan(transaction_id=f"scale-{vehicle_count}", plans=plans), states, now_s=now)
    assert result.safe
    assert result.checked_vehicle_ids == sorted(plans)


def test_validator_reports_dense_pair_filtering_work():
    now = time.time()
    states = {
        veh_id: state(veh_id, -1, index * 20.0)
        for index, veh_id in enumerate(range(950, 958))
    }
    plans = {
        veh_id: VehiclePlan(
            veh_id=veh_id,
            summary="maintain dense flow",
            steps=[
                PlanStep(
                    action=Action.KEEP_LANE,
                    target_lane_id=-1,
                    target_speed_kmh=40.0,
                    duration_s=3.0,
                )
            ],
        )
        for veh_id in states
    }
    result = DeterministicPlanValidator().validate(
        JointPlan(transaction_id="dense-scale", plans=plans),
        states,
        now_s=now,
    )
    assert result.safe
    assert result.total_pair_count == 28
    assert result.candidate_pair_count > 0
    assert result.sample_count > 0


def test_validator_detects_conflict_across_connected_carla_road_seam():
    now = time.time()
    states = {
        1001: VehicleState(
            veh_id=1001,
            observed_at_s=now,
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
            s_m=59.99,
            speed_mps=40.0 / 3.6,
            lane_id=-4,
            road_id=6,
        ),
        1002: VehicleState(
            veh_id=1002,
            observed_at_s=now,
            x_m=3.5,
            y_m=3.5,
            yaw_deg=0.0,
            s_m=0.05,
            speed_mps=40.0 / 3.6,
            lane_id=-3,
            road_id=45,
        ),
    }
    proposal = lane_proposal(1001, -4, -3)

    result = DeterministicPlanValidator().validate(
        JointPlan(
            transaction_id="connected-road-seam",
            plans={1001: proposal.plan},
        ),
        states,
        now_s=now,
    )

    assert not result.safe
    assert result.candidate_pair_count == 1
    assert any(
        conflict.kind == "predicted_separation"
        and {conflict.veh_a, conflict.veh_b} == {1001, 1002}
        for conflict in result.conflicts
    )


def test_validator_models_implicit_vehicle_commanded_speed():
    now = time.time()
    ego = VehicleState(
        veh_id=1005,
        observed_at_s=now,
        x_m=12.0,
        y_m=0.0,
        yaw_deg=0.0,
        s_m=12.0,
        speed_mps=15.0 / 3.6,
        lane_id=-2,
        road_id=6,
    )
    follower = VehicleState(
        veh_id=1006,
        observed_at_s=now,
        x_m=0.0,
        y_m=0.0,
        yaw_deg=0.0,
        s_m=0.0,
        speed_mps=15.0 / 3.6,
        lane_id=-2,
        road_id=6,
    )
    plan = VehiclePlan(
        veh_id=1005,
        summary="change lane at low speed",
        horizon_s=8.0,
        steps=[
            PlanStep(
                action=Action.LANE_LEFT,
                target_lane_id=-1,
                target_speed_kmh=15.0,
                duration_s=3.0,
            )
        ],
    )
    joint = JointPlan(transaction_id="implicit-desired-speed", plans={1005: plan})
    validator = DeterministicPlanValidator()

    constant_speed = validator.validate(
        joint,
        {1005: ego, 1006: follower},
        now_s=now,
    )
    follower.desired_speed_mps = 50.0 / 3.6
    commanded_speed = validator.validate(
        joint,
        {1005: ego, 1006: follower},
        now_s=now,
    )

    assert constant_speed.safe
    assert not commanded_speed.safe
    assert any(
        conflict.veh_a == 1005 and conflict.veh_b == 1006
        for conflict in commanded_speed.conflicts
    )


def test_validator_keeps_distant_different_roads_out_of_conflict_graph():
    now = time.time()
    states = {
        1011: VehicleState(
            veh_id=1011,
            observed_at_s=now,
            x_m=0.0,
            y_m=0.0,
            yaw_deg=0.0,
            s_m=59.99,
            speed_mps=40.0 / 3.6,
            lane_id=-4,
            road_id=6,
        ),
        1012: VehicleState(
            veh_id=1012,
            observed_at_s=now,
            x_m=500.0,
            y_m=3.5,
            yaw_deg=0.0,
            s_m=0.05,
            speed_mps=40.0 / 3.6,
            lane_id=-3,
            road_id=45,
        ),
    }
    proposal = lane_proposal(1011, -4, -3)

    result = DeterministicPlanValidator().validate(
        JointPlan(
            transaction_id="distant-different-roads",
            plans={1011: proposal.plan},
        ),
        states,
        now_s=now,
    )

    assert result.safe
    assert result.total_pair_count == 1
    assert result.candidate_pair_count == 0


def test_openai_prompt_declares_simultaneous_single_step_contract():
    proposal = lane_proposal(980, -1, -2)
    states = {980: state(980, -1, 0.0), 981: state(981, -2, 0.0)}
    prompt = OpenAIJointPlanProposer(api_key="test")._prompt(
        proposal,
        states,
        [],
        {},
    )
    assert "exactly one step" in prompt
    assert "executes simultaneously" in prompt
    assert "HOLD for the ego and YIELD for the blocker" in prompt
