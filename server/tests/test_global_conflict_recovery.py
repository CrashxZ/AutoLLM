"""Joint recovery must address independent blockers without relaxing admission."""
from server.coordination.candidate_ranking import CandidatePlanGenerator
from server.coordination.models import Action, Goal, GoalKind, IntentProposal, JointPlan, PlanStep, VehiclePlan, VehicleState
from server.coordination.validator import DeterministicPlanValidator


def scene():
    states = {}
    for vid, lane, position, speed, desired in [
        (1, 3, 0, 7.6, 50), (2, 3, 13.3, 13.6, 13.9),
        (3, 5, 100, 10.7, 50), (4, 5, 115.8, 15.1, 13.9),
    ]:
        # Positive-ID lanes travel toward decreasing road s.
        states[vid] = VehicleState(veh_id=vid, observed_at_s=100, lane_id=lane, x_m=position, y_m=0, s_m=200-position, speed_mps=speed/3.6, desired_speed_mps=desired/3.6)
    proposal = IntentProposal(ego_veh_id=1, created_at_s=100, expires_at_s=105, observation_ts_s=100, goal=Goal(kind=GoalKind.TARGET_LANE,target_lane_id=4), protected_goal_vehicle_ids=[1,3], plan=VehiclePlan(veh_id=1,horizon_s=8,steps=[PlanStep(action=Action.LANE_RIGHT,target_lane_id=4,target_speed_kmh=50,duration_s=3)]))
    validator = DeterministicPlanValidator()
    conflicts = validator.validate(JointPlan(transaction_id='test',plans={1:proposal.plan}),states,now_s=100).conflicts
    return states, proposal, validator, conflicts


def test_global_recovery_addresses_disconnected_conflict_component():
    states, proposal, validator, conflicts = scene()
    candidates = CandidatePlanGenerator(enable_global_conflict_recovery=True).generate(proposal,states,conflicts)
    recovery = next(c for c in candidates if c.template=='global_flow_recovery')
    assert set(recovery.joint_plan.plans)=={1,2,3,4}
    assert recovery.joint_plan.plans[3].steps[0].action==Action.HOLD
    assert recovery.joint_plan.plans[2].steps[0].action==Action.ACCELERATE
    assert validator.validate(recovery.joint_plan,states,now_s=100).safe
    assert len(candidates)<=16


def test_historical_generator_does_not_enable_recovery_implicitly():
    states, proposal, _, conflicts = scene()
    candidates = CandidatePlanGenerator().generate(proposal,states,conflicts)
    assert not any(c.template=='global_flow_recovery' for c in candidates)
