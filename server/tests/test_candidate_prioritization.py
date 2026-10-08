import asyncio

from scripts.generate_candidate_ranker_dataset import build_scene
from server.coordination.candidate_prioritization import (
    CHEAP_FEATURE_NAMES,
    CheapCandidateFeatureExtractor,
    PrioritizedCandidateProposer,
)
from server.coordination.candidate_ranking import CandidatePlanGenerator
from server.coordination.audit import AuditLog
from server.coordination.models import JointPlan
from server.coordination.orchestrator import CoordinationOrchestrator
from server.coordination.transaction import ActivePlanStore
from server.coordination.validator import DeterministicPlanValidator


def scene(geometry: str):
    return build_scene(
        {
            "scene_id": f"test-{geometry}",
            "fleet_size": 2,
            "geometry": geometry,
            "speed_stratum": "near_flow",
            "seed": 8128,
            "ordinal": 0,
        }
    )


def initial_conflicts(proposal, states, validator):
    original = JointPlan(
        transaction_id=proposal.transaction_id,
        plans={proposal.ego_veh_id: proposal.plan},
    )
    return validator.validate(
        original,
        states,
        now_s=proposal.created_at_s,
    ).conflicts


def test_cheap_features_match_frozen_schema_without_validation() -> None:
    proposal, states = scene("side_by_side")
    validator = DeterministicPlanValidator()
    conflicts = initial_conflicts(proposal, states, validator)
    candidate = CandidatePlanGenerator(
        enable_liveness_preparation=True
    ).generate(proposal, states, conflicts)[0]
    features = CheapCandidateFeatureExtractor().extract(
        candidate,
        proposal,
        states,
        conflicts,
    )
    assert tuple(features) == CHEAP_FEATURE_NAMES
    assert features["fleet_count"] == 2.0
    assert features["template_original"] == 1.0


def test_sequential_validation_falls_through_invalid_first_candidate() -> None:
    proposal, states = scene("side_by_side")
    validator = DeterministicPlanValidator()
    conflicts = initial_conflicts(proposal, states, validator)
    proposer = PrioritizedCandidateProposer(
        validator=validator,
        batch_size=1,
    )
    plan = asyncio.run(proposer.propose(proposal, states, conflicts, {}))
    assert plan is not None
    assert proposer.last_stats.fallback_batches == 1
    assert 1 < proposer.last_stats.validated_candidates
    assert (
        proposer.last_stats.validated_candidates
        < proposer.last_stats.generated_candidates
    )
    assert validator.validate(
        plan,
        states,
        now_s=proposal.created_at_s,
    ).safe


def test_sequential_validation_stops_after_valid_fast_path() -> None:
    proposal, states = scene("sparse")
    validator = DeterministicPlanValidator()
    conflicts = initial_conflicts(proposal, states, validator)
    proposer = PrioritizedCandidateProposer(
        validator=validator,
        batch_size=1,
    )
    plan = asyncio.run(proposer.propose(proposal, states, conflicts, {}))
    assert plan is not None
    assert proposer.last_stats.validated_candidates == 1
    assert proposer.last_stats.fallback_batches == 0


def test_missing_model_falls_back_to_heuristic_ordering() -> None:
    proposer = PrioritizedCandidateProposer(model_path="missing-model.npz")
    assert proposer.model is None
    assert proposer.model_error == "FileNotFoundError"
    assert proposer.name == "sequential-heuristic-prioritizer"


def test_orchestrator_revalidates_prioritized_plan_before_commit() -> None:
    proposal, states = scene("side_by_side")
    audit = AuditLog()
    proposer = PrioritizedCandidateProposer(batch_size=1)
    orchestrator = CoordinationOrchestrator(
        proposer=proposer,
        store=ActivePlanStore(audit=audit),
    )
    decision = asyncio.run(
        orchestrator.review(
            proposal,
            states,
            now_s=proposal.created_at_s,
        )
    )
    assert decision.decision.value == "PLAN"
    assert proposer.last_stats.fallback_batches == 1
    assert decision.validation is not None
    assert decision.validation.safe
    assert audit.has_validation_before_commit(proposal.transaction_id)
