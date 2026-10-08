#!/usr/bin/env python3
"""Generate grouped exploratory scenes and constrained-candidate labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.coordination.candidate_ranking import CandidateFeatureExtractor, CandidatePlanGenerator
from server.coordination.models import (
    Action,
    CooperationRequest,
    Goal,
    GoalKind,
    IntentProposal,
    JointPlan,
    PlanStep,
    VehiclePlan,
    VehicleState,
)
from server.coordination.validator import DeterministicPlanValidator


MASTER_SEED = 20260821
FLEET_SIZES = (2, 4, 8, 16)
GEOMETRIES = ("side_by_side", "front_blocker", "rear_blocker", "multi_conflict", "sparse")
SPEED_STRATA = ("below_flow", "near_flow", "mixed_closing")


def split_by_stratum(scene_specs: List[dict], seed: int) -> Dict[str, str]:
    groups: Dict[Tuple[int, str, str], List[str]] = defaultdict(list)
    for spec in scene_specs:
        groups[(spec["fleet_size"], spec["geometry"], spec["speed_stratum"])].append(
            spec["scene_id"]
        )
    result = {}
    for key, scene_ids in sorted(groups.items()):
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(scene_ids)
        count = len(scene_ids)
        if count >= 3:
            train_count = max(1, int(0.70 * count))
            validation_count = max(1, int(0.15 * count))
            if train_count + validation_count >= count:
                train_count = count - validation_count - 1
        else:
            train_count = max(1, count - 1)
            validation_count = 0
        train_end = train_count
        validation_end = train_end + validation_count
        for index, scene_id in enumerate(scene_ids):
            result[scene_id] = (
                "train"
                if index < train_end
                else "validation" if index < validation_end else "test"
            )
    return result


def build_scene(spec: dict) -> Tuple[IntentProposal, Dict[int, VehicleState]]:
    rng = np.random.default_rng(spec["seed"])
    count = spec["fleet_size"]
    geometry = spec["geometry"]
    speed_stratum = spec["speed_stratum"]
    now = 1_800_000_000.0 + spec["ordinal"]

    if speed_stratum == "below_flow":
        ego_speed = float(rng.uniform(32.0, 40.0))
        other_speed = lambda: float(rng.uniform(30.0, 44.0))
    elif speed_stratum == "near_flow":
        ego_speed = float(rng.uniform(47.0, 53.0))
        other_speed = lambda: float(rng.uniform(46.0, 54.0))
    else:
        ego_speed = float(rng.uniform(42.0, 48.0))
        other_speed = lambda: float(rng.choice([rng.uniform(28.0, 38.0), rng.uniform(52.0, 62.0)]))

    positions = []
    if geometry == "side_by_side":
        positions = [(-1, 0.0, ego_speed), (-2, float(rng.uniform(-2.0, 2.0)), other_speed())]
    elif geometry == "front_blocker":
        positions = [(-1, 0.0, ego_speed), (-2, float(rng.uniform(6.0, 16.0)), min(other_speed(), ego_speed))]
    elif geometry == "rear_blocker":
        positions = [(-1, 0.0, ego_speed), (-2, -float(rng.uniform(6.0, 16.0)), max(other_speed(), ego_speed))]
    elif geometry == "multi_conflict":
        # The near target-lane vehicle conflicts with the proposed merge. A
        # farther, faster rear vehicle can create a connected secondary
        # conflict without beginning below the TTC threshold. The generator
        # must coordinate the complete ego-connected conflict component.
        positions = [(-1, 0.0, ego_speed), (-2, float(rng.uniform(6.0, 10.0)), min(other_speed(), ego_speed))]
        if count >= 3:
            positions.append((-2, -float(rng.uniform(42.0, 52.0)), max(other_speed(), ego_speed)))
    else:
        positions = [(-1, 0.0, ego_speed), (-2, float(rng.uniform(80.0, 120.0)), other_speed())]

    while len(positions) < count:
        index = len(positions)
        lane = -1 - (index % 4)
        rank = index // 4 + 1
        sign = -1.0 if index % 2 == 0 else 1.0
        # Background traffic must not introduce an unrelated conflict that no
        # ego-centred candidate can resolve. Same-lane vehicles are separated
        # far enough to remain clear under the full mixed-speed closing bound.
        s_m = sign * (80.0 + rank * 150.0 + float(rng.uniform(0.0, 5.0)))
        positions.append((lane, s_m, other_speed()))

    states = {}
    for index, (lane, s_m, speed_kmh) in enumerate(positions):
        veh_id = 1000 + index
        states[veh_id] = VehicleState(
            veh_id=veh_id,
            observed_at_s=now,
            x_m=s_m,
            y_m=float(abs(lane) - 1) * 3.5,
            s_m=s_m,
            d_m=float(abs(lane) - 1) * 3.5,
            speed_mps=max(0.0, speed_kmh / 3.6),
            lane_id=lane,
            road_id=1,
            vehicle_class="emergency" if rng.random() < 0.03 else "standard",
        )

    request = None
    if geometry != "sparse" and rng.random() < 0.75:
        request = CooperationRequest(
            to_vehicle_ids=[1001],
            requested_action=Action.YIELD,
            reason="Create a merge gap",
            expires_at_s=now + 5.0,
        )
    proposal = IntentProposal(
        transaction_id=f"scene-{spec['scene_id']}",
        ego_veh_id=1000,
        created_at_s=now,
        expires_at_s=now + 5.0,
        observation_ts_s=now,
        goal=Goal(kind=GoalKind.TARGET_LANE, target_lane_id=-2),
        plan=VehiclePlan(
            veh_id=1000,
            summary="Move toward target lane",
            horizon_s=3.0,
            steps=[
                PlanStep(
                    action=Action.LANE_RIGHT,
                    target_lane_id=-2,
                    target_speed_kmh=50.0,
                    duration_s=3.0,
                )
            ],
        ),
        request=request,
        source="candidate-dataset-generator",
    )
    return proposal, states


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    specs = []
    ordinal = 0
    for fleet_size in args.fleet_sizes:
        for geometry in GEOMETRIES:
            for speed_stratum in SPEED_STRATA:
                for replicate in range(args.scenes_per_stratum):
                    scene_id = f"n{fleet_size}-{geometry}-{speed_stratum}-{replicate:04d}"
                    specs.append(
                        {
                            "scene_id": scene_id,
                            "fleet_size": fleet_size,
                            "geometry": geometry,
                            "speed_stratum": speed_stratum,
                            "seed": args.seed + ordinal * 104729,
                            "ordinal": ordinal,
                        }
                    )
                    ordinal += 1
    split_map = split_by_stratum(specs, args.seed)
    validator = DeterministicPlanValidator()
    generator = CandidatePlanGenerator()
    extractor = CandidateFeatureExtractor(validator)
    rows = 0
    scenes_with_admissible = 0
    scenes_without_admissible = []
    split_counts = Counter()
    template_counts = Counter()
    path = output / "candidate_dataset.jsonl"
    with path.open("w", encoding="utf-8", buffering=1) as handle:
        for spec in specs:
            proposal, states = build_scene(spec)
            now_s = proposal.created_at_s
            original = JointPlan(
                transaction_id=proposal.transaction_id,
                plans={proposal.ego_veh_id: proposal.plan},
                proposer="vehicle",
            )
            initial = validator.validate(original, states, now_s=now_s)
            candidates = generator.generate(proposal, states, initial.conflicts)
            evaluations = [
                extractor.evaluate(
                    candidate,
                    proposal,
                    states,
                    initial.conflicts,
                    {},
                    now_s,
                )
                for candidate in candidates
            ]
            admissible = [item for item in evaluations if item.valid]
            if admissible:
                scenes_with_admissible += 1
            else:
                scenes_without_admissible.append(spec["scene_id"])
            expert_id = min(
                admissible,
                key=lambda item: (item.expert_cost, item.candidate_id),
                default=None,
            )
            split = split_map[spec["scene_id"]]
            split_counts[split] += 1
            for evaluation in evaluations:
                row = {
                    **spec,
                    "split": split,
                    "candidate_id": evaluation.candidate_id,
                    "template": evaluation.template,
                    "valid": evaluation.valid,
                    "expert_cost": evaluation.expert_cost,
                    "expert_selected": bool(
                        expert_id and evaluation.candidate_id == expert_id.candidate_id
                    ),
                    "features": evaluation.features,
                    "validation": evaluation.validation.model_dump(
                        mode="json", exclude_none=True
                    ),
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows += 1
                template_counts[evaluation.template] += 1

    manifest = {
        "label": "EXPLORATORY_TRAINING_DATA",
        "master_seed": args.seed,
        "scenes": len(specs),
        "candidate_rows": rows,
        "scenes_with_admissible_candidate": scenes_with_admissible,
        "scenes_without_admissible_candidate": scenes_without_admissible,
        "fleet_sizes": args.fleet_sizes,
        "geometries": GEOMETRIES,
        "speed_strata": SPEED_STRATA,
        "scenes_per_stratum": args.scenes_per_stratum,
        "split_scene_counts": dict(split_counts),
        "template_counts": dict(template_counts),
        "dataset": str(path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if scenes_without_admissible:
        raise RuntimeError(
            f"{len(scenes_without_admissible)} generated scenes have no admissible candidate"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/derived/candidate_ranker")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--scenes-per-stratum", type=int, default=100)
    parser.add_argument("--fleet-sizes", default="2,4,8,16")
    args = parser.parse_args()
    args.fleet_sizes = [int(value) for value in args.fleet_sizes.split(",")]
    generate(args)


if __name__ == "__main__":
    main()
