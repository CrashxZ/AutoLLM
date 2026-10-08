#!/usr/bin/env python3
"""Benchmark deterministic and learned candidate-ranking latency.

The output is exploratory. Ranking-only measurements use identical candidate
feature batches; full-proposer measurements exercise generation, validation,
feature extraction, and production ranking.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_candidate_ranker_dataset import build_scene
from server.coordination.candidate_ranking import (
    CandidateFeatureExtractor,
    NumpyMLPRanker,
    RankedCandidateProposer,
)
from server.coordination.models import JointPlan
from server.coordination.validator import DeterministicPlanValidator


DEFAULT_SEED = 2026091701


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, q)) if len(array) else float("nan")


def load_feature_pool(dataset: Path) -> dict[int, list[dict]]:
    pool: dict[int, list[dict]] = defaultdict(list)
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "test" or not row.get("valid"):
                continue
            pool[int(row["fleet_size"])].append(
                {
                    "features": {
                        name: float(value)
                        for name, value in row["features"].items()
                    },
                    "expert_cost": float(row["expert_cost"]),
                }
            )
    if not pool:
        raise ValueError("dataset contains no admissible test candidates")
    return dict(pool)


def candidate_batch(
    pool: dict[int, list[dict]], fleet_size: int, candidate_count: int, seed: int
) -> list[dict]:
    source_fleet = fleet_size if fleet_size in pool else max(pool)
    source = pool[source_fleet]
    rng = random.Random(seed)
    batch = []
    for _ in range(candidate_count):
        item = source[rng.randrange(len(source))]
        features = dict(item["features"])
        features["fleet_count"] = float(fleet_size)
        batch.append(
            {
                "features": features,
                "expert_cost": CandidateFeatureExtractor.expert_cost(features),
            }
        )
    return batch


def deterministic_select(batch: list[dict]) -> int:
    return min(
        range(len(batch)),
        key=lambda index: (batch[index]["expert_cost"], index),
    )


def learned_select(model: NumpyMLPRanker, batch: list[dict]) -> int:
    predictions = model.predict([item["features"] for item in batch])
    return min(
        range(len(batch)),
        key=lambda index: (float(predictions[index]), index),
    )


def timed_call(function: Callable[[], int], inner_loops: int) -> tuple[float, int]:
    result = -1
    started_ns = time.perf_counter_ns()
    for _ in range(inner_loops):
        result = function()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6 / inner_loops
    return elapsed_ms, result


def ranking_only_benchmark(
    pool: dict[int, list[dict]],
    model: NumpyMLPRanker,
    fleet_sizes: list[int],
    candidate_counts: list[int],
    blocks: int,
    inner_loops: int,
    seed: int,
) -> list[dict]:
    rows = []
    order_rng = random.Random(seed)
    for fleet_size in fleet_sizes:
        for candidate_count in candidate_counts:
            batch = candidate_batch(
                pool,
                fleet_size,
                candidate_count,
                seed + fleet_size * 1009 + candidate_count * 9176,
            )
            for _ in range(50):
                deterministic_select(batch)
                learned_select(model, batch)
            for block in range(blocks):
                methods = ["deterministic", "learned"]
                order_rng.shuffle(methods)
                for order, method in enumerate(methods):
                    if method == "deterministic":
                        latency_ms, selected = timed_call(
                            lambda: deterministic_select(batch), inner_loops
                        )
                    else:
                        latency_ms, selected = timed_call(
                            lambda: learned_select(model, batch), inner_loops
                        )
                    rows.append(
                        {
                            "benchmark": "ranking_only",
                            "fleet_size": fleet_size,
                            "candidate_count": candidate_count,
                            "block": block,
                            "order": order,
                            "method": method,
                            "latency_ms": latency_ms,
                            "selected_index": selected,
                            "stress_extrapolation": candidate_count > 16,
                        }
                    )
    return rows


async def full_proposer_benchmark(
    model_path: Path,
    fleet_sizes: list[int],
    blocks: int,
    seed: int,
) -> list[dict]:
    rows = []
    order_rng = random.Random(seed + 1)
    validator = DeterministicPlanValidator()
    proposers = {
        "deterministic": RankedCandidateProposer(
            validator=validator,
            enable_liveness_preparation=True,
        ),
        "learned": RankedCandidateProposer(
            validator=validator,
            model_path=str(model_path),
            enable_liveness_preparation=True,
        ),
    }
    if proposers["learned"].model is None:
        raise ValueError(
            f"learned ranker failed to load: {proposers['learned'].model_error}"
        )

    for fleet_size in fleet_sizes:
        warm_spec = {
            "scene_id": f"latency-warmup-n{fleet_size}",
            "fleet_size": fleet_size,
            "geometry": "multi_conflict",
            "speed_stratum": "mixed_closing",
            "seed": seed + fleet_size * 99_991,
            "ordinal": fleet_size * 10_000 - 1,
        }
        warm_proposal, warm_states = build_scene(warm_spec)
        warm_original = JointPlan(
            transaction_id=warm_proposal.transaction_id,
            plans={warm_proposal.ego_veh_id: warm_proposal.plan},
            proposer="vehicle",
        )
        warm_initial = validator.validate(
            warm_original,
            warm_states,
            now_s=warm_proposal.created_at_s,
        )
        for proposer in proposers.values():
            await proposer.propose(
                warm_proposal,
                warm_states,
                warm_initial.conflicts,
                {},
            )
        for block in range(blocks):
            spec = {
                "scene_id": f"latency-n{fleet_size}-{block:04d}",
                "fleet_size": fleet_size,
                "geometry": "multi_conflict",
                "speed_stratum": "mixed_closing",
                "seed": seed + fleet_size * 100_003 + block * 104_729,
                "ordinal": fleet_size * 10_000 + block,
            }
            proposal, states = build_scene(spec)
            original = JointPlan(
                transaction_id=proposal.transaction_id,
                plans={proposal.ego_veh_id: proposal.plan},
                proposer="vehicle",
            )
            initial = validator.validate(
                original,
                states,
                now_s=proposal.created_at_s,
            )
            methods = ["deterministic", "learned"]
            order_rng.shuffle(methods)
            for order, method in enumerate(methods):
                proposer = proposers[method]
                started_ns = time.perf_counter_ns()
                selected = await proposer.propose(
                    proposal,
                    states,
                    initial.conflicts,
                    {},
                )
                latency_ms = (time.perf_counter_ns() - started_ns) / 1e6
                evaluations = proposer.last_evaluations
                rows.append(
                    {
                        "benchmark": "full_proposer",
                        "fleet_size": fleet_size,
                        "candidate_count": len(evaluations),
                        "admissible_count": sum(item.valid for item in evaluations),
                        "block": block,
                        "order": order,
                        "method": method,
                        "latency_ms": latency_ms,
                        "selected_candidate_id": proposer.last_selected_candidate_id,
                        "returned_plan": selected is not None,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["benchmark"],
            int(row["fleet_size"]),
            int(row["candidate_count"]),
            row["method"],
        )
        grouped[key].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        values = [float(row["latency_ms"]) for row in group]
        output.append(
            {
                "benchmark": key[0],
                "fleet_size": key[1],
                "candidate_count": key[2],
                "method": key[3],
                "blocks": len(group),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 95),
                "max_ms": max(values),
            }
        )
    return output


def paired_summary(rows: list[dict]) -> list[dict]:
    by_block: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row["benchmark"],
            int(row["fleet_size"]),
            int(row["candidate_count"]),
            int(row["block"]),
        )
        by_block[key][row["method"]] = float(row["latency_ms"])
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for key, methods in by_block.items():
        if set(methods) == {"deterministic", "learned"}:
            grouped[key[:3]].append(methods["learned"] - methods["deterministic"])
    return [
        {
            "benchmark": key[0],
            "fleet_size": key[1],
            "candidate_count": key[2],
            "paired_blocks": len(values),
            "learned_minus_deterministic_mean_ms": statistics.fmean(values),
            "learned_minus_deterministic_median_ms": statistics.median(values),
            "learned_faster_blocks": sum(value < 0 for value in values),
            "deterministic_faster_blocks": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
        }
        for key, values in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    fieldnames.extend(
        sorted({name for row in rows for name in row} - set(fieldnames))
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def main_async(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pool = load_feature_pool(args.dataset)
    model = NumpyMLPRanker(str(args.model))
    ranking_rows = ranking_only_benchmark(
        pool,
        model,
        args.fleet_sizes,
        args.candidate_counts,
        args.blocks,
        args.inner_loops,
        args.seed,
    )
    full_rows = await full_proposer_benchmark(
        args.model,
        args.full_fleet_sizes,
        args.full_blocks,
        args.seed,
    )
    rows = ranking_rows + full_rows
    summaries = summarize(rows)
    paired = paired_summary(rows)
    write_csv(output / "latency_raw.csv", rows)
    write_csv(output / "latency_summary.csv", summaries)
    write_csv(output / "latency_paired.csv", paired)
    manifest = {
        "label": "EXPLORATORY_CANDIDATE_RANKER_LATENCY",
        "seed": args.seed,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "fleet_sizes": args.fleet_sizes,
        "candidate_counts": args.candidate_counts,
        "blocks": args.blocks,
        "inner_loops": args.inner_loops,
        "full_fleet_sizes": args.full_fleet_sizes,
        "full_blocks": args.full_blocks,
        "python": sys.version,
        "platform": platform.platform(),
        "production_note": (
            "Both production paths compute validation, features, and expert cost; "
            "the learned path additionally invokes the MLP."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "paired": paired}, indent=2))


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/derived/candidate_ranker/candidate_dataset.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/models/candidate_ranker.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/candidate_ranker_latency_ood/latency"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fleet-sizes", type=parse_ints, default=parse_ints("2,4,8,16,32")
    )
    parser.add_argument(
        "--candidate-counts", type=parse_ints, default=parse_ints("4,8,16,32,64")
    )
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--inner-loops", type=int, default=20)
    parser.add_argument(
        "--full-fleet-sizes", type=parse_ints, default=parse_ints("2,8,32")
    )
    parser.add_argument("--full-blocks", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
