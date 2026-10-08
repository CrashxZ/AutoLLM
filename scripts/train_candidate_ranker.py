#!/usr/bin/env python3
"""Train and export the compact constrained-candidate cost ranker."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.coordination.candidate_ranking import FEATURE_NAMES, NumpyMLPRanker


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("valid") and row.get("expert_cost") is not None:
            rows.append(row)
    return rows


def matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    x = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([float(row["expert_cost"]) for row in rows], dtype=np.float64)
    return x, y


def ranking_metrics(rows: list[dict], predictions: np.ndarray) -> dict:
    scenes = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        scenes[row["scene_id"]].append((row, float(prediction)))
    agreement = 0
    regret = []
    for candidates in scenes.values():
        expert = min(candidates, key=lambda item: (float(item[0]["expert_cost"]), item[0]["candidate_id"]))
        learned = min(candidates, key=lambda item: (item[1], item[0]["candidate_id"]))
        agreement += int(expert[0]["candidate_id"] == learned[0]["candidate_id"])
        regret.append(float(learned[0]["expert_cost"]) - float(expert[0]["expert_cost"]))
    return {
        "scenes": len(scenes),
        "top1_agreement": agreement / max(len(scenes), 1),
        "mean_expert_regret": float(np.mean(regret)) if regret else None,
        "p95_expert_regret": float(np.percentile(regret, 95)) if regret else None,
        "max_expert_regret": max(regret, default=None),
    }


def export_model(path: Path, scaler: StandardScaler, model: MLPRegressor) -> None:
    arrays = {
        "feature_names": np.asarray(FEATURE_NAMES),
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "layer_count": np.asarray([len(model.coefs_)], dtype=np.int64),
    }
    for index, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        arrays[f"coef_{index}"] = coef
        arrays[f"intercept_{index}"] = intercept
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/derived/candidate_ranker/candidate_dataset.jsonl")
    parser.add_argument("--output", default="data/models/candidate_ranker.npz")
    parser.add_argument("--metrics", default="data/derived/candidate_ranker/training_metrics.json")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    rows = load_rows(Path(args.dataset))
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    x_train, y_train = matrix(by_split["train"])
    if not len(x_train):
        raise ValueError("candidate dataset has no admissible training rows")
    scaler = StandardScaler().fit(x_train)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(256, len(x_train)),
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=args.seed,
        early_stopping=False,
    )
    started = time.perf_counter()
    model.fit(scaler.transform(x_train), y_train)
    training_s = time.perf_counter() - started
    output_path = Path(args.output)
    export_model(output_path, scaler, model)
    portable = NumpyMLPRanker(str(output_path))

    metrics = {
        "label": "EXPLORATORY_TRAINING",
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "hidden_layer_sizes": [32, 16],
        "training_seconds": training_s,
        "iterations": int(model.n_iter_),
        "loss": float(model.loss_),
        "splits": {},
        "portable_max_abs_error": 0.0,
    }
    all_portable_errors = []
    for split, split_rows in by_split.items():
        if not split_rows:
            metrics["splits"][split] = {
                "candidate_rows": 0,
                "scenes": 0,
                "mae": None,
                "rmse": None,
                "top1_agreement": None,
                "mean_expert_regret": None,
                "p95_expert_regret": None,
                "max_expert_regret": None,
            }
            continue
        x, y = matrix(split_rows)
        predictions = model.predict(scaler.transform(x))
        portable_predictions = portable.predict([row["features"] for row in split_rows])
        all_portable_errors.extend(np.abs(predictions - portable_predictions).tolist())
        metrics["splits"][split] = {
            "candidate_rows": len(split_rows),
            "mae": float(mean_absolute_error(y, predictions)),
            "rmse": float(mean_squared_error(y, predictions) ** 0.5),
            **ranking_metrics(split_rows, predictions),
        }
    metrics["portable_max_abs_error"] = max(all_portable_errors, default=0.0)

    test_groups = defaultdict(list)
    for row in by_split["test"]:
        test_groups[(row["scene_id"], row["fleet_size"])].append(row)
    latencies = defaultdict(list)
    for (_, fleet_size), candidates in list(test_groups.items())[:2000]:
        started_ns = time.perf_counter_ns()
        portable.predict([row["features"] for row in candidates])
        latencies[int(fleet_size)].append((time.perf_counter_ns() - started_ns) / 1e6)
    metrics["inference_latency_ms"] = {
        str(fleet_size): {
            "calls": len(values),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }
        for fleet_size, values in sorted(latencies.items())
    }
    test_metrics = metrics["splits"]["test"]
    if not test_metrics["candidate_rows"]:
        raise ValueError("candidate dataset has no admissible test rows")
    metrics["acceptance"] = {
        "top1_agreement_at_least_90pct": test_metrics["top1_agreement"] >= 0.90,
        "portable_matches_sklearn": metrics["portable_max_abs_error"] < 1e-9,
        "p95_inference_below_10ms": all(
            item["p95"] < 10.0 for item in metrics["inference_latency_ms"].values()
        ),
    }
    metrics["acceptance_passed"] = all(metrics["acceptance"].values())
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
