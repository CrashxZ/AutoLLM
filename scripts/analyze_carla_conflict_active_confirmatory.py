#!/usr/bin/env python3
"""Execute the frozen conflict-active CARLA confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_carla_paired_coordination import (
    build_report,
    read_json,
    read_jsonl,
    replay_gap_metrics,
    sha256_file,
    write_trial_csv,
)
from scripts.run_carla_conflict_active_pilot import METHODS, SCENARIO_FAMILIES


REFERENCE_METHOD = "MIND_CAV_DETERMINISTIC"
PRIMARY_COMPARATORS = ("FCFS_GAP", "MAPPO_ADAPTED")
PRIMARY_FLEET_SIZES = {2, 4}
SECONDARY_FLEET_SIZES = {8}
BOOTSTRAP_SEED = 2_026_091_613
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_COVERAGE = 0.975
FAMILY_ALPHA = 0.05
GAP_SENSITIVITY_THRESHOLDS_M = (3.0, 5.0, 7.0)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_blocks(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    blocks: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        block_id = str(row["block_id"])
        method = str(row["method"])
        if method in blocks[block_id]:
            raise ValueError(f"duplicate method {method} in block {block_id}")
        blocks[block_id][method] = row
    return dict(blocks)


def exact_mcnemar_p(reference_only: int, comparator_only: int) -> float:
    discordant = reference_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = min(reference_only, comparator_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * probability)


def stratified_paired_bootstrap(
    values_by_stratum: Mapping[tuple[str, int], Sequence[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    coverage: float = BOOTSTRAP_COVERAGE,
) -> tuple[float, float]:
    if replicates < 1 or not 0.0 < coverage < 1.0:
        raise ValueError("invalid bootstrap configuration")
    cells = [
        np.asarray(values_by_stratum[key], dtype=np.float64)
        for key in sorted(values_by_stratum)
    ]
    if not cells or any(cell.size == 0 for cell in cells):
        raise ValueError("every registered stratum must contain paired observations")
    rng = np.random.default_rng(seed)
    estimates = np.zeros(replicates, dtype=np.float64)
    total = sum(cell.size for cell in cells)
    for cell in cells:
        indices = rng.integers(0, cell.size, size=(replicates, cell.size))
        estimates += cell[indices].sum(axis=1)
    estimates /= total
    tail = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def holm_adjust(rows: list[dict[str, Any]], *, p_key: str, adjusted_key: str) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda item: float(item[1][p_key])
    )
    running = 0.0
    total = len(ordered)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(
            1.0, float(row[p_key]) * (total - rank)
        )
        running = max(running, adjusted)
        rows[original_index][adjusted_key] = running


def restricted_time(row: Mapping[str, Any], horizon_s: float) -> float:
    return (
        float(row["elapsed_sim_s"])
        if bool(row["safe_task_success"])
        else float(horizon_s)
    )


def primary_analysis(
    rows: Iterable[Mapping[str, Any]], horizon_s: float = 60.0
) -> list[dict[str, Any]]:
    efficacy = [
        row for row in rows if int(row["fleet_size"]) in PRIMARY_FLEET_SIZES
    ]
    blocks = group_blocks(efficacy)
    output: list[dict[str, Any]] = []
    for comparator in PRIMARY_COMPARATORS:
        differences: list[float] = []
        by_stratum: dict[tuple[str, int], list[float]] = defaultdict(list)
        for block_id, methods in blocks.items():
            if REFERENCE_METHOD not in methods or comparator not in methods:
                raise ValueError(f"incomplete primary pair in {block_id}")
            reference = methods[REFERENCE_METHOD]
            baseline = methods[comparator]
            pairing = (
                str(reference["scenario_family"]),
                int(reference["fleet_size"]),
                int(reference["seed"]),
            )
            comparator_pairing = (
                str(baseline["scenario_family"]),
                int(baseline["fleet_size"]),
                int(baseline["seed"]),
            )
            if pairing != comparator_pairing:
                raise ValueError(f"pairing mismatch in {block_id}")
            difference = restricted_time(reference, horizon_s) - restricted_time(
                baseline, horizon_s
            )
            differences.append(difference)
            by_stratum[pairing[:2]].append(difference)
        ci_lower, ci_upper = stratified_paired_bootstrap(by_stratum)
        if all(abs(value) <= 1e-12 for value in differences):
            wilcoxon_p = 1.0
        else:
            wilcoxon_p = float(
                wilcoxon(
                    differences,
                    zero_method="pratt",
                    correction=False,
                    alternative="two-sided",
                    method="approx",
                ).pvalue
            )
        output.append(
            {
                "comparison": f"{REFERENCE_METHOD}-minus-{comparator}",
                "reference_method": REFERENCE_METHOD,
                "comparator": comparator,
                "paired_blocks": len(differences),
                "restricted_time_difference_mean_s": float(np.mean(differences)),
                "restricted_time_difference_median_s": float(
                    np.median(differences)
                ),
                "restricted_time_difference_ci97_5_low_s": ci_lower,
                "restricted_time_difference_ci97_5_high_s": ci_upper,
                "reference_faster_blocks": sum(value < 0.0 for value in differences),
                "comparator_faster_blocks": sum(value > 0.0 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "wilcoxon_zero_method": "pratt",
                "wilcoxon_p": wilcoxon_p,
                "wilcoxon_holm_p": None,
                "h1_supported": None,
            }
        )
    holm_adjust(output, p_key="wilcoxon_p", adjusted_key="wilcoxon_holm_p")
    for row in output:
        row["h1_supported"] = bool(
            float(row["restricted_time_difference_mean_s"]) < 0.0
            and float(row["wilcoxon_holm_p"]) < FAMILY_ALPHA
        )
    return output


def binary_completion_analysis(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    efficacy = [
        row for row in rows if int(row["fleet_size"]) in PRIMARY_FLEET_SIZES
    ]
    blocks = group_blocks(efficacy)
    output: list[dict[str, Any]] = []
    for comparator in PRIMARY_COMPARATORS:
        differences: list[int] = []
        by_stratum: dict[tuple[str, int], list[float]] = defaultdict(list)
        for block_id, methods in blocks.items():
            if REFERENCE_METHOD not in methods or comparator not in methods:
                raise ValueError(f"incomplete primary pair in {block_id}")
            reference = methods[REFERENCE_METHOD]
            baseline = methods[comparator]
            pairing = (
                str(reference["scenario_family"]),
                int(reference["fleet_size"]),
                int(reference["seed"]),
            )
            comparator_pairing = (
                str(baseline["scenario_family"]),
                int(baseline["fleet_size"]),
                int(baseline["seed"]),
            )
            if pairing != comparator_pairing:
                raise ValueError(f"pairing mismatch in {block_id}")
            difference = int(bool(reference["safe_task_success"])) - int(
                bool(baseline["safe_task_success"])
            )
            differences.append(difference)
            by_stratum[pairing[:2]].append(float(difference))
        reference_only = sum(value == 1 for value in differences)
        comparator_only = sum(value == -1 for value in differences)
        ci_lower, ci_upper = stratified_paired_bootstrap(by_stratum)
        output.append(
            {
                "comparison": f"{REFERENCE_METHOD}-minus-{comparator}",
                "reference_method": REFERENCE_METHOD,
                "comparator": comparator,
                "paired_blocks": len(differences),
                "risk_difference": float(np.mean(differences)),
                "risk_difference_ci97_5_low": ci_lower,
                "risk_difference_ci97_5_high": ci_upper,
                "reference_only_successes": reference_only,
                "comparator_only_successes": comparator_only,
                "ties": sum(value == 0 for value in differences),
                "mcnemar_exact_p": exact_mcnemar_p(
                    reference_only, comparator_only
                ),
            }
        )
    return output


def descriptive_summary(
    rows: Iterable[Mapping[str, Any]], horizon_s: float
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["scenario_family"]),
                int(row["fleet_size"]),
                str(row["method"]),
            )
        ].append(row)
    output = []
    for (family, fleet_size, method), values in sorted(groups.items()):
        statuses = Counter(str(row.get("status") or "unknown") for row in values)
        planned = sum(int(row["planned_command_count"]) for row in values)
        terminal = sum(int(row["terminal_done_count"]) for row in values)
        restricted_times = [
            float(row["elapsed_sim_s"])
            if bool(row["safe_task_success"])
            else horizon_s
            for row in values
        ]
        transactions = [row.get("transactions") or {} for row in values]
        applicable_transactions = [
            item for item in transactions if bool(item.get("applicable"))
        ]
        executable = sum(
            int(item.get("executable_decision_count") or 0)
            for item in applicable_transactions
        )
        closed = sum(
            int(item.get("closed_transaction_count") or 0)
            for item in applicable_transactions
        )
        decisions: Counter[str] = Counter()
        for row in values:
            decisions.update(
                {
                    str(key): int(value)
                    for key, value in (row.get("decision_counts") or {}).items()
                }
            )
        output.append(
            {
                "scenario_family": family,
                "fleet_size": fleet_size,
                "method": method,
                "episodes": len(values),
                "fleet_completion_rate": float(
                    np.mean([bool(row["safe_task_success"]) for row in values])
                ),
                "terminal_done_command_fraction": terminal / planned,
                "accepted_command_fraction": sum(
                    int(row["accepted_command_count"]) for row in values
                )
                / planned,
                "restricted_completion_time_mean_s": float(
                    np.mean(restricted_times)
                ),
                "collision_episode_rate": float(
                    np.mean([int(row["collision_count"]) > 0 for row in values])
                ),
                "collision_event_count": sum(
                    int(row["collision_count"]) for row in values
                ),
                "timeout_episode_count": statuses["timeout"],
                "retry_limit_episode_count": statuses["retry_limit"],
                "deadlock_episode_count": statuses["deadlock"],
                "corridor_departure_episode_count": statuses[
                    "corridor_departure"
                ],
                "corridor_departure_episode_rate": statuses[
                    "corridor_departure"
                ]
                / len(values),
                "gap_violation_episode_rate_5m": float(
                    np.mean(
                        [int(row["gap_violation_samples"]) > 0 for row in values]
                    )
                ),
                "gap_violation_sample_count_5m": sum(
                    int(row["gap_violation_samples"]) for row in values
                ),
                "decision_ack_count": decisions["ACK"],
                "decision_plan_count": decisions["PLAN"],
                "decision_nack_count": decisions["NACK"],
                "transaction_audit_completeness": (
                    closed / executable if executable else None
                ),
            }
        )
    return output


def ranker_ablation(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocks = group_blocks(rows)
    output = []
    for fleet_label, fleet_sizes in (
        ("primary_2_4", PRIMARY_FLEET_SIZES),
        ("scalability_8", SECONDARY_FLEET_SIZES),
        ("all", PRIMARY_FLEET_SIZES | SECONDARY_FLEET_SIZES),
    ):
        pairs = [
            (
                methods["MIND_CAV_DETERMINISTIC"],
                methods["MIND_CAV_LEARNED"],
            )
            for methods in blocks.values()
            if int(methods["MIND_CAV_DETERMINISTIC"]["fleet_size"])
            in fleet_sizes
        ]
        differences = [
            int(bool(learned["safe_task_success"]))
            - int(bool(deterministic["safe_task_success"]))
            for deterministic, learned in pairs
        ]
        output.append(
            {
                "population": fleet_label,
                "paired_blocks": len(pairs),
                "learned_minus_deterministic_risk_difference": float(
                    np.mean(differences)
                ),
                "learned_only_successes": sum(value == 1 for value in differences),
                "deterministic_only_successes": sum(
                    value == -1 for value in differences
                ),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return output


def gap_sensitivity(root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        method_dir = metadata_path.parent
        metadata = read_json(metadata_path)
        trajectory = read_jsonl(method_dir / "trajectory.jsonl")
        for threshold_m in GAP_SENSITIVITY_THRESHOLDS_M:
            _, samples = replay_gap_metrics(
                trajectory, threshold_m=threshold_m
            )
            grouped[
                (
                    str(metadata["scenario_family"]),
                    int(metadata["fleet_size"]),
                    str(metadata["method"]),
                    threshold_m,
                )
            ].append((int(samples > 0), samples))
    output = []
    for (family, fleet_size, method, threshold_m), values in sorted(
        grouped.items()
    ):
        output.append(
            {
                "scenario_family": family,
                "fleet_size": fleet_size,
                "method": method,
                "threshold_m": threshold_m,
                "episodes": len(values),
                "gap_violation_episode_rate": float(
                    np.mean([episode for episode, _ in values])
                ),
                "gap_violation_sample_count": sum(
                    samples for _, samples in values
                ),
            }
        )
    return output


def validate_registered_inputs(
    report: Mapping[str, Any], schedule: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if report.get("audit_passed") is not True:
        raise ValueError("independent trajectory audit failed")
    if schedule.get("claim_status") != "frozen_confirmatory":
        raise ValueError("analysis requires a frozen_confirmatory schedule")
    if tuple(schedule.get("methods") or ()) != tuple(METHODS):
        raise ValueError("registered method set changed")
    if tuple(schedule.get("scenario_families") or ()) != tuple(SCENARIO_FAMILIES):
        raise ValueError("registered scenario set changed")
    rows = list(report.get("methods") or [])
    if len(rows) != int(schedule["episode_count"]):
        raise ValueError("observed episode count differs from the fixed schedule")
    blocks = group_blocks(rows)
    if len(blocks) != int(schedule["block_count"]):
        raise ValueError("observed block count differs from the fixed schedule")
    if any(set(methods) != set(METHODS) for methods in blocks.values()):
        raise ValueError("one or more blocks has incomplete method coverage")
    if any(
        int(row.get("initial_conflict_count") or 0) < 1 for row in rows
    ):
        raise ValueError("one or more episodes failed the initial-conflict gate")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    report = build_report(args.root)
    schedule_path = args.root / "schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    rows = validate_registered_inputs(report, schedule)

    args.output.mkdir(parents=True)
    write_csv(
        args.output / "primary_comparisons.csv",
        primary_analysis(rows, float(schedule["trial_timeout_sim_s"])),
    )
    write_csv(
        args.output / "secondary_binary_completion.csv",
        binary_completion_analysis(rows),
    )
    write_csv(
        args.output / "descriptive_by_cell.csv",
        descriptive_summary(rows, float(schedule["trial_timeout_sim_s"])),
    )
    write_csv(args.output / "ranker_ablation.csv", ranker_ablation(rows))
    write_csv(args.output / "gap_threshold_sensitivity.csv", gap_sensitivity(args.root))
    write_trial_csv(args.output / "audited_trials.csv", rows)
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "claim_status": "confirmatory",
        "input_root": str(args.root.resolve()),
        "input_manifest_sha256": sha256_file(args.root / "manifest.json"),
        "input_schedule_sha256": sha256_file(schedule_path),
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "trajectory_audit_source_sha256": sha256_file(
            Path("scripts/analyze_carla_paired_coordination.py")
        ),
        "documented_audit_repair": (
            "docs/carla_conflict_active_audit_repair_2026-09-09.md"
        ),
        "primary_reference": REFERENCE_METHOD,
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "primary_fleet_sizes": sorted(PRIMARY_FLEET_SIZES),
        "secondary_fleet_sizes": sorted(SECONDARY_FLEET_SIZES),
        "primary_outcome": "restricted validator-admissible completion time",
        "primary_test": (
            "two-sided paired Wilcoxon signed-rank, Pratt zero rule, "
            "normal approximation, Holm correction"
        ),
        "family_alpha": FAMILY_ALPHA,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_coverage": BOOTSTRAP_COVERAGE,
        "gap_sensitivity_thresholds_m": list(
            GAP_SENSITIVITY_THRESHOLDS_M
        ),
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
