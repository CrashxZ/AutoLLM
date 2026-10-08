import subprocess
import sys

from scripts.analyze_carla_conflict_active_confirmatory import (
    exact_mcnemar_p,
    primary_analysis,
    stratified_paired_bootstrap,
)


def test_confirmatory_analysis_cli_is_importable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_carla_conflict_active_confirmatory.py",
            "--help",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _row(
    block,
    method,
    success,
    fleet_size=2,
    family="close_reciprocal",
    elapsed_sim_s=10.0,
):
    return {
        "block_id": block,
        "method": method,
        "safe_task_success": success,
        "fleet_size": fleet_size,
        "scenario_family": family,
        "seed": 100 + int(block[1:]),
        "elapsed_sim_s": elapsed_sim_s,
    }


def test_exact_mcnemar_is_two_sided() -> None:
    assert exact_mcnemar_p(4, 0) == 0.125
    assert exact_mcnemar_p(0, 4) == 0.125
    assert exact_mcnemar_p(0, 0) == 1.0


def test_primary_analysis_excludes_fleet_eight_and_applies_holm() -> None:
    rows = []
    outcomes = [
        (True, False, False),
        (True, False, False),
        (True, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, False),
    ]
    for index, (mind, fcfs, mappo) in enumerate(outcomes, start=1):
        block = f"b{index}"
        rows.extend(
            [
                _row(block, "MIND_CAV_DETERMINISTIC", mind),
                _row(block, "FCFS_GAP", fcfs),
                _row(block, "MAPPO_ADAPTED", mappo),
            ]
        )
    rows.extend(
        [
            _row("b7", "MIND_CAV_DETERMINISTIC", False, fleet_size=8),
            _row("b7", "FCFS_GAP", True, fleet_size=8),
            _row("b7", "MAPPO_ADAPTED", True, fleet_size=8),
        ]
    )

    result = primary_analysis(rows)

    assert {row["paired_blocks"] for row in result} == {6}
    assert all(row["wilcoxon_holm_p"] is not None for row in result)
    assert all(row["wilcoxon_zero_method"] == "pratt" for row in result)


def test_stratified_bootstrap_is_reproducible() -> None:
    cells = {("a", 2): [1, 0, 1], ("b", 4): [0, -1, 1]}

    first = stratified_paired_bootstrap(cells, seed=42, replicates=100)
    second = stratified_paired_bootstrap(cells, seed=42, replicates=100)

    assert first == second
