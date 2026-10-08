#!/usr/bin/env python3
"""Plot six scenario panels from the existing CARLA v4 episode analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'data/derived/campaign'
METHODS = (
    "FCFS_GAP",
    "MAPPO_ADAPTED",
    "MIND_CAV_DETERMINISTIC",
    "MIND_CAV_LEARNED",
)
LABELS = (
    "FCFS-GAP",
    "Adapted MAPPO",
    "MIND-CAV deterministic",
    "MIND-CAV learned",
)
SCENARIOS = (
    ("close_reciprocal", "S1: Reciprocal exchange"),
    ("contested_merge", "S2: Contested merge"),
    ("blocked_merge", "S3: Blocked merge"),
)
FLEETS = (2, 4, 8)
COLORS = ("#4C78A8", "#D2A14A", "#238A83", "#9D76B2")
GRAY = ("#D2D2D2", "#A8A8A8", "#727272", "#303030")
HORIZON = 60.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(source: Path, cell_source: Path) -> list[dict]:
    rows = read_csv(source)
    if len(rows) != 756:
        raise ValueError(f"Expected 756 episodes, found {len(rows)}")
    if len({(row["block_id"], row["method"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate episode identifiers")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario_family"], int(row["fleet_size"]), row["method"])].append(row)
    expected = {
        (scenario, fleet, method)
        for scenario, _ in SCENARIOS
        for fleet in FLEETS
        for method in METHODS
    }
    if set(groups) != expected:
        raise ValueError("Unexpected or missing scenario/fleet/method cells")
    reference = {
        (row["scenario_family"], int(row["fleet_size"]), row["method"]): row
        for row in read_csv(cell_source)
    }
    output = []
    for scenario, _ in SCENARIOS:
        for fleet in FLEETS:
            for method in METHODS:
                key = (scenario, fleet, method)
                cell = groups[key]
                if len(cell) != 21:
                    raise ValueError(f"Expected 21 episodes in {key}")
                success = [row["safe_task_success"].strip().lower() == "true" for row in cell]
                times = [
                    float(row["elapsed_sim_s"]) if passed else HORIZON
                    for row, passed in zip(cell, success)
                ]
                rate = float(np.mean(success))
                mean = float(np.mean(times))
                if not np.isclose(rate, float(reference[key]["fleet_completion_rate"]), atol=1e-12):
                    raise ValueError(f"Success differs from canonical summary: {key}")
                if not np.isclose(mean, float(reference[key]["restricted_completion_time_mean_s"]), atol=1e-10):
                    raise ValueError(f"Restricted time differs from canonical summary: {key}")
                output.append({
                    "scenario": scenario,
                    "fleet_size": fleet,
                    "method": method,
                    "runs": len(cell),
                    "successful_runs": sum(success),
                    "fleet_task_success_percent": 100 * rate,
                    "mean_restricted_completion_time_s": mean,
                })
    return output


def plot(rows: list[dict], output: Path, grayscale: bool = False, note: str | None = None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    values = {(r["scenario"], r["fleet_size"], r["method"]): r for r in rows}
    palette = GRAY if grayscale else COLORS
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), sharey="row", sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.86, wspace=0.12, hspace=0.18)
    if note:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=7)
    positions = np.arange(3)
    width = 0.19
    for col, (scenario, title) in enumerate(SCENARIOS):
        axes[0, col].set_title(title, pad=8)
        for method_index, method in enumerate(METHODS):
            cells = [values[(scenario, fleet, method)] for fleet in FLEETS]
            x = positions + (method_index - 1.5) * width
            for row_index, field in enumerate((
                "fleet_task_success_percent", "mean_restricted_completion_time_s"
            )):
                axes[row_index, col].bar(
                    x, [cell[field] for cell in cells], width=width * 0.93,
                    color=palette[method_index], edgecolor="#3F3F3F" if grayscale else "none",
                    linewidth=0.35, zorder=3,
                )
                if row_index == 0:
                    for bar_x, cell in zip(x, cells):
                        if cell[field] == 0:
                            axes[row_index, col].text(
                                bar_x, 1.6, "0", ha="center", va="bottom",
                                fontsize=6.5, color=palette[method_index],
                            )
        for row_index in range(2):
            ax = axes[row_index, col]
            ax.set_xlim(-0.53, 2.53)
            ax.set_xticks(positions, [str(fleet) for fleet in FLEETS])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, linestyle="-")
            ax.tick_params(length=3, width=0.6)
            ax.text(0.02, 0.96, f"({chr(97 + row_index * 3 + col)})",
                    transform=ax.transAxes, va="top", fontsize=8)
        axes[0, col].set_ylim(0, 112)
        axes[0, col].set_yticks([0, 25, 50, 75, 100])
        axes[1, col].set_ylim(0, 67)
        axes[1, col].set_yticks([0, 15, 30, 45, 60])
        axes[1, col].set_xlabel("Fleet size", labelpad=4)
    axes[0, 0].set_ylabel("Fleet task success (%)", labelpad=5)
    axes[1, 0].set_ylabel("Mean restricted\ncompletion time (s)", labelpad=5)
    handles = [Patch(facecolor=color, edgecolor="none", label=label)
               for color, label in zip(palette, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=4, frameon=False, fontsize=7.5, columnspacing=1.1,
               handlelength=1.15, handletextpad=0.45)
    name = "scenario_results_six_panels" + ("_grayscale" if grayscale else "")
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=600,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/scenario_figures")
    args = parser.parse_args()
    source = args.analysis / "audited_trials.csv"
    cell_source = args.analysis / "descriptive_by_cell.csv"
    rows = summarize(source, cell_source)
    args.output.mkdir(parents=True, exist_ok=True)
    for gray in (False, True):
        plot(rows, args.output, grayscale=gray)
    with (args.output / "scenario_results_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    caption = (
        "Results by highway scenario and fleet size. Columns show S1 reciprocal "
        "exchange, S2 contested merge, and S3 blocked merge. Top panels show fleet "
        "task success; bottom panels show mean restricted completion time. Each "
        "scenario--fleet--method combination contains 21 runs. Restricted time "
        "equals observed completion time for a successful run and 60 s otherwise. "
        "Bars are descriptive summaries without uncertainty intervals."
    )
    (args.output / "caption.txt").write_text(caption + "\n")
    (args.output / "figure_sources.json").write_text(json.dumps({
        "campaign": "carla_conflict_active_confirmatory_v4_20260916",
        "episode_source": str(source),
        "episode_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "crosscheck": str(cell_source),
        "runs": 756,
        "cells": 36,
        "runs_per_cell": 21,
        "restricted_time_failure_penalty_s": HORIZON,
        "operation": "descriptive aggregation of existing episode records",
        "caption": caption,
    }, indent=2) + "\n")
    print(f"Verified all 36 cells against the canonical analysis; saved to {args.output}")


if __name__ == "__main__":
    main()
