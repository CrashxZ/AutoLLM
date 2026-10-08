"""Local processing benchmark; not a network/deployment latency measurement."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from scripts.generate_candidate_ranker_dataset import build_scene
from server.coordination.audit import AuditLog
from server.coordination.candidate_ranking import RankedCandidateProposer
from server.coordination.orchestrator import CoordinationOrchestrator
from server.coordination.transaction import ActivePlanStore
from server.coordination.validator import DeterministicPlanValidator

OUT = ROOT / 'results/mec_latency_20261008'
STAGES = ('generation', 'validation', 'ranking', 'recording')


class Meter:
    """Exclusive nested timings prevent validation from being counted twice."""
    def __init__(self):
        self.stack = []
        self.reset()

    def reset(self):
        assert not self.stack
        self.values = defaultdict(float)

    def enter(self):
        frame = [time.perf_counter_ns(), 0]
        self.stack.append(frame)
        return frame

    def leave(self, frame, name):
        elapsed = time.perf_counter_ns() - frame[0]
        assert self.stack.pop() is frame
        self.values[name] += (elapsed - frame[1]) / 1e6
        if self.stack:
            self.stack[-1][1] += elapsed

    def wrap(self, obj, attribute, name, asynchronous=False):
        original = getattr(obj, attribute)
        if asynchronous:
            async def wrapped(*args, **kwargs):
                frame = self.enter()
                try:
                    return await original(*args, **kwargs)
                finally:
                    self.leave(frame, name)
        else:
            def wrapped(*args, **kwargs):
                frame = self.enter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.leave(frame, name)
        setattr(obj, attribute, wrapped)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def main():
    global OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mec_latency')
    OUT = parser.parse_args().output
    OUT.mkdir(parents=True, exist_ok=False)
    geometries = ('side_by_side', 'front_blocker', 'rear_blocker', 'multi_conflict', 'sparse')
    specs = [dict(scene_id=f'local-n{fleet}-{index:02d}', fleet_size=fleet,
                  geometry=geometry, speed_stratum=speed, seed=2026100800 + fleet * 100 + index,
                  ordinal=fleet * 100 + index)
             for fleet in (2, 4, 8)
             for index, (geometry, speed) in enumerate((g, s) for g in geometries for s in ('near_flow', 'mixed_closing'))]
    model = V2 / 'models/candidate_ranker.npz'
    configuration = {
        'claim_status': 'exploratory_local_microbenchmark',
        'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'scenarios_per_fleet': 10, 'timed_repetitions_per_case_variant': 30,
        'warmup_repetitions_per_case_variant': 3, 'scene_specs': specs,
        'hardware': subprocess.check_output(['lscpu'], text=True),
        'platform': platform.platform(), 'python': sys.version,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=V2, text=True).strip(),
        'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=V2)).hexdigest(),
        'ranker_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'validation': asdict(DeterministicPlanValidator().config),
        'persistence': 'Append-only AuditLog and complete decision JSONL, OS-buffered writes, no fsync',
        'scope': 'Typed inputs and model loaded before timing; fresh empty commitment store per review; no network, perception, dispatch, execution, cold start, queueing, or crash-durable disk flush',
        'workload': 'Ten generated kinematic traffic configurations per fleet; not CARLA episodes or ten new scenario families',
    }
    (OUT / 'configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')
    rows, decisions = [], []
    order = random.Random(20261008)
    for spec in specs:
        proposal, states = build_scene(spec)
        (OUT / (spec['scene_id'] + '.json')).write_text(json.dumps({
            'spec': spec, 'proposal': proposal.model_dump(mode='json'),
            'states': {str(k): v.model_dump(mode='json') for k, v in states.items()},
        }, indent=2) + '\n')
        variants = ['deterministic', 'learned']
        order.shuffle(variants)
        for variant in variants:
            validator = DeterministicPlanValidator()
            proposer = RankedCandidateProposer(
                validator=validator, model_path=str(model) if variant == 'learned' else None,
                enable_liveness_preparation=True, enable_global_conflict_recovery=True,
            )
            if variant == 'learned' and proposer.model is None:
                raise RuntimeError('Learned model could not load')
            meter = Meter()
            meter.wrap(validator, 'validate', 'validation')
            meter.wrap(proposer.generator, 'generate', 'generation')
            meter.wrap(proposer.extractor, 'evaluate', 'ranking')
            meter.wrap(proposer, 'propose', 'ranking', asynchronous=True)
            for repeat in range(-3, 30):
                audit = AuditLog(str(OUT / 'transaction_audit.jsonl'))
                meter.wrap(audit, 'append', 'recording')
                coordinator = CoordinationOrchestrator(
                    validator=validator, proposer=proposer, store=ActivePlanStore(audit=audit),
                )
                meter.reset()
                started = time.perf_counter_ns()
                decision = await coordinator.review(proposal, states, now_s=proposal.created_at_s)
                frame = meter.enter()
                with (OUT / 'decisions.jsonl').open('a') as stream:
                    stream.write(decision.model_dump_json(exclude_none=True) + '\n')
                meter.leave(frame, 'recording')
                total = (time.perf_counter_ns() - started) / 1e6
                if repeat < 0:
                    continue
                stage = {k + '_ms': meter.values[k] for k in STAGES}
                other = total - sum(stage.values())
                assert other >= 0, (total, stage)
                rows.append({
                    'scene_id': spec['scene_id'], 'fleet_size': spec['fleet_size'],
                    'geometry': spec['geometry'], 'speed_stratum': spec['speed_stratum'],
                    'variant': variant, 'repeat': repeat, 'decision': decision.decision.value,
                    'candidate_count': len(proposer.last_evaluations), **stage,
                    'other_ms': other, 'total_ms': total,
                })
            print(spec['scene_id'], variant, decision.decision.value, flush=True)
    assert len(rows) == 1800
    write_csv(OUT / 'timings_raw.csv', rows)
    summaries = []
    rng = np.random.default_rng(20261008)
    for fleet in (2, 4, 8):
        for variant in ('deterministic', 'learned'):
            cell = [r for r in rows if r['fleet_size'] == fleet and r['variant'] == variant]
            scenes = sorted({r['scene_id'] for r in cell})
            assert len(scenes) == 10 and len(cell) == 300
            totals = np.array([np.mean([r['total_ms'] for r in cell if r['scene_id'] == scene]) for scene in scenes])
            means = totals[rng.integers(0, 10, size=(10000, 10))].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            summary = dict(fleet_size=fleet, variant=variant, scenarios=10, reviews=300)
            for key in [k + '_ms' for k in STAGES] + ['other_ms', 'total_ms']:
                summary[key] = float(np.mean([r[key] for r in cell]))
            summary.update(total_case_sd_ms=float(totals.std(ddof=1)), total_ci95_low_ms=float(low), total_ci95_high_ms=float(high), total_p95_ms=float(np.percentile([r['total_ms'] for r in cell], 95)), total_max_ms=max(r['total_ms'] for r in cell), over_500ms=sum(r['total_ms'] > 500 for r in cell))
            for kind in ('ACK', 'PLAN', 'NACK'):
                summary[kind + '_cases'] = sum(next(r for r in cell if r['scene_id'] == scene)['decision'].upper() == kind for scene in scenes)
            summaries.append(summary)
    write_csv(OUT / 'latency_summary.csv', summaries)
    lines = ['# MEC review processing latency', '', 'Local measurements on Intel Core i7-8700K (6 cores / 12 logical CPUs), using the production coordinator and corrected candidate configuration. Ten generated traffic configurations per fleet size (five geometries × two speed strata), with 3 warm-up and 30 measured reviews per case and variant. These are synthetic kinematic inputs, not CARLA recordings.', '', '| Vehicles | Configuration | Generation (ms) | Validation (ms) | Features + ranking (ms) | Recording (ms) | Other (ms) | Total mean ± case SD (ms) [95% CI] | P95 (ms) |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    tex = [r'\begin{tabular}{rlrrrrrr}', r'\hline', r'Vehicles & Ranker & Generate & Validate & Rank & Record & Total & P95 \\', r'\hline']
    for r in summaries:
        lines.append(f"| {r['fleet_size']} | {r['variant']} | {r['generation_ms']:.3f} | {r['validation_ms']:.3f} | {r['ranking_ms']:.3f} | {r['recording_ms']:.3f} | {r['other_ms']:.3f} | {r['total_ms']:.3f} ± {r['total_case_sd_ms']:.3f} [{r['total_ci95_low_ms']:.3f}, {r['total_ci95_high_ms']:.3f}] | {r['total_p95_ms']:.3f} |")
        tex.append(f"{r['fleet_size']} & {r['variant']} & {r['generation_ms']:.3f} & {r['validation_ms']:.3f} & {r['ranking_ms']:.3f} & {r['recording_ms']:.3f} & {r['total_ms']:.3f} & {r['total_p95_ms']:.3f}" + r' \\')
    tex += [r'\hline', r'\end{tabular}']
    lines += ['', 'Validation includes submitted-plan checks, candidate validation and final revalidation. Features + ranking includes rollout features, fixed-cost scoring, optional MLP inference, priority sorting and selection. Recording includes transaction-audit serialization/appends and a complete decision JSONL append; OS-buffered writes are not crash-durable fsync. Other covers remaining orchestration and in-memory transaction/commit work. Exclusive nested instrumentation prevents double counting.', '', 'Means average 300 reviews across 10 equally weighted cases. SD is across the 10 case means; 95% percentile bootstrap intervals resample case means (10,000 resamples), not 300 repetitions as independent traffic cases. P95 pools 300 timed invocations and is descriptive. ACK fast paths have zero generation/ranking time; those zeroes remain included.', '', 'This table measures warm, single-request local processing with an empty active-plan store per review. Model loading and input parsing are outside the timer. Network/V2X delays, queueing, perception, command delivery, physical execution and durable storage flushes are excluded. CARLA/backend may be resident on the same workstation; no claim of dedicated MEC hardware or end-to-end real-time guarantees is made.', '', 'Reproduce: `python scripts/benchmark_mec_review_stages.py` (output directory must not already exist). Raw timing records, configurations, inputs, model/source hashes, decisions and audit logs are retained in this folder.']
    (OUT / 'latency_table.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'latency_table.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    asyncio.run(main())
