"""Run explicitly versioned corrected-configuration CARLA comparisons."""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from marl.numpy_policy import NumpyActorPolicy
from scripts.run_carla_conflict_active_pilot import configure_ranker
from scripts.run_carla_paired_coordination import RunnerBudget, run_method_episode
from scripts.analyze_carla_paired_coordination import audit_method, audit_pairing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['smoke', 'eight', 'full'], default='smoke')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-base', default='http://127.0.0.1:8000')
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    schedule = json.loads((ROOT / 'experiments/historical/carla_v4_schedule.json').read_text())
    blocks = [dict(b, claim_status='post_hoc_corrected_implementation_followup')
              for b in schedule['blocks']
              if (args.preset != 'eight' or int(b['fleet_size']) == 8)
              and (args.preset != 'smoke' or int(b['repetition']) == 4)]
    if args.max_blocks is not None:
        if args.max_blocks < 1:
            parser.error('--max-blocks must be positive')
        blocks = blocks[:args.max_blocks]
    print(json.dumps({'preset': args.preset, 'blocks': len(blocks),
                      'episodes': sum(len(b['method_order']) for b in blocks),
                      'claim_status': 'post_hoc_corrected_implementation_followup'}))
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    budget = RunnerBudget()
    actor = NumpyActorPolicy.load(str(ROOT / 'models/mappo_actor.npz'))
    with httpx.Client(timeout=120) as client:
        if client.get(args.api_base + '/health').json().get('running'):
            raise RuntimeError('Backend is busy')
        previous = client.get(args.api_base + '/mec/v2/status').json()
        if not previous.get('liveness_preparation') or not previous.get('global_conflict_recovery') or previous.get('execution_speed_prediction_v5'):
            raise RuntimeError('Backend configuration differs from corrected_followup.json')
        selected = dict(schedule, blocks=blocks, block_count=len(blocks),
                        episode_count=4*len(blocks), claim_status='post_hoc_corrected_implementation_followup')
        (args.output / 'schedule.json').write_text(json.dumps(selected, indent=2)+'\n')
        (args.output / 'manifest.json').write_text(json.dumps({
            'claim_status': selected['claim_status'], 'budget': asdict(budget),
            'configuration': json.loads((ROOT/'configs/corrected_followup.json').read_text()),
            'api_status': previous,
            'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'models': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('*.npz')},
        }, indent=2)+'\n')
        try:
            for block in blocks:
                audits = []
                for method in block['method_order']:
                    print(block['block_id'], method, flush=True)
                    configure_ranker(client, args.api_base, method)
                    destination = args.output / block['block_id'] / method
                    run_method_episode(client, args.api_base, block, method, destination,
                                       actor if method == 'MAPPO_ADAPTED' else None,
                                       budget, archive=False, enable_cameras=False)
                    audit = audit_method(destination)
                    if not audit.get('audit_complete'):
                        raise RuntimeError(f'Invalid episode records: {destination}')
                    audits.append(audit)
                if not audit_pairing(audits)['paired']:
                    raise RuntimeError(f'Pairing mismatch: {block["block_id"]}')
        finally:
            variant = 'learned' if previous.get('ranker_model_loaded') else 'deterministic'
            client.post(args.api_base+'/mec/v2/ranker',json={'variant':variant}).raise_for_status()


if __name__ == '__main__':
    main()
