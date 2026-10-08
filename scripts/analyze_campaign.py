"""Descriptive episode-level reanalysis with recorded 95% CI definitions."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_carla_paired_coordination import audit_method, audit_pairing, read_jsonl, replay_gap_metrics, write_trial_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for block in json.loads((args.runs/'schedule.json').read_text())['blocks']:
        audits=[]
        for method in block['method_order']:
            directory=args.runs/block['block_id']/method
            row=audit_method(directory)
            if not row.get('audit_complete'):raise RuntimeError(f'Incomplete/invalid episode: {directory}')
            trajectory=read_jsonl(directory/'trajectory.jsonl')
            for t in [3,5,7]:row[f'proximity_lt_{t}m_runs']=int(replay_gap_metrics(trajectory,t)[1]>0)
            audits.append(row)
        if not audit_pairing(audits)['paired']:raise RuntimeError(f'Unpaired block: {block["block_id"]}')
        rows.extend(audits)
    write_trial_csv(args.output/'audited_trials.csv',rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['scenario_family'],row['fleet_size'],row['method'])].append(row)
    rng=np.random.default_rng(20261008); summaries=[]
    for (scenario,fleet,method),cell in sorted(groups.items()):
        n=len(cell);times=np.array([r['elapsed_sim_s'] if r['safe_task_success'] else 60 for r in cell])
        lo,hi=np.quantile(times[rng.integers(0,n,size=(10000,n))].mean(axis=1),[.025,.975])
        result=dict(scenario_family=scenario,fleet_size=fleet,method=method,episodes=n,
                    fleet_completion_rate=sum(r['safe_task_success'] for r in cell)/n,
                    restricted_completion_time_mean_s=float(times.mean()),
                    restricted_completion_time_sd_s=float(times.std(ddof=1)) if n>1 else None,
                    time_ci95_low_s=float(lo),time_ci95_high_s=float(hi),
                    completed_maneuvers=sum(r['centered_dwell_completion_count'] for r in cell),
                    planned_maneuvers=sum(r['planned_command_count'] for r in cell))
        counts={'fleet_successes':sum(r['safe_task_success'] for r in cell),'collision_runs':sum(r['collision_count']>0 for r in cell)}
        counts.update({f'proximity_lt_{t}m_runs':sum(r[f'proximity_lt_{t}m_runs'] for r in cell) for t in [3,5,7]})
        for key,count in counts.items():
            z=1.959963984540054;p=count/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
            result.update({key:count,key+'_ci95_low':max(0,c-h),key+'_ci95_high':min(1,c+h)})
        summaries.append(result)
    with (args.output/'descriptive_by_cell.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=summaries[0]);writer.writeheader();writer.writerows(summaries)
    (args.output/'analysis.json').write_text(json.dumps({'episodes':len(rows),'interpretation':'post hoc descriptive reanalysis; not registered primary inference','time_ci':'10000 episode bootstrap percentile resamples within scenario/fleet/method','rate_ci':'95% Wilson score; no multiplicity adjustment','bootstrap_seed':20261008,'singleton_cells':'bootstrap interval is degenerate and not an informative uncertainty estimate'},indent=2)+'\n')
    print(f'Analyzed {len(rows)} episodes')


if __name__=='__main__':main()
