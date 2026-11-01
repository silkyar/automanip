"""Deterministic action-preservation comparison, with both strong controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_readout_score import contrast


ROLES = dict(closed_form='base', raw_refinement='search', executed_refinement='unbranched')


def analyze(root):
    root = Path(root); counts, details, work, seconds = {}, {}, 0, []
    for name, role in ROLES.items():
        counts[name], details[name] = [], []
        for index in range(2):
            r = json.loads((root/f'collection-{index}'/role/'report.json').read_text())
            if r['state'] != 'complete' or r['environment_completed'] != 25600 or r['episode_count'] != 512 or r['method'] != role or r['collection_index'] != index:
                raise ValueError('Incomplete or mislabeled assessment')
            if any(e['outcome']['failed'] for e in r['episodes']): raise ValueError('Execution failure')
            counts[name].append([b['successes'] for b in r['batches']])
            stages = {k: sum(e['diagnostics']['ever'][k] for e in r['episodes']) for k in ('grasp', 'lift', 'transport', 'placement', 'release', 'stable_stack_10_steps')}
            details[name].append(dict(parent=index+2, successes=r['successes'], episodes=512,
                                      ever_grasp=r['ever_grasp'], stages=stages, weights_sha256=r['weights_sha256']))
            work += r['environment_completed']; seconds.append(r['wall_seconds'])
    cf = contrast(counts['executed_refinement'], counts['closed_form'], seed=96609999)
    raw = contrast(counts['executed_refinement'], counts['raw_refinement'], seed=96609999)
    retained = all(r['ever_grasp']/512 >= .95 for r in details['executed_refinement'])
    passed = retained and all(r['mean_gain'] >= .10 and r['interval_97_5'][0] > 0 for r in (cf, raw))
    return dict(format='automanip-executed-rehearsal-results-v1', state='complete',
                new_public_transitions=work, optimizer_calls=0, worker_seconds=seconds,
                policies=details, pooled_success={m: float(np.asarray(v).sum()/1024) for m, v in counts.items()},
                executed_minus_closed_form=cf, executed_minus_raw_refinement=raw,
                ever_grasp_retained=retained, primary_screen_passed=bool(passed),
                decision='Local material screen passed; replicate before any promotion' if passed else
                         'Reject material improvement for this fixed objective/learner; retain evidence without score-driven refit',
                scope='Deterministic bounded-action rehearsal on two acquired actors; not a stochastic-policy or from-scratch learning claim')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('policies','worker_seconds')},indent=2))
