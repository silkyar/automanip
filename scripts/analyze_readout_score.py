"""Closed-loop public outcomes; distinguish learning from search-specific benefit."""
import argparse
import json
from pathlib import Path

import numpy as np


def contrast(left, right, seed=96209999):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (2, 8) or right.shape != left.shape: raise ValueError('Paired block layout differs')
    gain = (left-right)/64
    indices = np.random.default_rng(seed).integers(0, 8, size=(10000, 2, 8))
    draws = np.take_along_axis(np.broadcast_to(gain, (10000, 2, 8)), indices, axis=2).mean((1, 2))
    return dict(mean_gain=float(gain.mean()), interval_97_5=np.quantile(draws, [.0125, .9875]).tolist())


def analyze(root):
    root = Path(root); counts, grasp, reports, physics, seconds = {}, {}, {}, 0, []
    for method in ('base', 'search', 'unbranched'):
        counts[method], grasp[method], reports[method] = [], [], []
        for index in range(2):
            r = json.loads((root/f'collection-{index}'/method/'report.json').read_text())
            if r['state'] != 'complete' or r['environment_completed'] != 25600 or r['episode_count'] != 512 or r['method'] != method or r['collection_index'] != index:
                raise ValueError('Incomplete or misidentified scorer')
            if any(e['outcome']['failed'] for e in r['episodes']): raise ValueError('Execution failures retained; no complete scientific comparison')
            actual = sum(e['outcome']['terminal_success'] and not e['outcome']['failed'] and
                         (e['outcome']['terminated'] or e['outcome']['truncated']) for e in r['episodes'])
            if actual != r['successes'] or len(r['batches']) != 8: raise ValueError('Outcome accounting differs')
            counts[method].append([b['successes'] for b in r['batches']]); grasp[method].append(r['ever_grasp'])
            reports[method].append(dict(parent=r['parent'], successes=r['successes'], episodes=512,
                                        ever_grasp=r['ever_grasp'], weights_sha256=r['weights_sha256']))
            physics += r['environment_completed']; seconds.append(r['wall_seconds'])
    vs_base, vs_control = contrast(counts['search'], counts['base']), contrast(counts['search'], counts['unbranched'])
    retention = all(x/512 >= .95 for x in grasp['search'])
    learning = vs_base['mean_gain'] >= .10 and vs_base['interval_97_5'][0] > 0 and retention
    search_benefit = learning and vs_control['mean_gain'] >= .05 and vs_control['interval_97_5'][0] > 0
    if search_benefit: decision = 'Local assisted learning and search-specific screen passed; replicate before any promotion or from-scratch claim'
    elif learning: decision = 'Search student improved, but search-specific criterion failed; credit assisted learning only'
    else: decision = 'Fixed readout distillation failed its learning/retention screen; preserve failure, no score-driven refit'
    return dict(format='automanip-readout-score-analysis-v1', state='complete', new_public_transitions=physics,
                optimizer_calls=0, worker_seconds=seconds, policies=reports,
                pooled_success={m: float(np.asarray(v).sum()/1024) for m, v in counts.items()},
                search_minus_base=vs_base, search_minus_unbranched=vs_control,
                search_grasp_retention_passed=retention, learning_screen_passed=bool(learning),
                search_specific_screen_passed=bool(search_benefit), decision=decision,
                scope='Two acquired fixed parent policies, one fixed readout fit per method, fresh public policy-only episodes; not from-scratch or held-task validation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); r = analyze(a.root); Path(a.output).write_text(json.dumps(r, indent=2)+'\n')
    print(json.dumps({k: v for k, v in r.items() if k not in ('policies', 'worker_seconds')}, indent=2))
