"""Fixed endpoint test of action preservation versus actual skill retention."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired(left, right):
    difference = (np.asarray(left)-np.asarray(right))/64
    if difference.shape != (2, 4): raise ValueError('Endpoint block layout differs')
    indices = np.random.default_rng(96509999).integers(0, 4, size=(10000, 2, 4))
    draws = np.take_along_axis(np.broadcast_to(difference, (10000, 2, 4)), indices, axis=2).mean((1, 2))
    return dict(mean_gain=float(difference.mean()), interval_97_5=np.quantile(draws, [.0125, .9875]).tolist())


def analyze(root):
    root = Path(root); success, grasp, details, work, seconds = {}, {}, {}, 0, []
    for method in ('base', 'rehearsal', 'teacher_only'):
        success[method], grasp[method], details[method] = [], [], []
        for index in range(2):
            r = json.loads((root/f'collection-{index}'/method/'report.json').read_text())
            if r['state'] != 'complete' or r['environment_completed'] != 12800 or r['episode_count'] != 256 or len(r['batches']) != 4:
                raise ValueError('Incomplete endpoint')
            if r['collection_index'] != index or r['method'] != method or any(e['outcome']['failed'] for e in r['episodes']):
                raise ValueError('Endpoint identity or execution failure')
            success[method].append([b['successes'] for b in r['batches']])
            grasp[method].append([b['ever_grasp'] for b in r['batches']])
            stages = {k: sum(e['diagnostics']['ever'][k] for e in r['episodes']) for k in ('grasp', 'lift', 'transport', 'placement', 'release', 'stable_stack_10_steps')}
            details[method].append(dict(parent=index+2, successes=r['successes'], ever_grasp=r['ever_grasp'], episodes=256, stages=stages))
            work += r['environment_completed']; seconds.append(r['wall_seconds'])
    sg = paired(success['teacher_only'], success['rehearsal']); gg = paired(grasp['teacher_only'], grasp['rehearsal'])
    retained = all(sum(x)/256 >= .95 for x in grasp['teacher_only'])
    overconstraint = retained and sg['mean_gain'] >= .10 and sg['interval_97_5'][0] > 0
    forgetting = gg['mean_gain'] <= -.20 and gg['interval_97_5'][1] < 0 and sg['mean_gain'] <= 0
    conclusion = ('Action rehearsal overconstrained this local correction while actual grasping was retained' if overconstraint else
                  'Removing rehearsal caused major actual grasp loss without success gain in this fixed learner' if forgetting else
                  'Endpoint evidence does not establish either preregistered causal explanation')
    return dict(format='automanip-rehearsal-endpoint-analysis-v1', state='complete', new_public_transitions=work,
                optimizer_calls=0, worker_seconds=seconds, policies=details,
                pooled_success={m: float(np.asarray(v).sum()/512) for m, v in success.items()},
                teacher_only_minus_rehearsal_success=sg, teacher_only_minus_rehearsal_grasp=gg,
                teacher_only_grasp_retained=retained, overconstraint_screen=bool(overconstraint),
                harmful_forgetting_screen=bool(forgetting), conclusion=conclusion,
                scope='Fixed rehearsal endpoints on two acquired actors; mechanism probe, not an optimized learner or general continual-learning result')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('policies','worker_seconds')},indent=2))
