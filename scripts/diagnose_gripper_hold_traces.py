"""Descriptive blind-lane actuator statistics, never inferred grasp labels."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path('artifacts/cumulative/gripper-hold-20261031/training')
    methods = {}
    for index, method in enumerate(('full', 'masked', 'hold')):
        folder = root/f'trial-{index}'
        receipt = json.loads((folder/'receipt.json').read_text())
        trials = []
        for path in sorted((folder/'work/runs'/receipt['run_name']).glob('trace-*.npz'),
                           key=lambda p: int(p.stem.split('-')[-1])):
            with np.load(path, allow_pickle=False) as z:
                observations = z['observations']
                actions = z['executed_actions']
                rewards = z['rewards']
                if observations.shape != (50, 64, 48) or actions.shape != (50, 64, 8):
                    raise ValueError('Blind sample layout changed')
                q = observations[:, :, 7]
                targets = .015+.025*actions[:, :, 7]
                trials.append(dict(steps=int(path.stem.split('-')[-1]),
                    qpos_mean_metres=float(q.mean()), qpos_final_mean_metres=float(q[-1].mean()),
                    absolute_target_mean_metres=float(targets.mean()),
                    target_minus_current_mean_metres=float((targets-q).mean()),
                    gripper_clip_fraction=float((np.abs(actions[:, :, 7]) >= 1).mean()),
                    mean_normalized_dense_reward=float(rewards.mean()),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if len(trials) != 80:
            raise ValueError('Incomplete registered transition grid')
        methods[method] = dict(rollouts=trials,
            early_10_rollouts={k:float(np.mean([r[k] for r in trials[:10]])) for k in trials[0] if k not in ('steps','source_sha256')},
            late_10_rollouts={k:float(np.mean([r[k] for r in trials[-10:]])) for k in trials[0] if k not in ('steps','source_sha256')})
    result = dict(state='complete', new_environment_transitions=0, methods=methods,
        limitation='Descriptive stochastic training sample: first64/1024lanes, all80rollouts, no posthoc outcome selection. Missing contact forces and scored stage diagnostics prevent actual grasp/release attribution. Not policy-only performance.')
    Path('cumulative/results/gripper-hold-traces-20261031.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{k:v for k,v in r.items() if k!='rollouts'} for m,r in methods.items()},indent=2))


if __name__=='__main__':main()
