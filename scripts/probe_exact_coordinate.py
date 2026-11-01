"""Test coordinate availability with existing data; no actor export or physics."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.prefix_search import raw_mean
from research.readout_distill import features, fit_delta


def augment(phi, observations):
    return np.column_stack([phi, (observations[:, 7]-.015)/.025])


def probe():
    started = time.process_time()
    rows = []
    for parent in (2, 3):
        path = Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
        audit_path = Path(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz')
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        with np.load(audit_path, allow_pickle=False) as z:
            audit = {k: z[k] for k in z.files}
        train = data['teacher_observations'][audit['teacher_row_indices']]
        target = data['teacher_actions'][audit['teacher_row_indices']]
        retain = data['retain_observations'][audit['fitting_episode_indices']].reshape(-1, 48)[audit['rehearsal_row_indices']]
        mask = np.isin(data['teacher_episode'], audit['validation_episode_indices'])
        validation = data['teacher_observations'][mask]
        validation_target = data['teacher_actions'][mask]
        vr = data['retain_observations'][audit['validation_episode_indices']].reshape(-1, 48)
        schema = PolicySchema(STACK_TASK.fingerprint,48,8)
        base = NumpyPolicy(load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}', schema), schema)
        t, r, v, memory = [features(base, x) for x in (train, retain, validation, vr)]
        old_t, old_v, old_memory = [raw_mean(base, x) for x in (train, validation, vr)]
        if audit['delta'].shape != (257, 8):
            raise ValueError('Original fitted delta missing')
        delta, fit = fit_delta(augment(t, train), target, old_t, augment(r, retain))
        methods = {}
        for name, prediction, retention in [
            ('original', old_v+v@audit['delta'], old_memory+memory@audit['delta']),
            ('exact_coordinate', old_v+augment(v, validation)@delta,
             old_memory+augment(memory, vr)@delta)]:
            gap = .015+.025*np.clip(prediction[:, 7], -1, 1)-validation[:, 7]
            methods[name] = dict(
                finger_target_RMSE_metres=float(.025*np.sqrt(np.square(np.clip(prediction[:, 7], -1, 1)-validation_target[:, 7]).mean())),
                teacher_raw_MSE=float(np.square(prediction-validation_target).mean()),
                retained_executed_RMS_change=float(np.sqrt(np.square(np.clip(retention, -1, 1)-np.clip(old_memory, -1, 1)).mean())),
                positive_increment_fraction=float((gap>1e-6).mean()))
        old, new = methods['original'], methods['exact_coordinate']
        passed = (new['finger_target_RMSE_metres'] < .001
                  and new['finger_target_RMSE_metres'] < .5*old['finger_target_RMSE_metres']
                  and new['retained_executed_RMS_change'] <= old['retained_executed_RMS_change'])
        rows.append(dict(parent=parent,methods=methods,passed=passed,fit=fit,
            exact_coordinate_coefficient=delta[-1].tolist(),teacher_fit_rows=len(train),rehearsal_fit_rows=len(retain),
            validation_rows=len(validation),validation_rehearsal_rows=len(vr),
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest()))
    return dict(state='complete', passed=all(r['passed'] for r in rows), parents=rows,
        CPU_seconds=time.process_time()-started, linear_solves=2, optimizer_calls=0,
        new_environment_transitions=0,policy_exports=0,robot_assessments=0,
        scope='Exact observed coordinate sufficiency under fixed assisted readout objective; adaptively reused diagnostic corpus, not a fresh generalization or from-scratch test')


if __name__=='__main__':
    result=probe()
    with Path('cumulative/results/exact-coordinate-CPU-20261031.json').open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps(result,indent=2))
