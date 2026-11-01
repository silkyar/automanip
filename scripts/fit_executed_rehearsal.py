"""One fixed behavioral-preservation comparison, using only cached fitting rows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.executed_rehearsal import objective
from research.readout_distill import export_readout, features


def fit(parent, method, output):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if parent not in (2, 3) or method not in ('raw_refinement', 'executed_refinement'): raise ValueError('Closed fit')
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source = Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
    with np.load(source, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    with np.load(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz', allow_pickle=False) as z: audit = {k: z[k] for k in z.files}
    t_obs = data['teacher_observations'][audit['teacher_row_indices']]; targets = data['teacher_actions'][audit['teacher_row_indices']]
    r_obs = data['retain_observations'][audit['fitting_episode_indices']].reshape(-1, 48)[audit['rehearsal_row_indices']]
    mask = np.isin(data['teacher_episode'], audit['validation_episode_indices'])
    v_obs, v_target = data['teacher_observations'][mask], data['teacher_actions'][mask]
    vr_obs = data['retain_observations'][audit['validation_episode_indices']].reshape(-1, 48)
    schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
    artifact = load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}', schema)
    base = NumpyPolicy(artifact, schema); w, b = base.layers[-1]; original = np.column_stack([w, b]).T
    tf, rf, vf, vrf = [features(base, x) for x in (t_obs, r_obs, v_obs, vr_obs)]
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    arrays = [tensor(x) for x in (tf, targets, tf@original, rf, rf@original)]
    delta = torch.tensor(audit['delta'], dtype=torch.float64, requires_grad=True)
    initial = delta.detach().numpy().copy()
    executed = method == 'executed_refinement'
    loss = objective(delta, *arrays, executed)
    gradient, = torch.autograd.grad(loss, delta)
    initial_gradient = float(gradient.abs().max())
    optimizer = torch.optim.Adam([delta], lr=.0003)
    losses = []
    for step in range(256):
        loss = objective(delta, *arrays, executed)
        if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in (0, 63, 127, 255): losses.append(dict(step=step+1, objective=float(loss.detach())))
    learned = delta.detach().numpy().copy()
    np.savez_compressed(output/'fit-audit.npz', delta=learned, initial_delta=initial,
                        teacher_row_indices=audit['teacher_row_indices'], rehearsal_row_indices=audit['rehearsal_row_indices'])
    manifest = export_readout(artifact, schema, learned, output/'policy'); updated = NumpyPolicy(load_policy(output/'policy', schema), schema)
    expected = np.clip(vf[:64]@(original+learned), -1, 1)
    parity = float(np.abs(np.asarray(updated.act(v_obs[:64]))-expected).max())
    if parity > 5e-5: raise ValueError('Safe export differs')
    v = np.clip(vf@(original+learned), -1, 1); vr = vrf@(original+learned)
    report = dict(state='complete', parent=parent, method=method, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), adam_steps=256,
        cached_hidden_feature_rows=len(tf)+len(rf)+len(vf)+len(vrf), last_layer_training_rows=256*(len(tf)+len(rf)),
        new_physics=0, initial_gradient_max=initial_gradient, delta_parameter_RMS_change=float(np.sqrt(np.square(learned-initial).mean())),
        validation_teacher_action_MSE=float(np.square(v-v_target).mean()),
        validation_finger_target_RMSE_metres=float(.025*np.sqrt(np.square(v[:, 7]-v_target[:, 7]).mean())),
        validation_positive_increment_fraction=float(((.015+.025*v[:, 7]-v_obs[:, 7])>1e-6).mean()),
        validation_rehearsal_executed_RMS_change=float(np.sqrt(np.square(np.clip(vr, -1, 1)-np.clip(vrf@original, -1, 1)).mean())),
        validation_rehearsal_raw_RMS_change=float(np.sqrt(np.square(vrf@learned).mean())),
        export_parity_max_error=parity, weights_sha256=manifest['weights_sha256'], losses=losses,
        CPU_seconds=time.monotonic()-started)
    (output/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True);p.add_argument('--method',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(fit(a.parent,a.method,a.output),indent=2))
