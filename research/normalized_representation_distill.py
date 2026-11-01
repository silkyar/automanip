"""Fixed full-MLP distillation with a training-only actuator-coordinate task."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, ValidatedPolicy, load_policy
from automanip.stackcube import STACK_TASK
from research.readout_distill import export_readout, features
from research.relative_gripper_search import inverse_target
from research.prefix_search import raw_mean
from research.affine_student import AffineStudent, export_affine_student


class Student(nn.Module):
    def __init__(self, artifact, observation_dim=48, action_dim=8):
        super().__init__()
        dims = (observation_dim, *artifact.hidden_sizes, action_dim)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims, dims[1:])])
        self.coordinate = nn.Linear(dims[-2], 1)
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                layer.weight.copy_(torch.tensor(artifact.tensors[f'layers.{i}.weight']).reshape_as(layer.weight))
                layer.bias.copy_(torch.tensor(artifact.tensors[f'layers.{i}.bias']))
    def forward(self, x):
        for layer in self.layers[:-1]: x = torch.tanh(layer(x))
        return self.layers[-1](x), self.coordinate(x).squeeze(-1)


def export_student(model, artifact, schema, directory):
    tensors = {}
    for i, layer in enumerate(model.layers):
        for name, tensor in [('weight', layer.weight), ('bias', layer.bias)]:
            tensors[f'layers.{i}.{name}'] = tuple(tensor.detach().cpu().numpy().reshape(-1).tolist())
    acting = ValidatedPolicy(artifact.hidden_sizes, tensors, 'new-numeric-weights')
    return export_readout(acting, schema, np.zeros((artifact.hidden_sizes[-1]+1, schema.action_dim)), directory)


def fit(parent, method, output):
    if parent not in (2, 3) or method != 'normalized_aux': raise ValueError('Closed method')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(96340000+parent)
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    path = Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
    audit_path = Path(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz')
    with np.load(path, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    with np.load(audit_path, allow_pickle=False) as z: audit = {k: z[k] for k in z.files}
    t_obs = data['teacher_observations'][audit['teacher_row_indices']]
    targets = data['teacher_actions'][audit['teacher_row_indices']]
    r_obs = data['retain_observations'][audit['fitting_episode_indices']].reshape(-1, 48)[audit['rehearsal_row_indices']]
    mask = np.isin(data['teacher_episode'], audit['validation_episode_indices'])
    v_obs, v_target = data['teacher_observations'][mask], data['teacher_actions'][mask]
    vr_obs = data['retain_observations'][audit['validation_episode_indices']].reshape(-1, 48)
    schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
    artifact = load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}', schema)
    base = NumpyPolicy(artifact, schema)
    fitting_observations = np.vstack([t_obs, r_obs]).astype(np.float64)
    mean = fitting_observations.mean(0); scale = np.maximum(fitting_observations.std(0), .001)
    model = AffineStudent(artifact, mean, scale)
    with torch.no_grad():
        initial_error = float(np.abs(model(torch.tensor(fitting_observations[:64],dtype=torch.float32))[0].numpy()-raw_mean(base,fitting_observations[:64])).max())
    if initial_error > 5e-5: raise ValueError('Initial function preservation failed')
    if method == 'frozen_mlp':
        for layer in model.layers[:-1]:
            for parameter in layer.parameters(): parameter.requires_grad_(False)
    combined = np.vstack([t_obs, r_obs]); phi = features(base, combined)
    coefficient = np.linalg.solve(phi.T@phi/len(phi)+.001*np.eye(phi.shape[1]),
                                  phi.T@inverse_target(combined[:, 7])/len(phi))
    with torch.no_grad():
        model.coordinate.weight.copy_(torch.tensor(coefficient[:-1], dtype=torch.float32)[None])
        model.coordinate.bias.copy_(torch.tensor(coefficient[-1:], dtype=torch.float32))
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    t, y, r = tensor(t_obs), tensor(targets), tensor(r_obs)
    with torch.no_grad(): old_r = model(r)[0].detach().clone()
    parameters = list(model.layers.parameters()); original = [p.detach().clone() for p in model.raw_parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    generator = torch.Generator().manual_seed(96340000+parent)
    auxiliary_weight = 1.
    losses = []
    for step in range(256):
        ti = torch.randint(len(t), (128,), generator=generator); ri = torch.randint(len(r), (128,), generator=generator)
        x = torch.cat([t[ti], r[ri]])
        action, coordinate = model(x)
        teacher = (action[:128]-y[ti]).square().mean()
        rehearsal = (action[128:]-old_r[ri]).square().mean()
        anchor = sum((p-before).square().sum() for p, before in zip(model.raw_parameters(), original))
        auxiliary = (coordinate-(x[:, 7]-.015)/.025).square().mean()
        loss = teacher+4*rehearsal+.001*anchor+auxiliary_weight*auxiliary
        if not torch.isfinite(loss): raise ValueError('Nonfinite learner')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step in (0, 63, 127, 255): losses.append(dict(step=step+1, teacher=float(teacher.detach()),
            rehearsal=float(rehearsal.detach()), auxiliary=float(auxiliary.detach())))
        if time.monotonic()-started > 120: raise TimeoutError('Fixed CPU model deadline')
    # Preserve completed numeric work before diagnostics/export, so a reporting
    # error cannot silently force another optimizer run. Never native pickle.
    arrays = {f'layer_{i}_{name}': tensor.detach().numpy().copy()
              for i, layer in enumerate(model.layers)
              for name, tensor in [('weight', layer.weight), ('bias', layer.bias)]}
    arrays.update(coordinate_weight=model.coordinate.weight.detach().numpy().copy(),
                  coordinate_bias=model.coordinate.bias.detach().numpy().copy())
    np.savez_compressed(output/'trained-arrays.npz', **arrays)
    (output/'training-complete.json').write_text(json.dumps(dict(parent=parent,method=method,adam_steps=256,
        training_forward_rows=65536,losses=losses,elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    with torch.no_grad():
        va, vq = model(tensor(v_obs)); vr, _ = model(tensor(vr_obs))
    va, vq, vr = va.numpy(), vq.numpy(), vr.numpy()
    old_vr = raw_mean(base, vr_obs)
    gap = .015+.025*np.clip(va[:, 7], -1, 1)-v_obs[:, 7]
    metrics = dict(validation_teacher_mse=float(np.square(va-v_target).mean()),
                   validation_finger_target_RMSE_metres=float(.025*np.sqrt(np.square(np.clip(va[:, 7], -1, 1)-v_target[:, 7]).mean())),
                   validation_coordinate_RMSE_metres=float(.025*np.sqrt(np.square(vq-inverse_target(v_obs[:, 7])).mean())),
                   validation_positive_increment_fraction=float((gap>1e-6).mean()),
                   validation_rehearsal_clipped_RMS_change=float(np.sqrt(np.square(np.clip(vr, -1, 1)-np.clip(old_vr, -1, 1)).mean())))
    manifest = export_affine_student(model, artifact, schema, output/'policy')
    exported = NumpyPolicy(load_policy(output/'policy', schema), schema)
    check = np.vstack([t_obs[:32], r_obs[:32]])
    with torch.no_grad(): expected = model(tensor(check))[0].clamp(-1, 1).numpy()
    parity = float(np.abs(np.asarray(exported.act(check))-expected).max())
    if parity > 5e-5: raise ValueError('Safe export parity failed')
    report = dict(state='complete', parent=parent, method=method, adam_steps=256,
        batch_teacher=128, batch_rehearsal=128, training_forward_rows=65536,
        old_reference_rows=1024, ridge_feature_rows=len(phi), linear_solves=1,
        validation_rows=len(v_obs)+len(vr_obs), old_validation_feature_rows=len(vr_obs),
        parity_forward_rows=128, backward_passes=256, auxiliary_weight=auxiliary_weight,
        acting_trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
        auxiliary_in_exported_artifact=False, weights_sha256=manifest['weights_sha256'],
        initial_raw_output_max_error=initial_error, normalization_mean=mean.tolist(), normalization_scale=scale.tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), fit_row_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        new_physics=0, losses=losses, metrics=metrics, export_parity_max_error=parity,
        CPU_seconds=time.monotonic()-started)
    (output/'fit-report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--parent', type=int, required=True); p.add_argument('--method', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); print(json.dumps(fit(a.parent, a.method, a.output), indent=2))
