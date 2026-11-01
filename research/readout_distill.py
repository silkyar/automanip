"""One fixed supervised readout fit with early-behavior rehearsal.

This is an assisted policy adaptation probe, not a new from-scratch RL algorithm.
No outcome, clock or corrective controller is present in the exported MLP.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


REHEARSAL_WEIGHT = 4.
RIDGE = .001


def features(policy, observations):
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != policy.layers[0][0].shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid feature observations')
    for weight, bias in policy.layers[:-1]:
        x = np.tanh(x @ weight.T+bias)
    return np.column_stack([x, np.ones(len(x))])


def fit_delta(teacher_features, teacher_actions, parent_mean, retain_features):
    t, y, old, r = [np.asarray(x, dtype=np.float64) for x in
                    (teacher_features, teacher_actions, parent_mean, retain_features)]
    if t.ndim != 2 or r.ndim != 2 or t.shape[1] != r.shape[1] or not len(t) or not len(r) or y.ndim != 2 or y.shape != old.shape or len(y) != len(t):
        raise ValueError('Invalid fit interface')
    if not all(np.isfinite(x).all() for x in (t, y, old, r)):
        raise ValueError('Nonfinite fit data')
    gram = t.T@t/len(t)+REHEARSAL_WEIGHT*(r.T@r)/len(r)+RIDGE*np.eye(t.shape[1])
    target = t.T@(y-old)/len(t)
    delta = np.linalg.solve(gram, target)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite fitted readout')
    return delta, dict(teacher_rows=len(t), retain_rows=len(r), linear_solves=1,
                       teacher_mse_before=float(np.square(old-y).mean()),
                       teacher_mse_after=float(np.square(old+t@delta-y).mean()),
                       retain_raw_mean_square_change=float(np.square(r@delta).mean()),
                       normal_equation_max_residual=float(np.abs(gram@delta-target).max()),
                       rehearsal_weight=REHEARSAL_WEIGHT, ridge=RIDGE)


def export_readout(artifact, schema, delta, directory):
    """Write only the existing bounded JSON MLP format; never native checkpoints."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    dims = (schema.observation_dim, *artifact.hidden_sizes, schema.action_dim)
    last = len(dims)-2
    if np.shape(delta) != (dims[-2]+1, dims[-1]) or not np.isfinite(delta).all():
        raise ValueError('Readout shape differs')
    tensors = {}
    for i, (left, right) in enumerate(zip(dims, dims[1:])):
        for kind, shape in [('weight', (right, left)), ('bias', (right,))]:
            key = f'layers.{i}.{kind}'
            array = np.asarray(artifact.tensors[key], dtype=np.float64).reshape(shape).copy()
            if i == last: array += delta[:-1].T if kind == 'weight' else delta[-1]
            array = array.astype(np.float32)
            if not np.isfinite(array).all(): raise ValueError('Float32 export overflow')
            tensors[key] = dict(shape=list(shape), values=array.reshape(-1).tolist())
    raw = json.dumps(tensors, separators=(',', ':'), allow_nan=False).encode()
    (directory/'weights.json').write_bytes(raw)
    manifest = dict(format='automanip-json-f32-v1', task_fingerprint=schema.task_fingerprint,
                    observation_dim=schema.observation_dim, action_dim=schema.action_dim,
                    hidden_sizes=list(artifact.hidden_sizes), activation='tanh',
                    weights_sha256=hashlib.sha256(raw).hexdigest())
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
