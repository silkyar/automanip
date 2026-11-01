"""Fixed, registered CPU readout fit on training-only successful continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.readout_distill import export_readout, features, fit_delta


def fit(data_path, parent_path, output, parent, method):
    started = time.monotonic()
    data_path, output = Path(data_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if data_path.stat().st_size > 64*1024**2: raise ValueError('Data size bound')
    with np.load(data_path, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    obs, actions, episode = (data[k] for k in ('teacher_observations', 'teacher_actions', 'teacher_episode'))
    retain = data['retain_observations']
    if retain.ndim != 3 or retain.shape[1:] != (30, 48) or obs.shape != (len(actions), 48) or actions.shape[1:] != (8,) or episode.shape != (len(obs),):
        raise ValueError('Dataset interface differs')
    if not all(np.isfinite(x).all() for x in (obs, actions, retain)): raise ValueError('Nonfinite data')
    rng = np.random.default_rng(96140000+parent)
    order = rng.permutation(len(retain)); fitting = order[:int(.75*len(order))]
    validation = order[int(.75*len(order)):]
    t_indices = np.flatnonzero(np.isin(episode, fitting))
    r_observations = retain[fitting].reshape(-1, 48)
    if len(t_indices) < 384 or len(r_observations) < 1024: raise ValueError('Insufficient registered fitting data')
    t_indices = rng.choice(t_indices, 384, replace=False)
    r_indices = rng.choice(len(r_observations), 1024, replace=False)
    t_obs, target, r_obs = obs[t_indices], actions[t_indices], r_observations[r_indices]
    schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
    artifact = load_policy(parent_path, schema); policy = NumpyPolicy(artifact, schema)
    weight, bias = policy.layers[-1]; parent_readout = np.column_stack([weight, bias])
    tf, rf = features(policy, t_obs), features(policy, r_obs)
    old = tf@parent_readout.T
    delta, report = fit_delta(tf, target, old, rf)
    validation_t = np.flatnonzero(np.isin(episode, validation))
    if len(validation_t): validation_t = rng.choice(validation_t, min(64, len(validation_t)), replace=False)
    validation_r = retain[validation].reshape(-1, 48)
    validation_r = validation_r[rng.choice(len(validation_r), min(256, len(validation_r)), replace=False)]
    vf = features(policy, obs[validation_t]) if len(validation_t) else np.empty((0, tf.shape[1]))
    vr = features(policy, validation_r)
    report.update(parent=parent, method=method, partition='training-only-fit-and-internal-validation',
                  dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  source_episodes=len(retain), successful_teacher_rows=len(obs),
                  fitting_episodes=fitting.tolist(), validation_episodes=validation.tolist(),
                  validation_teacher_rows=len(vf), validation_rehearsal_rows=len(vr),
                  validation_teacher_mse_before=float(np.square(vf@parent_readout.T-actions[validation_t]).mean()) if len(vf) else None,
                  validation_teacher_mse_after=float(np.square(vf@(parent_readout.T+delta)-actions[validation_t]).mean()) if len(vf) else None,
                  validation_rehearsal_raw_change=float(np.square(vr@delta).mean()),
                  fitted_feature_rows=len(tf)+len(rf), validation_feature_rows=len(vf)+len(vr),
                  teacher_gripper_mean_before=float(np.clip(old[:, 7], -1, 1).mean()),
                  teacher_gripper_mean_after=float(np.clip((old+tf@delta)[:, 7], -1, 1).mean()),
                  rehearsal_clipped_action_rms_change=float(np.sqrt(np.square(np.clip(rf@(parent_readout.T+delta), -1, 1)-np.clip(rf@parent_readout.T, -1, 1)).mean())),
                  new_physics=0, optimizer_steps=0, linear_solves=1)
    manifest = export_readout(artifact, schema, delta, output/'policy')
    updated = NumpyPolicy(load_policy(output/'policy', schema), schema)
    check = np.vstack([t_obs[:32], r_obs[:32]])
    expected = np.clip(features(policy, check)@(parent_readout.T+delta), -1, 1)
    error = float(np.abs(np.asarray(updated.act(check))-expected).max())
    if error > 5e-5: raise ValueError('Float32 export parity failed')
    report.update(export_parity_max_error=error, parity_hidden_forward_rows=128,
                  weights_sha256=manifest['weights_sha256'], state='complete', CPU_seconds=time.monotonic()-started)
    (output/'fit-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'fit-audit.npz', delta=delta, teacher_row_indices=t_indices,
                        rehearsal_row_indices=r_indices, fitting_episode_indices=fitting, validation_episode_indices=validation)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--parent-policy', required=True)
    p.add_argument('--output', required=True); p.add_argument('--parent', type=int, choices=[2, 3], required=True)
    p.add_argument('--method', choices=['search', 'unbranched'], required=True)
    a = p.parse_args(); r = fit(a.data, a.parent_policy, a.output, a.parent, a.method)
    print(json.dumps({k: v for k, v in r.items() if k not in ('fitting_episodes', 'validation_episodes')}, indent=2))
