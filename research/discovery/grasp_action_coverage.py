"""One-step command coverage on pretreatment states, never physical returns."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import log_ndtr
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from analyze_randomized_lift import wilson
from panda_kinematics import Kinematics, angle_between, parse_chain, quaternion_matrix
from sac_baseline import Actor


def summary(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)), min=float(np.min(x)), max=float(np.max(x)))


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    base = ROOT/'runs/randomized-lift-20261029'
    trajectory = base/'run/trajectories.npz'
    entry = next(row for row in json.loads((base/'inventory.json').read_bytes()) if row['path'] == 'run/trajectories.npz')
    if hashlib.sha256(trajectory.read_bytes()).hexdigest() != entry['sha256']: raise ValueError('Diagnostic raw data differs')
    state_path = ROOT/'runs/temperature-sufficiency-20261028/run/actor-step-256000-state.npz'
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != 'ed55d938dcf21a5e80a1268921303ebed23d67dbb556b761cdd736f761f4ae6f':
        raise ValueError('Frozen Gaussian state differs')
    with np.load(trajectory, allow_pickle=False) as data:
        lanes = np.flatnonzero(data['eligibility_step'] >= 0)
        times = data['eligibility_step'][lanes]
        observations = data['observations'][times, lanes].astype(np.float64)
    if len(lanes) != 78: raise ValueError('Declared all-eligible set differs')
    actor = Actor().double().eval().requires_grad_(False)
    with np.load(state_path, allow_pickle=False) as saved:
        actor.load_state_dict({name: torch.from_numpy(saved['actor.'+name].copy()).double() for name in actor.state_dict()})
    with torch.no_grad(): mean, logstd = [x.numpy() for x in actor(torch.from_numpy(observations))]
    schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f', 48, 8)
    safe = load_policy(base/'source/inputs/policy', schema)
    if safe.weights_sha256 != '98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a':
        raise ValueError('Safe actor differs')
    safe_action = np.asarray(NumpyPolicy(safe, schema).act(observations))
    parity = float(np.abs(np.tanh(mean)-safe_action).max())
    if parity > 1e-10: raise ValueError('Safe actor parity failed')
    location = Path(importlib.util.find_spec('mani_skill').origin).parent
    urdf = location/'assets/robots/panda/panda_v3.urdf'
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != 'd040277352476639950e2ddce1dff8a1c0e95df470e5e9ed186cd01c468fa412':
        raise ValueError('Public URDF differs')
    chain = parse_chain(urdf)
    kin = Kinematics(chain)
    limits = np.asarray([joint['limits'] for joint in chain if joint['type'] == 'revolute'])
    q = observations[:, :7]
    current = kin.fk(q)
    position_error = float(np.linalg.norm(current[:, :3, 3]-observations[:, 18:21], axis=1).max())
    rotation_error = float(angle_between(current[:, :3, :3], quaternion_matrix(observations[:, 21:25])).max())
    if not np.isfinite([position_error, rotation_error]).all() or position_error > 1e-4 or rotation_error > 1e-3:
        raise ValueError('Current-state analytic FK gate failed')
    sample_count = 512
    std = np.exp(logstd)
    arm_results = {}
    for method, seed in [('current_gaussian', 93314100), ('uniform_reference', 93314200)]:
        generator = np.random.default_rng(seed)
        if method == 'current_gaussian':
            actions = np.tanh(mean[:, None, :7]+std[:, None, :7]*generator.normal(size=(78, sample_count, 7)))
        else:
            actions = generator.uniform(-1, 1, size=(78, sample_count, 7))
        targets = q[:, None, :]+.1*actions
        valid = ((targets >= limits[:, 0]) & (targets <= limits[:, 1])).all(-1)
        future = kin.fk(targets.reshape(-1, 7))
        repeated = np.repeat(current, sample_count, axis=0)
        displacement = (future[:, :3, 3]-repeated[:, :3, 3]).reshape(78, sample_count, 3)
        rotations = angle_between(future[:, :3, :3], repeated[:, :3, :3]).reshape(78, sample_count)
        up = valid & (displacement[:, :, 2] >= .005)
        precise = up & (np.linalg.norm(displacement[:, :, :2], axis=-1) <= .015) & (rotations <= .05)
        arm_results[method] = dict(seed=seed, valid_fraction=valid.mean(1), up_count=up.sum(1),
                                   up_fraction=up.mean(1), secondary_fraction=precise.mean(1))
    rows = []
    for index, lane in enumerate(lanes):
        row = dict(lane=int(lane), eligibility_step=int(times[index]),
                   gripper_latent_mean=float(mean[index, 7]), gripper_latent_std=float(std[index, 7]),
                   deterministic_gripper_action=float(safe_action[index, 7]), arms={}, closing_proxies={})
        for method, info in arm_results.items():
            row['arms'][method] = dict(valid_joint_target_fraction=float(info['valid_fraction'][index]),
                primary_up_count=int(info['up_count'][index]), primary_up_probability=float(info['up_fraction'][index]),
                primary_up_MC_Wilson95=wilson(int(info['up_count'][index]), sample_count),
                secondary_restricted_up_probability=float(info['secondary_fraction'][index]))
        for mm, bound in [(0, -.6), (10, -.2), (20, .2)]:
            cutoff = np.arctanh(bound)
            logp = float(log_ndtr((cutoff-mean[index, 7])/std[index, 7]))
            reflected_logp = float(log_ndtr((cutoff+mean[index, 7])/std[index, 7]))
            probability, reflected = float(np.exp(logp)), float(np.exp(reflected_logp))
            arm_p = float(arm_results['current_gaussian']['up_fraction'][index])
            uniform_arm = float(arm_results['uniform_reference']['up_fraction'][index])
            ci = row['arms']['current_gaussian']['primary_up_MC_Wilson95']
            row['closing_proxies'][str(mm)+'mm'] = dict(normalized_bound=bound,
                exact_probability=probability, exact_log_probability=logp,
                reflected_probability=reflected, reflected_log_probability=reflected_logp,
                numeric_CDF_underflow=probability == 0, uniform_gripper_probability=(bound+1)/2,
                current_joint_proxy_probability=probability*arm_p,
                current_joint_proxy_MC_Wilson95=[probability*endpoint for endpoint in ci],
                reflected_joint_proxy_probability=reflected*arm_p,
                fully_uniform_joint_proxy_probability=(bound+1)/2*uniform_arm)
        rows.append(row)
    summaries = {}
    for key in ('0mm', '10mm', '20mm'):
        summaries[key] = {field: summary([row['closing_proxies'][key][field] for row in rows])
            for field in ('exact_probability', 'exact_log_probability', 'reflected_probability',
                          'current_joint_proxy_probability', 'reflected_joint_proxy_probability',
                          'fully_uniform_joint_proxy_probability')}
    result = dict(state='complete', kind='Frozen one-step setpoint/FK command coverage, not physical success',
        eligible_pretreatment_states=78, summaries=summaries, states=rows, samples_per_state_per_arm_reference=512,
        actor_forward_rows=78, safe_numpy_policy_forward_rows=78, safe_policy_parity_max_abs=parity,
        arm_vectors_sampled=2*78*sample_count, scalar_random_variates=2*78*sample_count*7,
        Gaussian_log_CDF_evaluations=78*3*2, analytic_FK_pose_evaluations=kin.fk_pose_evaluations,
        analytic_jacobian_pose_evaluations=kin.geometric_jacobian_pose_evaluations,
        current_FK_position_max_error_m=position_error, current_FK_rotation_max_error_rad=rotation_error,
        MC_resolution_per_state=1/512, new_simulator_transitions=0, optimizer_calls=0,
        protected_evaluations=0, elapsed_seconds=time.monotonic()-started,
        input_sha256=dict(randomized_trajectory=entry['sha256'], actor_state=hashlib.sha256(state_path.read_bytes()).hexdigest()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['All78pretreatment states; no selection by intervention outcome',
                     'Setpoints are not force/aperture/success conditions; exactboundaryaction−1haszero continuous probability',
                     'Target FK is not actual next-state physics; violating targets are marked, not clipped',
                     'Reflected mean and uniform variance/entropy differ from original; diagnostic references only',
                     'No multiplication acrossfuturesteps or algorithm admission from coverage'])
    destination = ROOT/'research/discovery/grasp-action-coverage-20261029.json'
    if destination.exists(): raise FileExistsError('Preserve prior coverage receipt')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('state', 'eligible_pretreatment_states', 'summaries',
                                                 'analytic_FK_pose_evaluations', 'elapsed_seconds')}))


if __name__ == '__main__': run()
