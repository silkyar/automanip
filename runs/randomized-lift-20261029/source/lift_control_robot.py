"""Preregistered assisted 2x2 mechanical probe; no learning or benchmark scorer."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from lift_probe_control import upward_action
from panda_kinematics import Kinematics, angle_between, parse_chain, quaternion_matrix

PINS = {'mani_skill': '3.0.0b21', 'sapien': '3.0.0b1', 'torch': '2.6.0+cu124',
        'numpy': '1.26.4', 'gymnasium': '0.29.1'}
TASK_SHA = 'ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440'
URDF_SHA = 'd040277352476639950e2ddce1dff8a1c0e95df470e5e9ed186cd01c468fa412'
FINGERPRINT = 'b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f'
POLICY_SHA = '98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def match_residual(left, right):
    """Per-lane measured-state checks, not hidden solver-state equality."""
    positions = np.concatenate([np.arange(a, a+3) for a in (18, 25, 32)])
    rotation = np.stack([angle_between(quaternion_matrix(left[:, a:a+4]),
                                      quaternion_matrix(right[:, a:a+4]))
                         for a in (21, 28, 35)], axis=1).max(1)
    return dict(position_m=np.abs(left[:, positions]-right[:, positions]).max(1).tolist(),
                qpos=np.abs(left[:, :9]-right[:, :9]).max(1).tolist(),
                qvel=np.abs(left[:, 9:18]-right[:, 9:18]).max(1).tolist(),
                rotation_rad=rotation.tolist())


def require_match(residual):
    limits = dict(position_m=1e-4, qpos=1e-4, qvel=1e-2, rotation_rad=1e-3)
    return all(max(residual[key], default=0) <= limit for key, limit in limits.items())


def run(inputs, output):
    started = time.monotonic()
    deadline = min(started+270, float(os.environ['AUTOMANIP_DIAGNOSTIC_DEADLINE']))
    output.mkdir(exist_ok=False)
    report = dict(state='started', kind='Assisted mechanical diagnostic, not a learned method',
                  actual_public_transitions=0, attempted_public_transitions=0, vector_step_calls=0,
                  explicit_vector_reset_calls=0, optimizer_calls=0, protected_evaluations=0,
                  new_training_runs=0, policy_inference_rows=0, inverse_kinematics_pose_evaluations=0,
                  assistance=dict(own_policy_training_transitions=256000,
                                  own_policy_training_seconds=195.434352986, public_URDF=True),
                  conditions=[], validity_passed=True,
                  versions={k: importlib.metadata.version(k) for k in PINS})
    env = None
    kinematics = None
    eligibility = np.full(8, -1, dtype=int)
    baseline = None
    write(output/'report.json', report)
    try:
        plan = json.loads((inputs/'plan.json').read_bytes())
        if not plan['admitted']: raise PermissionError('Diagnostic not admitted')
        for name, digest in plan['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Reviewed source differs')
        if report['versions'] != PINS or torch.cuda.get_device_name(0) != 'NVIDIA L40S':
            raise ValueError('Software or hardware pins differ')
        if plan['public_initial_seeds'] != list(range(93284000, 93284008)) or plan['max_public_transitions'] != 1600:
            raise ValueError('Registered public protocol differs')
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest() != TASK_SHA:
            raise ValueError('Frozen task differs')
        urdf = Path(mani_skill.__file__).parent/'assets/robots/panda/panda_v3.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != URDF_SHA: raise ValueError('URDF differs')
        cache = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest() != '4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':
            raise ValueError('Pinned PhysX cache differs')
        chain = parse_chain(urdf)
        kinematics = Kinematics(chain)
        limits = np.asarray([j['limits'] for j in chain if j['type'] == 'revolute'])
        artifact = load_policy(inputs/'policy', PolicySchema(FINGERPRINT, 48, 8))
        if artifact.weights_sha256 != POLICY_SHA: raise ValueError('Frozen actor differs')
        actor = NumpyPolicy(artifact, PolicySchema(FINGERPRINT, 48, 8))
        env = gym.make('StackCube-v1', num_envs=8, robot_uids='panda_wristcam', obs_mode='state',
                       reward_mode='normalized_dense', control_mode='pd_joint_delta_pos',
                       sim_backend='physx_cuda', render_mode='rgb_array')
        if env.unwrapped.single_observation_space.shape != (48,) or env.unwrapped.single_action_space.shape != (8,):
            raise ValueError('Fixed interface differs')

        def live_fk(observation):
            transform = kinematics.fk(observation[:, :7])
            residual = dict(position_m=float(np.linalg.norm(transform[:, :3, 3]-observation[:, 18:21], axis=1).max()),
                            rotation_rad=float(angle_between(transform[:, :3, :3], quaternion_matrix(observation[:, 21:25])).max()))
            report['live_fk_max_position_m'] = max(report.get('live_fk_max_position_m', 0), residual['position_m'])
            report['live_fk_max_rotation_rad'] = max(report.get('live_fk_max_rotation_rad', 0), residual['rotation_rad'])
            if residual['position_m'] > 1e-4 or residual['rotation_rad'] > 1e-3:
                raise ValueError('Live FK gate failed')
            return transform

        for condition in ('A', 'B', 'C', 'D'):
            if time.monotonic() > deadline: raise TimeoutError('Original diagnostic lease expired')
            obs, _ = env.reset(seed=plan['public_initial_seeds'])
            report['explicit_vector_reset_calls'] += 1
            obs = obs.detach().cpu().numpy().copy()
            initial = obs.copy()
            current = dict(condition=condition, initial_match=None, intervention_matches=[], state='started')
            report['conditions'].append(current)
            arrays = {key: [] for key in ('observations', 'next_observations', 'actions', 'rewards',
                                         'grasp', 'success', 'terminated', 'truncated')}
            streak = np.zeros(8, dtype=int)
            target = np.broadcast_to(np.eye(4), (8, 4, 4)).copy()
            def save_current():
                # Preserve returned frames and pre-step attempted counts even if
                # the next simulator call hangs and the wrapper kills the child.
                current['completed_vector_steps'] = len(arrays['actions'])
                temporary = output/(condition+'.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, initial=initial, eligibility_step=eligibility,
                                        **{key: np.asarray(value) for key, value in arrays.items()})
                temporary.replace(output/(condition+'.npz'))
                write(output/'report.json', report)
            save_current()
            try:
                if baseline is not None:
                    current['initial_match'] = match_residual(initial, baseline['initial'])
                    if not require_match(current['initial_match']):
                        report['validity_passed'] = False
                        raise ValueError('Initial-state matching failed; no paired conclusion')
                for step in range(50):
                    if time.monotonic() > deadline: raise TimeoutError('Original diagnostic lease expired')
                    transform = live_fk(obs)
                    action = np.asarray(actor.act(obs), dtype=np.float64)
                    report['policy_inference_rows'] += 8
                    active = (eligibility >= 0) & (step >= eligibility)
                    starting = eligibility == step
                    if condition != 'A':
                        action[~active] = baseline['actions'][step][~active]
                        if starting.any():
                            residual = match_residual(obs[starting], baseline['observations'][step][starting])
                            own_grasp_history = np.asarray(arrays['grasp'][-5:])[:, starting]
                            contact_eligible = bool(len(own_grasp_history) == 5 and own_grasp_history.all())
                            current['intervention_matches'].append(dict(step=step, lanes=np.flatnonzero(starting).tolist(),
                                residual=residual, preceding_five_grasp_flags=own_grasp_history.tolist(), contact_eligible=contact_eligible))
                            if not require_match(residual) or not contact_eligible:
                                report['validity_passed'] = False
                                raise ValueError('Intervention-state matching failed; no paired conclusion')
                        target[starting] = transform[starting]
                        target[starting, :3, 3] = obs[starting, 18:21]
                        target[starting, :3, :3] = quaternion_matrix(obs[starting, 21:25])
                        target[starting, 2, 3] += .06
                        if condition in ('C', 'D') and active.any():
                            fk, jac = kinematics.fk(obs[active, :7], jacobian=True)
                            arm_action, _ = upward_action(obs[active, :7], fk, jac, target[active], limits)
                            report['inverse_kinematics_pose_evaluations'] += int(active.sum())
                            action[active, :7] = arm_action
                        if condition in ('B', 'D'): action[active, 7] = -1
                    if not np.isfinite(action).all() or np.abs(action).max() > 1.000001:
                        raise ValueError('Invalid bounded action')
                    if report['attempted_public_transitions']+8 > 1600: raise RuntimeError('Pre-step query cap')
                    report['attempted_public_transitions'] += 8
                    write(output/'report.json', report)
                    successor, reward, term, trunc, info = env.step(torch.as_tensor(action, dtype=torch.float32, device='cuda'))
                    report['actual_public_transitions'] += 8
                    report['vector_step_calls'] += 1
                    values = dict(observations=obs, next_observations=successor.detach().cpu().numpy().copy(),
                                  actions=action.copy(), rewards=reward.detach().cpu().numpy().copy(),
                                  grasp=info['is_cubeA_grasped'].detach().cpu().numpy().astype(bool),
                                  success=info['success'].detach().cpu().numpy().astype(bool),
                                  terminated=term.detach().cpu().numpy().astype(bool),
                                  truncated=trunc.detach().cpu().numpy().astype(bool))
                    for key, value in values.items(): arrays[key].append(value)
                    obs = values['next_observations']
                    if condition == 'A' and step+1 <= 25:
                        streak = np.where(values['grasp'], streak+1, 0)
                        eligibility[(eligibility < 0) & (streak >= 5)] = step+1
                    save_current()
                    live_fk(obs)
                    if not np.isfinite(values['rewards']).all() or values['rewards'].min() < -1e-5 or values['rewards'].max() > 1.00001:
                        raise ValueError('Reward convention differs')
                    if values['truncated'].any() and step < 49: raise ValueError('Early cutoff differs')
                if not values['truncated'].all(): raise ValueError('Expected fixed 50-step horizon')
                current['state'] = 'complete'
            finally:
                saved = {key: np.asarray(value) for key, value in arrays.items()}
                save_current()
                if condition == 'A' and current['state'] == 'complete':
                    baseline = dict(initial=initial, **saved)
                write(output/'report.json', report)
        report['state'] = 'complete'
    except Exception as exc:
        report.update(state='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if env is not None: env.close()
        if kinematics is not None:
            report['analytic_fk_pose_evaluations'] = kinematics.fk_pose_evaluations
            report['analytic_jacobian_pose_evaluations'] = kinematics.geometric_jacobian_pose_evaluations
        report['eligibility_step'] = eligibility.tolist()
        report['eligible_lanes'] = int((eligibility >= 0).sum())
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
