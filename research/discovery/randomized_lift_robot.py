"""Single-batch randomized assisted mechanical screen; no learning or replay."""
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
from lift_control_robot import write, PINS, TASK_SHA, URDF_SHA, FINGERPRINT, POLICY_SHA
from lift_probe_control import upward_action
from panda_kinematics import Kinematics, angle_between, parse_chain, quaternion_matrix


def assignment():
    return np.random.default_rng(93305000).permutation(np.repeat(np.arange(4), 32))


def update_eligibility(streak, eligible, grasp, elapsed):
    streak = np.where(grasp, streak+1, 0)
    if elapsed <= 25:
        eligible[(eligible < 0) & (streak >= 5)] = elapsed
    return streak


def masks(groups, eligible, step):
    active = (eligible >= 0) & (step >= eligible)
    return active, active & (groups >= 2), active & ((groups == 1) | (groups == 3))


def run(inputs, output):
    started = time.monotonic()
    deadline = min(started+270, float(os.environ['AUTOMANIP_DIAGNOSTIC_DEADLINE']))
    output.mkdir(exist_ok=False)
    # Assignment is fixed independently before any simulator observation/query.
    groups = assignment()
    eligible = np.full(128, -1, dtype=int)
    report = dict(state='started', kind='Unpaired randomized assisted mechanical screen, not a learned method',
                  assignment=groups.tolist(), assignment_seed=93305000,
                  public_initial_seeds=list(range(93304000, 93304128)),
                  actual_public_transitions=0, attempted_public_transitions=0, vector_step_calls=0,
                  explicit_vector_reset_calls=0, policy_inference_rows=0,
                  inverse_kinematics_pose_evaluations=0, optimizer_calls=0,
                  protected_evaluations=0, new_training_runs=0, live_FK_valid=True,
                  assistance=dict(own_policy_training_transitions=256000,
                                  own_policy_training_seconds=195.434352986, public_URDF=True),
                  versions={key: importlib.metadata.version(key) for key in PINS})
    write(output/'report.json', report)
    arrays = {key: [] for key in ('observations', 'next_observations', 'actions', 'rewards',
                                 'grasp', 'success', 'terminated', 'truncated', 'target_transforms')}
    initial = None
    env = None
    kinematics = None
    target = np.broadcast_to(np.eye(4), (128, 4, 4)).copy()

    def save():
        if kinematics is not None:
            report['analytic_FK_pose_evaluations'] = kinematics.fk_pose_evaluations
            report['analytic_jacobian_pose_evaluations'] = kinematics.geometric_jacobian_pose_evaluations
        if initial is not None:
            temporary = output/'trajectories.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, initial=initial, assignment=groups,
                    eligibility_step=eligible, **{key: np.asarray(value) for key, value in arrays.items()})
            temporary.replace(output/'trajectories.npz')
        report['eligibility_step'] = eligible.tolist()
        report['eligible_counts'] = {str(i): int(((groups == i) & (eligible >= 0)).sum()) for i in range(4)}
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'report.json', report)

    try:
        plan = json.loads((inputs/'plan.json').read_bytes())
        if not plan['admitted']: raise PermissionError('Default denied')
        if plan['max_public_transitions'] != 6400 or plan['public_initial_seeds'] != report['public_initial_seeds']:
            raise ValueError('Registered query/seed plan differs')
        if groups.tolist() != plan['assignment']: raise ValueError('Predeclared random assignment differs')
        for name, expected in plan['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reviewed source differs')
        if report['versions'] != PINS or torch.cuda.get_device_name(0) != 'NVIDIA L40S':
            raise ValueError('Software/hardware pins differ')
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest() != TASK_SHA: raise ValueError('Task differs')
        urdf = Path(mani_skill.__file__).parent/'assets/robots/panda/panda_v3.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != URDF_SHA: raise ValueError('Public URDF differs')
        cache = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest() != '4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':
            raise ValueError('Pinned PhysX cache differs')
        chain = parse_chain(urdf)
        kinematics = Kinematics(chain)
        limits = np.asarray([joint['limits'] for joint in chain if joint['type'] == 'revolute'])
        schema = PolicySchema(FINGERPRINT, 48, 8)
        artifact = load_policy(inputs/'policy', schema)
        if artifact.weights_sha256 != POLICY_SHA: raise ValueError('Fixed safe policy differs')
        actor = NumpyPolicy(artifact, schema)
        env = gym.make('StackCube-v1', num_envs=128, robot_uids='panda_wristcam', obs_mode='state',
                       reward_mode='normalized_dense', control_mode='pd_joint_delta_pos',
                       sim_backend='physx_cuda', render_mode='rgb_array')
        if env.unwrapped.single_observation_space.shape != (48,) or env.unwrapped.single_action_space.shape != (8,):
            raise ValueError('Frozen interface differs')
        tensor, _ = env.reset(seed=report['public_initial_seeds'])
        report['explicit_vector_reset_calls'] += 1
        obs = tensor.detach().cpu().numpy().copy()
        initial = obs.copy()
        streak = np.zeros(128, dtype=int)
        save()

        def live_fk(x):
            if not np.isfinite(x).all():
                report['live_FK_valid'] = False
                raise ValueError('Nonfinite physical observation')
            transform = kinematics.fk(x[:, :7])
            position = float(np.linalg.norm(transform[:, :3, 3]-x[:, 18:21], axis=1).max())
            rotation = float(angle_between(transform[:, :3, :3], quaternion_matrix(x[:, 21:25])).max())
            if not np.isfinite([position, rotation]).all():
                report['live_FK_valid'] = False
                raise ValueError('Nonfinite live FK residual')
            report['live_FK_max_position_m'] = max(report.get('live_FK_max_position_m', 0), position)
            report['live_FK_max_rotation_rad'] = max(report.get('live_FK_max_rotation_rad', 0), rotation)
            if position > 1e-4 or rotation > 1e-3:
                report['live_FK_valid'] = False
                raise ValueError('Live FK validity gate failed')
            return transform

        for step in range(50):
            if time.monotonic() > deadline: raise TimeoutError('Original lease expired')
            transform = live_fk(obs)
            policy_action = np.asarray(actor.act(obs), dtype=np.float64)
            report['policy_inference_rows'] += 128
            action = policy_action.copy()
            active, servo, close = masks(groups, eligible, step)
            starting = eligible == step
            if starting.any():
                target[starting] = transform[starting]
                target[starting, :3, 3] = obs[starting, 18:21]
                target[starting, :3, :3] = quaternion_matrix(obs[starting, 21:25])
                target[starting, 2, 3] += .06
            if servo.any():
                fk, jac = kinematics.fk(obs[servo, :7], jacobian=True)
                action[servo, :7], _ = upward_action(obs[servo, :7], fk, jac, target[servo], limits)
                report['inverse_kinematics_pose_evaluations'] += int(servo.sum())
            action[close, 7] = -1
            if not np.array_equal(action[~active], policy_action[~active]): raise ValueError('Pretreatment action changed')
            if not np.isfinite(action).all() or np.abs(action).max() > 1.000001: raise ValueError('Invalid bounded action')
            if report['attempted_public_transitions']+128 > 6400: raise ValueError('Pre-step transition cap')
            report['attempted_public_transitions'] += 128
            report['analytic_FK_pose_evaluations'] = kinematics.fk_pose_evaluations
            report['analytic_jacobian_pose_evaluations'] = kinematics.geometric_jacobian_pose_evaluations
            write(output/'report.json', report)
            following, reward, term, trunc, info = env.step(torch.as_tensor(action, dtype=torch.float32, device='cuda'))
            report['actual_public_transitions'] += 128
            report['vector_step_calls'] += 1
            values = dict(observations=obs, next_observations=following.detach().cpu().numpy().copy(),
                actions=action, rewards=reward.detach().cpu().numpy().copy(),
                grasp=info['is_cubeA_grasped'].detach().cpu().numpy().astype(bool),
                success=info['success'].detach().cpu().numpy().astype(bool),
                terminated=term.detach().cpu().numpy().astype(bool), truncated=trunc.detach().cpu().numpy().astype(bool),
                target_transforms=target.copy())
            for key, value in values.items(): arrays[key].append(value)
            streak = update_eligibility(streak, eligible, values['grasp'], step+1)
            obs = values['next_observations']
            save()
            live_fk(obs)
            if not np.isfinite(values['rewards']).all() or values['rewards'].min() < -1e-5 or values['rewards'].max() > 1.00001:
                raise ValueError('Reward convention differs')
            if values['truncated'].any() and step < 49: raise ValueError('Unexpected early cutoff')
        if not values['truncated'].all(): raise ValueError('Expected fixed 50-step horizon')
        report['state'] = 'complete'
    except Exception as exc:
        report.update(state='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if env is not None: env.close()
        if kinematics is not None:
            report['analytic_FK_pose_evaluations'] = kinematics.fk_pose_evaluations
            report['analytic_jacobian_pose_evaluations'] = kinematics.geometric_jacobian_pose_evaluations
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
