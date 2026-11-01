"""Check base-yaw symmetry against the simulator: static observation exactness and short-horizon dynamic equivariance.

Research probe only: uses public reset seeds 97,200,000+, never protected seeds, never scores policies.
"""
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import ROOT, trainer_image, results

SOURCE = ROOT/'research/sample-efficiency-20261031'
app = modal.App('automanip-symmetry-probe-20261031', include_source=False)
image = trainer_image.add_local_file(SOURCE/'symmetry.py', '/opt/se/symmetry.py', copy=True)
cache = results.with_mount_options(read_only=True, sub_path='/runtime/sapien-cache')


@app.function(image=image, gpu='L4', timeout=900, block_network=True, include_source=False, serialized=True,
              volumes={'/root/.sapien': cache})
def probe():
    sys.path.insert(0, '/opt/se')
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch
    from symmetry import rotate_obs, rotate_xy, rotate_quaternion
    n = 64
    env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_joint_delta_pos', num_envs=2*n,
                   reward_mode='normalized_dense', sim_backend='physx_cuda')
    u = env.unwrapped
    report = {}
    obs, _ = env.reset(seed=97200000)
    generator = torch.Generator(device='cpu').manual_seed(0)
    # Random walk so the state is not just the initial pose; keep halves identical by mirroring actions.
    for _ in range(15):
        a = torch.rand(n, 8, generator=generator)*2-1
        obs, *_ = env.step(torch.cat([a, a]).to(u.device))
    state = u.get_state_dict()
    center = u.agent.robot.pose.p[0, :2].cpu()
    report['robot_base'] = u.agent.robot.pose.raw_pose[0].tolist()
    theta = (torch.rand(n, generator=generator)-.5)*1.0  # +-0.5 rad
    # Copy first half onto second half, rotated.
    th = theta.to(u.device); cos, sin = torch.cos(th), torch.sin(th); hc, hs = torch.cos(th/2), torch.sin(th/2)
    off = torch.zeros(3, device=u.device); off[:2] = center.to(u.device)
    for name, s in state['actors'].items():
        if name not in ('cubeA', 'cubeB'):
            continue
        src = s[:n].clone()
        src[:, 0:3] = rotate_xy(src[:, 0:3]-off, cos, sin)+off
        src[:, 3:7] = rotate_quaternion(src[:, 3:7], hc, hs)
        src[:, 7:10] = rotate_xy(src[:, 7:10], cos, sin)
        src[:, 10:13] = rotate_xy(src[:, 10:13], cos, sin)
        s[n:] = src
    for name, s in state['articulations'].items():
        src = s[:n].clone()
        src[:, 13] = src[:, 13] + th  # qpos joint 1 follows root pose (7) and root velocity (6)
        s[n:] = src
    u.set_state_dict(state)
    if u.gpu_sim_enabled:
        u.scene._gpu_apply_all(); u.scene.px.gpu_update_articulation_kinematics(); u.scene._gpu_fetch_all()
    obs = u.get_obs().cpu()
    predicted = rotate_obs(obs[:n], theta, center)
    report['static_max_abs_error'] = float((predicted - obs[n:]).abs().max())
    report['static_error_by_dim'] = (predicted - obs[n:]).abs().max(0).values.tolist()
    errors, rewards = [], []
    for _ in range(10):
        a = torch.rand(n, 8, generator=generator)*2-1
        obs, reward, *_ = env.step(torch.cat([a, a]).to(u.device))
        obs = obs.cpu(); reward = reward.cpu()
        errors.append(float((rotate_obs(obs[:n], theta, center)-obs[n:]).abs().max()))
        rewards.append(float((reward[:n]-reward[n:]).abs().max()))
    report['dynamic_max_abs_obs_error_per_step'] = errors
    report['dynamic_max_abs_reward_error_per_step'] = rewards
    report['theta_range'] = [float(theta.min()), float(theta.max())]
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1))
