"""Off-policy TD3 for state StackCube on GPU-parallel ManiSkill, exporting the restricted 48-256-256-256-8 tanh MLP.

From random initialization; no demonstrations, teacher or prior data. Behavior-only exploration (per-env noise
scales, optional gripper-open bursts) needs no importance correction because TD3 is off-policy. Optional exact
base-yaw symmetry rotates replay samples (s, s') together; actions and rewards are invariant.
"""
from dataclasses import dataclass
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from export_helper import export_actor


@dataclass
class Args:
    exp_name: str = "td3"
    seed: int = 1
    env_id: str = "StackCube-v1"
    control_mode: str = "pd_joint_delta_pos"
    num_envs: int = 1024
    total_timesteps: int = 5_120_000
    checkpoint_every: int = 512_000
    gamma: float = 0.9
    n_step: int = 3
    buffer_steps: int = 2048
    """replay capacity in vector steps (x num_envs transitions)"""
    batch_size: int = 16384
    updates_per_step: int = 4
    learning_starts: int = 20
    """vector steps of pure exploration before updates"""
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_width: int = 512
    tau: float = 0.01
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    noise_min: float = 0.05
    noise_max: float = 0.4
    """each env draws its Gaussian exploration std uniformly in [min, max] at every episode start"""
    mean_barrier: float = 1.0
    """actor loss coefficient on relu(|mu| - 1)^2 (env clips; no free drift outside the box)"""
    burst_prob: float = 0.0
    """per-step probability that an env starts holding the gripper fully open (behavior only)"""
    burst_min: int = 3
    burst_max: int = 10
    per_alpha: float = 0.0
    """prioritized replay exponent on |TD error| (0 = uniform replay)"""
    per_beta0: float = 0.4
    """importance-sampling exponent, annealed linearly to 1 over training"""
    per_eps: float = 1e-3
    sym_theta: float = 0.0
    """rotate replay samples about the robot base by U(-theta, theta) rad"""
    transfer_eval: bool = False
    """other Panda tasks: research-side deterministic evaluation at each checkpoint instead of frozen-evaluator export"""
    eval_seed_base: int = 300_000_000


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, bias)
    return layer


class Actor(nn.Module):
    """Exactly the restricted exported architecture: linear output, evaluator clips to the action box."""
    def __init__(self, obs_dim=48):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(), layer_init(nn.Linear(256, 256)), nn.Tanh(),
                                 layer_init(nn.Linear(256, 256)), nn.Tanh(), layer_init(nn.Linear(256, 8), std=0.01))
        self.hidden_sizes = [256, 256, 256]

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, width, obs_dim=48):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(obs_dim+8, width), nn.LayerNorm(width), nn.ReLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
                                 nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def n_step_targets(rew, t, e, m, gamma):
    """Discounted sum of rewards rew[t+k, e] for k < m (m per sample, >= 1)."""
    total = torch.zeros_like(t, dtype=rew.dtype)
    for k in range(int(m.max())):
        live = k < m
        total = total + torch.where(live, rew[(t+k) % rew.shape[0], e]*gamma**k, torch.zeros_like(total))
    return total


def main():
    args = tyro.cli(Args)
    if args.checkpoint_every % args.num_envs or args.total_timesteps % args.checkpoint_every:
        raise ValueError('checkpoint grid must align with vector steps')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda')
    deadline = float(os.environ.get('AUTOMANIP_TRAINING_DEADLINE', time.monotonic()+1300))
    started = time.monotonic()
    envs = gym.make(args.env_id, num_envs=args.num_envs, obs_mode='state', control_mode=args.control_mode,
                    render_mode='rgb_array', sim_backend='physx_cuda')
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    horizon = gym_utils.find_max_episode_steps_value(envs._env)
    owner = envs._env.unwrapped
    if args.env_id != 'StackCube-v1' and not args.transfer_eval:
        raise ValueError('only StackCube exports to the frozen evaluator')
    obs_dim = envs.single_observation_space.shape[0]
    if (obs_dim != 48 and not args.transfer_eval) or envs.single_action_space.shape != (8,) or horizon != 50:
        raise ValueError('Frozen StackCube interface/horizon differs')
    if owner.control_freq != 20 or owner.sim_freq != 100:
        raise ValueError('Physical clocks differ')
    root = Path('runs')/args.exp_name; root.mkdir(parents=True)
    (root/'environment.json').write_text(json.dumps(dict(reward_mode=owner.reward_mode, control_freq=owner.control_freq,
                                                         sim_freq=owner.sim_freq, horizon=horizon)))
    journal = (root/'work.jsonl').open('x', buffering=1)
    def event(kind, count):
        journal.write(json.dumps(dict(event=kind, count=int(count)))+'\n')

    actor = Actor(obs_dim).to(device); actor_target = Actor(obs_dim).to(device); actor_target.load_state_dict(actor.state_dict())
    critic = Critic(args.critic_width, obs_dim).to(device); critic_target = Critic(args.critic_width, obs_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    if args.sym_theta:
        from symmetry import rotate_obs
        center = owner.agent.robot.pose.p[0, :2].to(device)

    T, E = args.buffer_steps, args.num_envs
    obs_buf = torch.zeros(T, E, obs_dim, device=device); next_buf = torch.zeros(T, E, obs_dim, device=device)
    act_buf = torch.zeros(T, E, 8, device=device); rew_buf = torch.zeros(T, E, device=device)
    epstep_buf = torch.zeros(T, E, dtype=torch.long, device=device)
    prio = torch.zeros(T, E, device=device)  # PER priorities^alpha; 0 = never written
    prio_max = torch.ones((), device=device)

    global_step = 0; optimizer_steps = 0; vector_step = 0
    if args.transfer_eval:
        transfer_envs = ManiSkillVectorEnv(gym.make(args.env_id, num_envs=128, obs_mode='state', control_mode=args.control_mode,
                                                    render_mode='rgb_array', sim_backend='physx_cuda'), 128, ignore_terminations=True)
    def export():
        if args.transfer_eval:
            o, _ = transfer_envs.reset(seed=[args.eval_seed_base+i for i in range(128)])
            once = torch.zeros(128, dtype=torch.bool, device=device)
            ever = {}  # stage diagnostics: every boolean info flag, episodes where it was ever true
            te = transfer_envs.unwrapped
            min_xy = torch.full((128,), float('inf'), device=device)
            for _ in range(horizon):
                with torch.no_grad():
                    o, _, _, _, info = transfer_envs.step(actor(o).clamp(-1, 1))
                end = info['final_info'] if 'final_info' in info else info
                once |= end['success'].bool()
                for k, v in end.items():
                    if torch.is_tensor(v) and v.dtype == torch.bool and v.shape == (128,):
                        ever[k] = ever.get(k, torch.zeros_like(v)) | v
                if hasattr(te, 'obj') and hasattr(te, 'bin'):
                    min_xy = torch.minimum(min_xy, torch.linalg.norm((te.obj.pose.p - te.bin.pose.p)[:, :2], dim=1))
            stages = {k: dict(ever=int(v.sum()), end=int(end[k].bool().sum())) for k, v in ever.items()}
            if torch.isfinite(min_xy).all():
                stages['min_obj_goal_xy_mm'] = [round(float(q)*1000, 1) for q in min_xy.quantile(torch.tensor([.1, .5, .9], device=device))]
            journal.write(json.dumps(dict(event='transfer_eval', steps=global_step, success_at_end=int(end['success'].sum()),
                                          success_once=int(once.sum()), episodes=128, stages=stages,
                                          trainer_seconds=time.monotonic()-started))+'\n')
            return
        artifact = export_actor(actor, root/'actors'/str(global_step))
        journal.write(json.dumps(dict(event='actor_export', steps=global_step, actor=artifact,
                                      trainer_seconds=time.monotonic()-started))+'\n')
    export()
    obs, _ = envs.reset(seed=args.seed)
    episode_step = 0
    noise_std = torch.empty(E, 1, device=device).uniform_(args.noise_min, args.noise_max)
    burst_left = torch.zeros(E, dtype=torch.long, device=device)
    stats = dict(success_once=[], q=[], critic_loss=[], burst_frac=[])
    last_log = time.monotonic()
    while global_step < args.total_timesteps:
        if time.monotonic() >= deadline:
            print('TRAINING_TIME_BUDGET_REACHED', global_step, flush=True); break
        with torch.no_grad():
            if vector_step < args.learning_starts:
                action = torch.rand(E, 8, device=device)*2-1
            else:
                action = (actor(obs) + torch.randn(E, 8, device=device)*noise_std).clamp(-1, 1)
            if args.burst_prob:
                start = (burst_left == 0) & (torch.rand(E, device=device) < args.burst_prob)
                burst_left = torch.where(start, torch.randint(args.burst_min, args.burst_max+1, (E,), device=device), burst_left)
                bursting = burst_left > 0
                action[:, 7] = torch.where(bursting, torch.ones_like(action[:, 7]), action[:, 7])
                burst_left = (burst_left - 1).clamp(min=0)
                stats['burst_frac'].append(bursting.float().mean().item())
        event('environment_reserved', global_step+E)
        next_obs, reward, terminations, truncations, infos = envs.step(action)
        global_step += E
        event('environment_completed', global_step)
        done = truncations.bool() | terminations.bool()
        real_next = next_obs.clone()
        if 'final_observation' in infos:
            mask = infos['_final_info']
            real_next[mask] = infos['final_observation'][mask]
            if 'final_info' in infos:
                stats['success_once'].append(infos['final_info']['episode']['success_once'][mask].float().mean().item())
        i = vector_step % T
        obs_buf[i] = obs; next_buf[i] = real_next; act_buf[i] = action; rew_buf[i] = reward.float()
        epstep_buf[i] = episode_step
        if args.per_alpha:
            prio[i] = prio_max
        episode_step += 1
        if bool(done.any()):
            if not bool(done.all()) or episode_step != horizon:
                raise RuntimeError('episodes are expected to stay aligned at the fixed horizon')
            episode_step = 0
            noise_std = torch.empty(E, 1, device=device).uniform_(args.noise_min, args.noise_max)
            burst_left.zero_()
        obs = next_obs
        vector_step += 1

        if vector_step >= args.learning_starts:
            stored = min(vector_step, T)
            for _ in range(args.updates_per_step):
                # sample (t, e); n-step window stops at episode end and at the newest stored step
                if args.per_alpha:
                    flat = prio.view(-1)
                    idx = torch.multinomial(flat, args.batch_size, replacement=True)
                    t, e = idx // E, idx % E
                    age = (vector_step - 1 - t) % T
                    beta = args.per_beta0 + (1 - args.per_beta0)*min(1.0, global_step/args.total_timesteps)
                    weight = (flat[idx]/flat.sum()*(stored*E)).pow(-beta)
                    weight = weight/weight.max()
                else:
                    age = torch.randint(0, stored, (args.batch_size,), device=device)  # 0 = newest
                    t = (vector_step - 1 - age) % T
                    e = torch.randint(0, E, (args.batch_size,), device=device)
                m = torch.minimum(torch.full_like(age, args.n_step), age + 1)
                m = torch.minimum(m, horizon - epstep_buf[t, e])
                s, a = obs_buf[t, e], act_buf[t, e]
                boot = next_buf[(t + m - 1) % T, e]
                ret = n_step_targets(rew_buf, t, e, m, args.gamma)
                if args.sym_theta:
                    theta = (torch.rand(args.batch_size, device=device)*2-1)*args.sym_theta
                    s = rotate_obs(s, theta, center); boot = rotate_obs(boot, theta, center)
                with torch.no_grad():
                    noise = (torch.randn_like(a)*args.target_noise).clamp(-args.target_noise_clip, args.target_noise_clip)
                    next_a = (actor_target(boot) + noise).clamp(-1, 1)
                    tq1, tq2 = critic_target(boot, next_a)
                    target = ret + args.gamma**m.float() * torch.minimum(tq1, tq2)
                q1, q2 = critic(s, a)
                if args.per_alpha:
                    critic_loss = (weight*((q1-target).square() + (q2-target).square())).mean()
                    with torch.no_grad():
                        new = ((q1-target).abs() + (q2-target).abs()).mul(.5).add(args.per_eps).pow(args.per_alpha)
                        prio[t, e] = new
                        prio_max = torch.maximum(prio_max, new.max())
                else:
                    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
                optimizer_steps += 1
                event('optimizer_completed', optimizer_steps)
                if optimizer_steps % args.policy_delay == 0:
                    mu = actor(s)
                    actor_loss = -critic(s, mu.clamp(-1, 1))[0].mean() + args.mean_barrier*F.relu(mu.abs()-1).square().sum(-1).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                    with torch.no_grad():
                        for p, tp in zip(actor.parameters(), actor_target.parameters()):
                            tp.lerp_(p, args.tau)
                        for p, tp in zip(critic.parameters(), critic_target.parameters()):
                            tp.lerp_(p, args.tau)
            stats['q'].append(q1.mean().item()); stats['critic_loss'].append(critic_loss.item())
        if global_step % args.checkpoint_every == 0:
            export()
        if time.monotonic() - last_log > 30:
            last_log = time.monotonic()
            summary = {k: (float(np.mean(v[-20:])) if v else None) for k, v in stats.items()}
            print(json.dumps(dict(step=global_step, sps=round(global_step/(last_log-started)), optimizer_steps=optimizer_steps,
                                  **summary)), flush=True)
    event('training_complete', global_step)
    journal.close()


if __name__ == '__main__':
    main()
