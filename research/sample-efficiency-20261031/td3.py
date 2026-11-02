"""Off-policy TD3 for state StackCube on GPU-parallel ManiSkill, exporting the restricted 48-256-256-256-8 tanh MLP.

From random initialization; no demonstrations, teacher or prior data. Behavior-only exploration (per-env noise
scales, optional gripper-open bursts) needs no importance correction because TD3 is off-policy. Optional exact
base-yaw symmetry rotates replay samples (s, s') together; actions and rewards are invariant.
"""
from dataclasses import dataclass
import contextlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import sapien
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from mani_skill.utils import gym_utils
from transforms3d.euler import euler2quat
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
    fast: bool = False
    """compute-only speedups: TF32 matmuls, fused Adam, torch.compile of training-time critic/actor, foreach target update"""
    bf16: bool = False
    """bf16 autocast for critic forward passes (changes numerics slightly)"""
    utd_late: int = 0
    """if > 0, switch to this many updates per step once training success_once reaches utd_switch_success"""
    utd_switch_success: float = 0.5
    sym_theta: float = 0.0
    """rotate replay samples about the robot base by U(-theta, theta) rad"""
    rc_source: str = "none"
    """reverse curriculum start states: none, demo (motion-planned states from rc_demos), self (own first successes, shared
    across envs), or self-local (each env replays only its own first success; needed when geometry differs per env)"""
    rc_demos: str = ""
    rc_frac: float = 0.25
    """fraction of envs started from a curriculum state at each episode start"""
    rc_stride: int = 4
    rc_threshold: float = 0.5
    """move a trajectory's start back by rc_stride states once this fraction of its episodes succeed at the end"""
    rc_pool: int = 16
    """self source: number of own successful true-start trajectories kept (first ones found, never replaced)"""
    rc_sampler: str = "march"
    """demo source: march (each trajectory's start moves back on mastery) or sgs (success-guided sampling over every demo
    state, weight eps + sqrt(p(1-p)) of its rolling end-of-episode success p, after AlphaNIST, arXiv 2610.13354)"""
    rc_history: int = 20
    """sgs: rolling outcome window per demo state; unvisited states count as p = 0.5"""
    reward_mode: str = "normalized_dense"
    """training reward: ManiSkill's own modes only (transfer tasks); StackCube stays normalized_dense"""
    transfer_eval: bool = False
    """other Panda tasks: research-side deterministic evaluation at each checkpoint instead of frozen-evaluator export"""
    eval_seed_base: int = 300_000_000


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, bias)
    return layer


class Actor(nn.Module):
    """Exactly the restricted exported architecture: linear output, evaluator clips to the action box."""
    def __init__(self, obs_dim=48, act_dim=8):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(), layer_init(nn.Linear(256, 256)), nn.Tanh(),
                                 layer_init(nn.Linear(256, 256)), nn.Tanh(), layer_init(nn.Linear(256, act_dim), std=0.01))
        self.hidden_sizes = [256, 256, 256]

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, width, obs_dim=48, act_dim=8):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(obs_dim+act_dim, width), nn.LayerNorm(width), nn.ReLU(),
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


class ReverseCurriculum:
    """Start some training episodes from states along successful trajectories, moving each start back as it is mastered.

    Training-side only (simulator state resets); evaluation always uses the benchmark's normal starts. Trajectories are
    flat state vectors in the env's own get_state() order, ending at their first success.
    """
    def __init__(self, args, owner, device):
        self.args, self.owner, self.device = args, owner, device
        self.trajs, self.ptr = [], []
        self.slot = None  # per env: trajectory index, or -1 for a true start
        self.done = False
        if args.rc_source == 'demo':
            demos = json.loads(Path(args.rc_demos).read_text())
            if demos['env_id'] != args.env_id:
                raise ValueError('demo task differs')
            order = owner._init_raw_state
            for d in demos['demos']:
                rows = []
                for st in d['states']:
                    parts = [st['actors'][k] for k in order['actors']] + [st['articulations'][k] for k in order['articulations']]
                    rows.append(torch.tensor([x for p in parts for x in p], dtype=torch.float32))
                self.add(torch.stack(rows).to(device))
            self.demo_transitions = demos['environment_transitions']
            if owner.get_state().shape[1] != self.trajs[0].shape[1]:
                raise ValueError('demo state layout differs')

        if args.rc_sampler == 'sgs':
            if args.rc_source != 'demo':
                raise ValueError('sgs sampling needs demo states')
            self.bank = torch.cat(self.trajs)
            self.hist = torch.zeros(len(self.bank), args.rc_history, device=device)
            self.seen = torch.zeros(len(self.bank), dtype=torch.long, device=device)
            self.bank_pos = torch.cat([torch.arange(t.shape[0], device=device)/(t.shape[0]-1) for t in self.trajs])

    def add(self, traj):
        self.trajs.append(traj); self.ptr.append(traj.shape[0]-1)

    def add_local(self, traj_buf, envs, lengths):
        """self-local: store each listed env's own trajectory (states [0, length) along traj_buf's time axis)."""
        if not hasattr(self, 'local_traj'):
            H, E, D = traj_buf.shape
            self.local_traj = torch.zeros(H, E, D, device=self.device)
            self.local_len = torch.zeros(E, dtype=torch.long, device=self.device)
            self.local_ptr = torch.zeros(E, dtype=torch.long, device=self.device)
        self.local_traj[:, envs] = traj_buf[:, envs]
        self.local_len[envs] = lengths; self.local_ptr[envs] = lengths-1

    def start(self, E, obs):
        """Called right after an aligned episode reset; overrides a fraction of envs. Returns observations."""
        self.slot = torch.full((E,), -1, dtype=torch.long, device=self.device)
        if self.args.rc_source == 'self-local':
            if not hasattr(self, 'local_traj'):
                return obs
            idx = torch.nonzero((self.local_len > 0) & (torch.rand(E, device=self.device) < self.args.rc_frac)).flatten()
            if not len(idx):
                return obs
            k = torch.minimum(self.local_ptr[idx] + torch.randint(0, self.args.rc_stride, (len(idx),), device=self.device),
                              self.local_len[idx]-1)
            self.owner.set_state(self.local_traj[k, idx], idx)
            which = torch.zeros_like(idx)
        elif self.args.rc_sampler == 'sgs':
            n = int(E*self.args.rc_frac)
            idx = torch.randperm(E, device=self.device)[:n].sort().values
            count = self.seen.clamp(max=self.args.rc_history)
            p = torch.where(count > 0, self.hist.sum(1)/count.clamp(min=1), torch.full_like(count, 0.5, dtype=torch.float32))
            which = torch.multinomial(1e-2 + (p*(1-p)).sqrt(), n, replacement=True)
            self.owner.set_state(self.bank[which], idx)
        else:
            if self.done or not self.trajs:
                return obs
            n = int(E*self.args.rc_frac)
            idx = torch.randperm(E, device=self.device)[:n].sort().values
            which = torch.randint(0, len(self.trajs), (n,), device=self.device)
            states = []
            for j, w in enumerate(which.tolist()):
                L = self.trajs[w].shape[0]
                k = min(L-1, self.ptr[w] + int(torch.randint(0, self.args.rc_stride, ()).item()))
                states.append(self.trajs[w][k])
            self.owner.set_state(torch.stack(states), idx)
        mask = torch.zeros(E, dtype=torch.bool, device=self.device); mask[idx] = True
        scene = self.owner.scene
        scene._reset_mask = mask
        controllers = self.owner.agent.controller
        for c in (controllers.values() if isinstance(controllers, dict) else [controllers]):
            c.reset()
        scene._reset_mask = torch.ones(E, dtype=torch.bool, device=self.device)
        if hasattr(self.owner, 'receptacle') and hasattr(self.owner, 'goal_pose'):
            # PlugCharger caches goal_pose from the receptacle at reset; restored states move the receptacle
            self.owner.goal_pose = self.owner.receptacle.pose * sapien.Pose(q=euler2quat(0, 0, np.pi))
        fresh = self.owner.get_obs(self.owner.get_info())
        obs = obs.clone(); obs[idx] = fresh[idx]
        self.slot[idx] = which
        return obs

    def finish(self, success_end, journal, global_step):
        """Episode end: advance mastered trajectories; report the curriculum state."""
        if self.args.rc_source == 'self-local':
            if hasattr(self, 'local_traj'):
                hit = (self.slot >= 0) & success_end
                self.local_ptr = torch.where(hit, (self.local_ptr-self.args.rc_stride).clamp(min=0), self.local_ptr)
                have = self.local_len > 0
                journal.write(json.dumps(dict(event='rc', steps=global_step, envs_with_traj=int(have.sum()),
                                              curriculum_envs=int((self.slot >= 0).sum()), advanced=int(hit.sum()),
                                              at_zero=int((have & (self.local_ptr == 0)).sum())))+'\n')
            return
        if self.args.rc_sampler == 'sgs':
            sel = torch.nonzero(self.slot >= 0).flatten()
            if len(sel):
                w = self.slot[sel]
                # several envs may share a bank entry; each outcome takes the next slot of that entry's window
                order = torch.argsort(w, stable=True); w, ok = w[order], success_end[sel][order].float()
                first = torch.searchsorted(w, w, right=False)
                rank = torch.arange(len(w), device=self.device) - first
                col = (self.seen[w] + rank) % self.args.rc_history
                self.hist[w, col] = ok
                self.seen.index_add_(0, w, torch.ones_like(w))
            count = self.seen.clamp(max=self.args.rc_history)
            p = self.hist.sum(1)/count.clamp(min=1)
            tried = count > 0
            journal.write(json.dumps(dict(event='rc', steps=global_step, sampler='sgs', bank=len(self.bank),
                                          tried=int(tried.sum()), curriculum_envs=int(len(sel)),
                                          curriculum_success=float(success_end[sel].float().mean()) if len(sel) else None,
                                          mastered=int((tried & (p >= 0.9)).sum()),
                                          frontier_pos=float((self.bank_pos*(tried & (p >= 0.5))).max()) if tried.any() else None,
                                          earliest_solved_pos=float(self.bank_pos[tried & (p >= 0.5)].min()) if (tried & (p >= 0.5)).any() else None))+'\n')
            return
        moved = 0
        for j in range(len(self.trajs)):
            sel = self.slot == j
            if sel.any() and success_end[sel].float().mean().item() >= self.args.rc_threshold and self.ptr[j] > 0:
                self.ptr[j] = max(0, self.ptr[j]-self.args.rc_stride); moved += 1
        full = self.args.rc_source == 'demo' or len(self.trajs) >= self.args.rc_pool
        if self.trajs and full and not self.done and all(p == 0 for p in self.ptr):
            self.done = True
        journal.write(json.dumps(dict(event='rc', steps=global_step, trajs=len(self.trajs), moved=moved, done=self.done,
                                      ptr=list(self.ptr), lengths=[t.shape[0] for t in self.trajs]))+'\n')


def main():
    args = tyro.cli(Args)
    if args.checkpoint_every % args.num_envs or args.total_timesteps % args.checkpoint_every:
        raise ValueError('checkpoint grid must align with vector steps')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda')
    deadline = float(os.environ.get('AUTOMANIP_TRAINING_DEADLINE', time.monotonic()+1300))
    started = time.monotonic()
    if args.reward_mode != 'normalized_dense' and not args.transfer_eval:
        raise ValueError('StackCube keeps its normalized_dense training reward')
    envs = gym.make(args.env_id, num_envs=args.num_envs, obs_mode='state', control_mode=args.control_mode,
                    render_mode='rgb_array', sim_backend='physx_cuda', reward_mode=args.reward_mode)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    horizon = gym_utils.find_max_episode_steps_value(envs._env)
    owner = envs._env.unwrapped
    if args.env_id != 'StackCube-v1' and not args.transfer_eval:
        raise ValueError('only StackCube exports to the frozen evaluator')
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    if not args.transfer_eval and (obs_dim != 48 or act_dim != 8 or horizon != 50):
        raise ValueError('Frozen StackCube interface/horizon differs')
    if owner.control_freq != 20 or owner.sim_freq != 100:
        raise ValueError('Physical clocks differ')
    root = Path('runs')/args.exp_name; root.mkdir(parents=True)
    (root/'environment.json').write_text(json.dumps(dict(reward_mode=owner.reward_mode, control_freq=owner.control_freq,
                                                         sim_freq=owner.sim_freq, horizon=horizon)))
    journal = (root/'work.jsonl').open('x', buffering=1)
    def event(kind, count):
        journal.write(json.dumps(dict(event=kind, count=int(count)))+'\n')

    actor = Actor(obs_dim, act_dim).to(device); actor_target = Actor(obs_dim, act_dim).to(device); actor_target.load_state_dict(actor.state_dict())
    critic = Critic(args.critic_width, obs_dim, act_dim).to(device); critic_target = Critic(args.critic_width, obs_dim, act_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    if args.fast:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, fused=args.fast)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, fused=args.fast)
    # training-time callables; the exported `actor` module and its parameters are shared and unchanged
    train_critic, train_critic_target, train_actor, train_actor_target = critic, critic_target, actor, actor_target
    if args.fast:
        train_critic, train_critic_target = torch.compile(critic), torch.compile(critic_target)
        train_actor, train_actor_target = torch.compile(actor), torch.compile(actor_target)
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if args.bf16 else contextlib.nullcontext
    actor_params, actor_target_params = list(actor.parameters()), list(actor_target.parameters())
    critic_params, critic_target_params = list(critic.parameters()), list(critic_target.parameters())
    updates_per_step = args.updates_per_step
    if args.sym_theta:
        from symmetry import rotate_obs
        center = owner.agent.robot.pose.p[0, :2].to(device)

    T, E = args.buffer_steps, args.num_envs
    obs_buf = torch.zeros(T, E, obs_dim, device=device); next_buf = torch.zeros(T, E, obs_dim, device=device)
    act_buf = torch.zeros(T, E, act_dim, device=device); rew_buf = torch.zeros(T, E, device=device)
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
                if hasattr(te, 'peg') and hasattr(te, 'goal_pose'):  # PegInsertionSide reward stages
                    head = (te.goal_pose.inv() * te.peg_head_pose).p[:, 1:].norm(dim=1)
                    body = (te.goal_pose.inv() * te.peg.pose).p[:, 1:].norm(dim=1)
                    grasped = te.agent.is_grasping(te.peg, max_angle=20)
                    for k, v in dict(peg_grasped=grasped, peg_pre_inserted=grasped & (head < 0.01) & (body < 0.01)).items():
                        ever[k] = ever.get(k, torch.zeros_like(v)) | v
                        end[k] = v
                if 'obj_to_goal_dist' in end and hasattr(te, 'charger'):  # PlugCharger: closest approach to the plugged pose
                    min_xy = torch.minimum(min_xy, end['obj_to_goal_dist'].float())
                if hasattr(te, 'obj') and hasattr(te, 'bin'):
                    min_xy = torch.minimum(min_xy, torch.linalg.norm((te.obj.pose.p - te.bin.pose.p)[:, :2], dim=1))
            stages = {k: dict(ever=int(v.sum()), end=int(end[k].bool().sum())) for k, v in ever.items()}
            if torch.isfinite(min_xy).all():
                stages['min_charger_goal_mm' if hasattr(te, 'charger') else 'min_obj_goal_xy_mm'] = [round(float(q)*1000, 1) for q in min_xy.quantile(torch.tensor([.1, .5, .9], device=device))]
            journal.write(json.dumps(dict(event='transfer_eval', steps=global_step, success_at_end=int(end['success'].sum()),
                                          success_once=int(once.sum()), episodes=128, stages=stages,
                                          trainer_seconds=time.monotonic()-started))+'\n')
            return
        artifact = export_actor(actor, root/'actors'/str(global_step))
        journal.write(json.dumps(dict(event='actor_export', steps=global_step, actor=artifact,
                                      trainer_seconds=time.monotonic()-started))+'\n')
    export()
    obs, _ = envs.reset(seed=args.seed)
    rc = None
    if args.rc_source != 'none':
        if args.rc_source not in ('demo', 'self', 'self-local'):
            raise ValueError('rc_source must be none, demo, self or self-local')
        rc = ReverseCurriculum(args, owner, device)
        if args.rc_source == 'demo':
            # demo generation counts as environment interaction
            journal.write(json.dumps(dict(event='demo_transitions', count=rc.demo_transitions))+'\n')
        obs = rc.start(E, obs)
        if args.rc_source in ('self', 'self-local'):
            traj_buf = None
            first_succ = torch.full((E,), -1, dtype=torch.long, device=device)
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
                action = torch.rand(E, act_dim, device=device)*2-1
            else:
                action = (actor(obs) + torch.randn(E, act_dim, device=device)*noise_std).clamp(-1, 1)
            if args.burst_prob:
                start = (burst_left == 0) & (torch.rand(E, device=device) < args.burst_prob)
                burst_left = torch.where(start, torch.randint(args.burst_min, args.burst_max+1, (E,), device=device), burst_left)
                bursting = burst_left > 0
                action[:, 7] = torch.where(bursting, torch.ones_like(action[:, 7]), action[:, 7])
                burst_left = (burst_left - 1).clamp(min=0)
                stats['burst_frac'].append(bursting.float().mean().item())
        if rc is not None and (args.rc_source == 'self-local' or (args.rc_source == 'self' and len(rc.trajs) < args.rc_pool)):
            state = owner.get_state()
            if traj_buf is None:
                traj_buf = torch.zeros(horizon, E, state.shape[1], device=device)
            traj_buf[episode_step] = state
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
        if rc is not None:
            final = infos['final_info'] if 'final_info' in infos else infos
            if args.rc_source in ('self', 'self-local') and traj_buf is not None:
                hit = final['success'].bool() & (first_succ < 0)
                first_succ = torch.where(hit, torch.full_like(first_succ, episode_step), first_succ)
        i = vector_step % T
        obs_buf[i] = obs; next_buf[i] = real_next; act_buf[i] = action; rew_buf[i] = reward.float()
        epstep_buf[i] = episode_step
        if args.per_alpha:
            prio[i] = prio_max
        episode_step += 1
        if bool(done.any()):
            if not bool(done.all()) or episode_step != horizon:
                raise RuntimeError('episodes are expected to stay aligned at the fixed horizon')
            if rc is not None:
                success_end = infos['final_info']['success'].bool()
                if args.rc_source == 'self' and traj_buf is not None:
                    for e_ in torch.nonzero((rc.slot < 0) & (first_succ >= 0)).flatten().tolist():
                        if len(rc.trajs) >= args.rc_pool:
                            break
                        rc.add(traj_buf[:min(int(first_succ[e_])+1, horizon-1)+1, e_].clone())
                    first_succ.fill_(-1)
                if args.rc_source == 'self-local' and traj_buf is not None:
                    new = (rc.slot < 0) & (first_succ >= 0)
                    if hasattr(rc, 'local_len'):
                        new &= rc.local_len == 0
                    envs_ = torch.nonzero(new).flatten()
                    if len(envs_):
                        rc.add_local(traj_buf, envs_, torch.clamp(first_succ[envs_]+1, max=horizon-1)+1)
                    first_succ.fill_(-1)
                rc.finish(success_end, journal, global_step)
            episode_step = 0
            noise_std = torch.empty(E, 1, device=device).uniform_(args.noise_min, args.noise_max)
            burst_left.zero_()
        obs = next_obs
        if rc is not None and bool(done.any()):
            obs = rc.start(E, obs)
        vector_step += 1

        if vector_step >= args.learning_starts:
            stored = min(vector_step, T)
            if args.utd_late and updates_per_step != args.utd_late and stats['success_once'] \
                    and np.mean(stats['success_once'][-3:]) >= args.utd_switch_success:
                updates_per_step = args.utd_late
                journal.write(json.dumps(dict(event='utd_switch', steps=global_step, updates_per_step=updates_per_step))+'\n')
            for _ in range(updates_per_step):
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
                with torch.no_grad(), autocast():
                    noise = (torch.randn_like(a)*args.target_noise).clamp(-args.target_noise_clip, args.target_noise_clip)
                    next_a = (train_actor_target(boot) + noise).clamp(-1, 1)
                    tq1, tq2 = train_critic_target(boot, next_a)
                    target = ret + args.gamma**m.float() * torch.minimum(tq1.float(), tq2.float())
                with autocast():
                    q1, q2 = train_critic(s, a)
                q1, q2 = q1.float(), q2.float()
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
                    mu = train_actor(s)
                    with autocast():
                        q_pi = train_critic(s, mu.clamp(-1, 1))[0].float()
                    actor_loss = -q_pi.mean() + args.mean_barrier*F.relu(mu.abs()-1).square().sum(-1).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                    with torch.no_grad():
                        torch._foreach_lerp_(actor_target_params, actor_params, args.tau)
                        torch._foreach_lerp_(critic_target_params, critic_params, args.tau)
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
