"""AutoManip PPO trainer, adapted from ManiSkill v3.0.0b21's PPO baseline.

Training-only code: the trusted evaluator never imports this file.
Changes: explicit budgets/provenance, bounded config, data-only actor exports,
CPU backend option, and separate evaluation. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from automanip.policy_io import PolicySchema, load_policy
from automanip.task_spec import TASK


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 1
    num_envs: int = 512
    num_steps: int = 50
    total_timesteps: int = 9_984_000
    max_seconds: float = 600.0
    learning_rate: float = 3e-4
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    # Upstream bootstraps final observations even for success terminations.
    bootstrap_on_termination: bool = True
    partial_reset: bool = True
    anneal_lr: bool = False

    def validate(self):
        for name in ("num_envs", "num_steps", "total_timesteps", "num_minibatches", "update_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        if self.total_timesteps < self.num_envs or self.total_timesteps % self.num_envs:
            raise ValueError("total_timesteps must be a positive multiple of num_envs")
        for name in ("max_seconds", "learning_rate", "gamma", "gae_lambda", "clip_coef",
                     "ent_coef", "vf_coef", "max_grad_norm", "target_kl"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.max_seconds, self.learning_rate, self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError("time, learning rate, gradient norm and KL limits must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_coef < 1:
            raise ValueError("invalid discount, GAE lambda or clipping coefficient")
        if not 1 <= len(self.hidden_sizes) <= 3 or any(type(n) is not int or not 1 <= n <= 256 for n in self.hidden_sizes):
            raise ValueError("policy must have one to three hidden layers of at most 256 units")
        for name in ("bootstrap_on_termination", "partial_reset", "anneal_lr"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        return self


def mlp(inputs, outputs, hidden_sizes, final_gain):
    layers = []
    dims = [inputs, *hidden_sizes, outputs]
    for index, (left, right) in enumerate(zip(dims, dims[1:])):
        layer = nn.Linear(left, right)
        nn.init.orthogonal_(layer.weight, final_gain if index == len(dims)-2 else math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if index < len(dims)-2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.hidden_sizes = tuple(config.hidden_sizes)
        self.actor = mlp(TASK.observation_dim, TASK.action_dim, self.hidden_sizes, 0.01*math.sqrt(2))
        self.critic = mlp(TASK.observation_dim, 1, self.hidden_sizes, 1.0)
        self.logstd = nn.Parameter(torch.full((1, TASK.action_dim), -0.5))

    def action_and_value(self, observations, actions=None):
        distribution = Normal(self.actor(observations), self.logstd.exp())
        if actions is None:
            actions = distribution.sample()
        return (actions, distribution.log_prob(actions).sum(-1),
                distribution.entropy().sum(-1), self.critic(observations).squeeze(-1))


@torch.no_grad()
def successor_values(agent, next_obs, terminated, truncated, info, bootstrap_on_termination=True):
    """Use pre-reset final observations, never an autoreset state's value."""
    done = terminated | truncated
    successor_obs = next_obs.clone()
    if done.any():
        if "final_observation" not in info or "_final_observation" not in info:
            raise ValueError("autoreset did not supply final observations")
        if not torch.equal(info["_final_observation"], done):
            raise ValueError("final observation mask differs from reset mask")
        successor_obs[done] = info["final_observation"][done]
    values = agent.critic(successor_obs).squeeze(-1)
    return values if bootstrap_on_termination else values.masked_fill(terminated, 0)


def generalized_advantage(rewards, values, next_values, dones, gamma, gae_lambda):
    """next_values already encodes the requested terminal bootstrap semantics."""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma*next_values[t] - values[t]
        following = delta + gamma*gae_lambda*(~dones[t])*following
        advantages[t] = following
    return advantages, advantages + values


def clipped_policy_loss(new_logprob, old_logprob, advantages, clip_coef):
    ratio = (new_logprob-old_logprob).exp()
    loss = torch.maximum(-advantages*ratio, -advantages*ratio.clamp(1-clip_coef, 1+clip_coef)).mean()
    kl = ((ratio-1)-(new_logprob-old_logprob)).mean()
    return loss, kl


def ppo_update(agent, optimizer, batch, config, deadline=math.inf):
    """Updates are budget-checked between minibatches; native calls are not preempted."""
    obs, actions, old_logprob, advantages, returns = batch
    size = len(obs)
    updates, stats = 0, {}
    for _ in range(config.update_epochs):
        permutation = torch.randperm(size, device=obs.device)
        for indices in permutation.tensor_split(min(size, config.num_minibatches)):
            if time.perf_counter() >= deadline:
                return updates, stats
            _, logprob, entropy, values = agent.action_and_value(obs[indices], actions[indices])
            advantage = advantages[indices]
            advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            policy_loss, kl = clipped_policy_loss(logprob, old_logprob[indices], advantage, config.clip_coef)
            if not torch.isfinite(kl):
                raise ValueError("nonfinite KL")
            if kl.item() > config.target_kl:
                return updates, stats
            value_loss = 0.5*(values-returns[indices]).square().mean()
            loss = policy_loss + config.vf_coef*value_loss - config.ent_coef*entropy.mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            stats = dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                         entropy=entropy.mean().item(), approx_kl=kl.item())
    return updates, stats


def export_actor(agent, destination: Path):
    """Actor-only artifact; no optimizer/critic pickle and no executable code."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for i, layer in enumerate(layer for layer in agent.actor if isinstance(layer, nn.Linear)):
        for name in ("weight", "bias"):
            value = getattr(layer, name).detach().cpu()
            tensors[f"layers.{i}.{name}"] = dict(shape=list(value.shape), values=value.flatten().tolist())
    raw = json.dumps(tensors, allow_nan=False, separators=(",", ":")).encode()
    (destination/"weights.json").write_bytes(raw)
    manifest = dict(format="automanip-json-f32-v1", task_fingerprint=TASK.fingerprint,
                    observation_dim=TASK.observation_dim, action_dim=TASK.action_dim,
                    hidden_sizes=list(agent.hidden_sizes), activation="tanh",
                    weights_sha256=hashlib.sha256(raw).hexdigest())
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Gate our own exports with the same trusted validator evaluation uses.
    load_policy(destination, PolicySchema(TASK.fingerprint, TASK.observation_dim))
    return manifest["weights_sha256"]


def make_environment(config, backend):
    if version("mani_skill") != TASK.mani_skill_version:
        raise RuntimeError("ManiSkill differs from pinned task")
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    kwargs = TASK.env_kwargs(num_envs=config.num_envs, backend=backend)
    kwargs["reward_mode"] = "normalized_dense"
    owner = gym.make(TASK.env_id, reconfiguration_freq=0, **kwargs)
    try:
        env = ManiSkillVectorEnv(owner, ignore_terminations=not config.partial_reset, record_metrics=True)
        if env.single_observation_space.shape != (TASK.observation_dim,) or env.single_action_space.shape != (TASK.action_dim,):
            raise ValueError("training interface differs from frozen task")
        if not (env.single_action_space.low == -1).all() or not (env.single_action_space.high == 1).all():
            raise ValueError("expected normalized actions")
        return env
    except Exception:
        owner.close()
        raise


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run(config: PPOConfig, backend: str, output: Path, hypothesis: str):
    config.validate()
    if backend == "physx_cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    TASK.env_kwargs(num_envs=config.num_envs, backend=backend)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    started = time.perf_counter()
    deadline = started + config.max_seconds
    device = torch.device("cuda" if backend == "physx_cuda" else "cpu")
    record = dict(config=asdict(config), hypothesis=hypothesis, task_fingerprint=TASK.fingerprint,
                  git_commit=git_output("rev-parse", "HEAD"), git_diff=git_output("diff", "HEAD"),
                  train_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  backend=backend, device=str(device), training_seconds=0.0,
                  environment_transitions=0, learner_updates=0, rollout_iterations=0,
                  status="running", decision="unevaluated", errors=[], evaluation=None,
                  versions={name: version(name) for name in ("mani_skill", "torch", "numpy", "gymnasium", "sapien")})
    (output/"train_source.py").write_bytes(Path(__file__).read_bytes())
    (output/"run.json").write_text(json.dumps(record, indent=2)+"\n")
    env, agent = None, None
    try:
        env = make_environment(config, backend)
        agent = Agent(config).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        next_obs, _ = env.reset(seed=config.seed)
        while record["environment_transitions"] < config.total_timesteps and time.perf_counter() < deadline:
            rollout = []
            episode_summaries = []
            count = min(config.num_steps, (config.total_timesteps-record["environment_transitions"])//config.num_envs)
            for _ in range(count):
                if time.perf_counter() >= deadline:
                    break
                with torch.no_grad():
                    actions, logprob, _, values = agent.action_and_value(next_obs)
                observations = next_obs
                if not torch.isfinite(actions).all():
                    raise ValueError("nonfinite sampled action")
                # Store/log probability of sampled Gaussian actions; clip only for simulation.
                next_obs, reward, terminated, truncated, info = env.step(actions.clamp(-1, 1))
                record["environment_transitions"] += config.num_envs
                bootstrap = successor_values(agent, next_obs, terminated, truncated, info,
                                             config.bootstrap_on_termination)
                if not torch.isfinite(reward).all() or not torch.isfinite(bootstrap).all():
                    raise ValueError("nonfinite reward or bootstrap")
                if "final_info" in info:
                    mask = info["_final_info"]
                    metrics = info["final_info"]["episode"]
                    episode_summaries.append({key: value[mask].float().mean().item()
                                              for key, value in metrics.items()})
                rollout.append((observations, actions, logprob, reward, values, bootstrap, terminated | truncated))
            if not rollout:
                break
            fields = [torch.stack(items) for items in zip(*rollout)]
            observations, actions, logprob, rewards, values, bootstrap, dones = fields
            advantages, returns = generalized_advantage(rewards, values, bootstrap, dones,
                                                       config.gamma, config.gae_lambda)
            if config.anneal_lr:
                optimizer.param_groups[0]["lr"] = config.learning_rate * (1-(record["environment_transitions"]-len(rollout)*config.num_envs)/config.total_timesteps)
            batch = (observations.flatten(0, 1), actions.flatten(0, 1), logprob.flatten(),
                     advantages.flatten(), returns.flatten())
            updates, stats = ppo_update(agent, optimizer, batch, config, deadline)
            record["learner_updates"] += updates
            record["rollout_iterations"] += 1
            with (output/"progress.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(environment_transitions=record["environment_transitions"],
                                             learner_updates=record["learner_updates"],
                                             training_seconds=time.perf_counter()-started,
                                             training_episode_metrics=episode_summaries, **stats), allow_nan=False)+"\n")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["training_seconds"] = time.perf_counter()-started
        record["status"] = "step_budget" if record["environment_transitions"] >= config.total_timesteps else "time_budget"
        record["checkpoint_sha256"] = export_actor(agent, output/"policy")
        record["trained"] = record["learner_updates"] > 0
    except BaseException as exc:
        record["status"] = "error"
        record["training_seconds"] = time.perf_counter()-started
        record["errors"].append(type(exc).__name__)
        raise
    finally:
        (output/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        if env is not None:
            env.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for PPOConfig fields")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=["physx_cpu", "physx_cuda"], default="physx_cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline-seed1"))
    parser.add_argument("--hypothesis", default="Random-init PPO reference with frozen end-effector control")
    parser.add_argument("--dry-run", action="store_true", help="validate/print configuration without simulation or training")
    parser.add_argument("--check-env", action="store_true", help="zero-action wrapper check; no optimizer updates")
    args = parser.parse_args()
    config = PPOConfig(**json.loads(args.config.read_text())) if args.config else PPOConfig()
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    config.validate()
    TASK.env_kwargs(num_envs=config.num_envs, backend=args.backend)
    if args.dry_run:
        print(json.dumps(dict(config=asdict(config), backend=args.backend), indent=2))
    elif args.check_env:
        env = make_environment(config, args.backend)
        try:
            env.reset(seed=config.seed)
            for _ in range(TASK.horizon):
                obs, _, term, trunc, info = env.step(torch.zeros((config.num_envs, 7), device=env.device))
            if not trunc.all() or "final_observation" not in info:
                raise RuntimeError("expected horizon autoreset with final observations")
            print(json.dumps(dict(wrapper_check="passed", num_envs=config.num_envs, observation_dim=obs.shape[-1])))
        finally:
            env.close()
    else:
        print(json.dumps(run(config, args.backend, args.output, args.hypothesis), indent=2))


if __name__ == "__main__":
    main()
