"""Small AMP-style manipulation prior and safe BC actor for an assisted track.

No simulator or evaluator imports. Teacher generation and BC/AMP compute are
additional costs. Full48→48 transition features make this a task-specific prior,
not the task-agnostic kinematic representation of the original AMP paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK_FINGERPRINT = "b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"
DATA_FORMAT = "automanip-public-training-demos-v1"
PRODUCTION_PINS = {"mani_skill": "3.0.0b21", "sapien": "3.0.0b1", "torch": "2.6.0+cu124",
                   "numpy": "1.26.4", "gymnasium": "0.29.1"}


class StudentActor(nn.Module):
    def __init__(self, hidden_sizes=(256, 256, 256)):
        super().__init__()
        if not 1 <= len(hidden_sizes) <= 3 or any(type(v) is not int or not 1 <= v <= 256 for v in hidden_sizes):
            raise ValueError("Unsupported public safe actor architecture")
        self.hidden_sizes = list(hidden_sizes)
        layers = []
        widths = [48, *hidden_sizes, 8]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(hidden_sizes):
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def deterministic(self, obs):
        return self(obs).clamp(-1, 1)


def export_actor(actor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    linear = [layer for layer in actor.net if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear):
        for key in ("weight", "bias"):
            value = getattr(layer, key).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite student actor")
            tensors[f"layers.{i}.{key}"] = dict(shape=list(value.shape), values=value.flatten().tolist())
    raw = json.dumps(tensors, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = dict(format="automanip-json-f32-v1", task_fingerprint=TASK_FINGERPRINT,
                    observation_dim=48, action_dim=8, hidden_sizes=actor.hidden_sizes,
                    activation="tanh", weights_sha256=digest)
    (directory/"weights.json").write_bytes(raw)
    (directory/"manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return dict(path=str(directory), weights_sha256=digest)


def read_actor(directory, device="cpu"):
    """Candidate-side numeric reader; never substitutes for trusted validation."""
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest["format"] != "automanip-json-f32-v1" or manifest["task_fingerprint"] != TASK_FINGERPRINT
            or manifest["activation"] != "tanh" or manifest["observation_dim"] != 48 or manifest["action_dim"] != 8):
        raise ValueError("Public actor contract mismatch")
    raw = (directory/"weights.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Actor digest mismatch")
    actor = StudentActor(tuple(manifest["hidden_sizes"]))
    data = json.loads(raw)
    expected = set()
    with torch.no_grad():
        for i, layer in enumerate(layer for layer in actor.net if isinstance(layer, nn.Linear)):
            for key in ("weight", "bias"):
                name = f"layers.{i}.{key}"
                expected.add(name)
                target = getattr(layer, key)
                entry = data[name]
                if entry["shape"] != list(target.shape):
                    raise ValueError("Actor tensor shape mismatch")
                value = torch.tensor(entry["values"], dtype=torch.float32).reshape(target.shape)
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite actor tensor")
                target.copy_(value)
    if set(data) != expected:
        raise ValueError("Unexpected actor tensors")
    return actor.to(device), manifest


@dataclass
class DemoData:
    observations: torch.Tensor
    actions: torch.Tensor
    task_rewards: torch.Tensor
    final_success: torch.Tensor
    train_episode_indices: list[int]
    validation_episode_indices: list[int]
    metadata: dict
    dataset_sha256: str

    def training_pairs(self, device="cpu"):
        episodes = self.observations[self.train_episode_indices]
        return episodes[:, :-1].reshape(-1, 48).to(device), episodes[:, 1:].reshape(-1, 48).to(device)

    def behavior_rows(self, validation=False, device="cpu"):
        index = self.validation_episode_indices if validation else self.train_episode_indices
        return (self.observations[index, :-1].reshape(-1, 48).to(device),
                self.actions[index].reshape(-1, 8).to(device))


def load_demos(directory, split_seed=0, allow_local_cpu_smoke=False):
    directory = Path(directory)
    metadata = json.loads((directory/"dataset.json").read_text())
    raw_path = directory/"trajectories.npz"
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if metadata["format"] != DATA_FORMAT or metadata["sha256"] != digest:
        raise ValueError("Demo format or digest mismatch")
    expected = dict(env_id="StackCube-v1", observation_dim=48, action_dim=8, horizon=50,
                    control_mode="pd_joint_delta_pos", obs_mode="state", reward_mode="normalized_dense",
                    simulation_backend="physx_cuda", source_partition="public-training")
    if allow_local_cpu_smoke:
        expected["simulation_backend"] = "physx_cpu"
        if (metadata.get("purpose") != "local-cpu-dataflow-smoke" or metadata.get("comparison_eligible") is not False
                or metadata.get("versions", {}).get("sapien") != "3.0.4.dev20261009+7dba23f1"):
            raise ValueError("Explicit local CPU smoke provenance missing")
    elif any(metadata.get("versions", {}).get(k) != v for k, v in PRODUCTION_PINS.items()):
        raise ValueError("Production demonstration software pins mismatch")
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Demo public task or training-partition contract mismatch")
    with np.load(raw_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"observations", "actions", "task_rewards", "final_success", "public_training_seed"}:
            raise ValueError("Unexpected demo arrays")
        obs, actions, rewards = (arrays[k] for k in ("observations", "actions", "task_rewards"))
        success, seeds = arrays["final_success"], arrays["public_training_seed"]
        episodes = len(obs)
        if obs.shape != (episodes, 51, 48) or actions.shape != (episodes, 50, 8) or rewards.shape != (episodes, 50):
            raise ValueError("Demo episode shape mismatch")
        if success.shape != (episodes,) or success.dtype != np.bool_ or seeds.shape != (episodes,) or seeds.dtype != np.int64:
            raise ValueError("Demo outcome/seed dtype mismatch")
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (obs, actions, rewards)):
            raise ValueError("Demo tensors must be finite float32")
        if np.max(np.abs(actions)) > 1.000001:
            raise ValueError("BC labels must be executed bounded teacher actions")
        if metadata["completed_episodes"] != episodes or metadata["successful_episodes"] != int(success.sum()):
            raise ValueError("Demo accounting mismatch")
        selected = np.flatnonzero(success).tolist()
        if len(selected) < 4:
            raise ValueError("At least4 successful training episodes required for BC split")
        ordered = sorted(selected, key=lambda i: hashlib.sha256(f"{digest}:{split_seed}:{i}".encode()).digest())
        validation_count = max(1, len(ordered)//10)
        validation, train = sorted(ordered[:validation_count]), sorted(ordered[validation_count:])
        return DemoData(torch.from_numpy(obs.copy()), torch.from_numpy(actions.copy()), torch.from_numpy(rewards.copy()),
                        torch.from_numpy(success.copy()), train, validation, metadata, digest)


def actual_successor(next_obs, done, infos):
    actual = next_obs.detach().clone()
    if done.bool().any():
        final = infos.get("final_observation")
        mask = infos.get("_final_observation")
        if final is None or mask is None or not torch.equal(mask.bool(), done.bool()) or final.shape != next_obs.shape:
            raise ValueError("AMP transition needs exact pre-reset final observations")
        actual[done.bool()] = final[done.bool()]
    if actual.shape[-1] != 48 or not torch.isfinite(actual).all():
        raise ValueError("Invalid AMP successor")
    return actual


def amp_style_reward(score):
    return (1-.25*(score.detach()-1).square()).clamp(0., 1.)


def mix_training_reward(task_reward, style_reward, weight=.25):
    if task_reward.shape != style_reward.shape or not 0 <= weight <= 1:
        raise ValueError("Training reward shape or declared weight invalid")
    if not torch.isfinite(task_reward).all() or not torch.isfinite(style_reward).all():
        raise ValueError("Nonfinite training reward")
    return task_reward.detach()+weight*style_reward.detach()


class TransitionDiscriminator(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clamp_min(.1).clone())
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(96, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, obs, next_obs):
        return torch.cat((((obs-self.mean)/self.std).clamp(-10, 10),
                          ((next_obs-self.mean)/self.std).clamp(-10, 10)), -1)

    def forward(self, features):
        return self.net(features).squeeze(-1)


def discriminator_loss(discriminator, expert_features, policy_features, gradient_penalty=10.):
    expert = expert_features.detach().requires_grad_(True)
    policy = policy_features.detach()
    expert_score, policy_score = discriminator(expert), discriminator(policy)
    gradient = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
    penalty = gradient.square().sum(-1).mean()
    least_squares = (expert_score-1).square().mean()+(policy_score+1).square().mean()
    return least_squares+.5*gradient_penalty*penalty, dict(
        least_squares=float(least_squares.detach()), gradient_penalty=float(penalty.detach()),
        expert_score=float(expert_score.detach().mean()), policy_score=float(policy_score.detach().mean()))


class MotionPrior:
    def __init__(self, expert_obs, expert_next_obs, seed, device="cpu", capacity=100000):
        obs = expert_obs.detach().to(device)
        next_obs = expert_next_obs.detach().to(device)
        stats = torch.cat((obs, next_obs))
        self.discriminator = TransitionDiscriminator(stats.mean(0), stats.std(0, unbiased=False)).to(device)
        with torch.no_grad():
            self.expert = self.discriminator.features(obs, next_obs)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.buffer = torch.empty(capacity, 96, device=device)
        self.capacity, self.position, self.size = capacity, 0, 0
        self.generator = torch.Generator(device=device).manual_seed(seed+27001)
        self.updates = self.sample_rows = self.transitions_added = 0

    @torch.no_grad()
    def reward_and_remember(self, obs, next_obs):
        features = self.discriminator.features(obs.detach(), next_obs.detach())
        reward = amp_style_reward(self.discriminator(features))
        count = len(features)
        if count > self.capacity:
            raise ValueError("Transition batch exceeds prior replay")
        index = (torch.arange(count, device=self.buffer.device)+self.position)%self.capacity
        self.buffer[index] = features
        self.position = (self.position+count)%self.capacity
        self.size = min(self.capacity, self.size+count)
        self.transitions_added += count
        return reward

    def update(self, rounds=32, batch_size=256):
        if self.size < 1:
            raise ValueError("Empty learner transition buffer")
        totals = {}
        for _ in range(rounds):
            expert_index = torch.randint(len(self.expert), (batch_size,), generator=self.generator, device=self.buffer.device)
            policy_index = torch.randint(self.size, (batch_size,), generator=self.generator, device=self.buffer.device)
            loss, stats = discriminator_loss(self.discriminator, self.expert[expert_index], self.buffer[policy_index])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite discriminator update")
            self.updates += 1
            self.sample_rows += 2*batch_size
            for key, value in dict(loss=float(loss.detach()), **stats).items():
                totals[key] = totals.get(key, 0.)+value/rounds
        return dict(totals, discriminator_updates=self.updates, discriminator_sample_rows=self.sample_rows,
                    learner_transition_rows=self.transitions_added, replay_rows=self.size)
