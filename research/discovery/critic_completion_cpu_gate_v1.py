"""Cheap fixed-target fit/overfit counterexample for value-only data reuse."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from research.discovery.critic_kl_completion_v1 import critic_only_epoch


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_mean = nn.Linear(3, 1)
        self.actor_logstd = nn.Parameter(torch.tensor([[-.5]]))
        self.critic = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))


def true_value(obs):
    return torch.sin(3*obs[:, 0])+.25*obs[:, 1]+.25*obs[:, 2]


def heldout(agent):
    grid = torch.linspace(-1, 1, 31)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    parts = []
    for stage in (0., 1.):
        obs = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), stage)), -1)
        with torch.no_grad():
            parts.append(float((agent.critic(obs).flatten()-true_value(obs)).square().mean()))
    return dict(ordinary_phase_mse=parts[0], rare_phase_mse=parts[1], balanced_mse=sum(parts)/2)


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    records = []
    for seed in range(6):
        for biased in (False, True):
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(61000+seed)
            obs = torch.rand(256, 3, generator=generator)*2-1
            obs[:, 2] = (torch.rand(256, generator=generator) < .15).float()
            target = true_value(obs)+.2*torch.randn(256, generator=generator)+(2.*obs[:, 2] if biased else 0.)
            agent = Agent()
            optimizer = torch.optim.Adam(agent.parameters(), lr=.003, eps=1e-5)
            # Two value epochs stand in for actor-KL-truncated fitting. The actor
            # is fixed throughout: this is not a policy-learning experiment.
            for _ in range(2):
                critic_only_epoch(agent, optimizer, obs, target, 8, generator)
            before = heldout(agent)
            actor_before = {key: value.detach().clone() for key, value in agent.state_dict().items() if not key.startswith("critic.")}
            stats = critic_only_epoch(agent, optimizer, obs, target, 8, generator)
            after = heldout(agent)
            if any(not torch.equal(value, agent.state_dict()[key]) for key, value in actor_before.items()):
                raise AssertionError("Value fitting moved actor")
            records.append(dict(seed=seed, deliberately_biased_targets=biased, before=before, after=after, stats=stats))
    clean = [r for r in records if not r["deliberately_biased_targets"]]
    biased = [r for r in records if r["deliberately_biased_targets"]]
    summary = dict(clean_paired_balanced_mse_improvements=sum(r["after"]["balanced_mse"] < r["before"]["balanced_mse"] for r in clean),
                   biased_paired_rare_phase_regressions=sum(r["after"]["rare_phase_mse"] > r["before"]["rare_phase_mse"] for r in biased),
                   clean_mean_before=sum(r["before"]["balanced_mse"] for r in clean)/6,
                   clean_mean_after=sum(r["after"]["balanced_mse"] for r in clean)/6,
                   biased_rare_mean_before=sum(r["before"]["rare_phase_mse"] for r in biased)/6,
                   biased_rare_mean_after=sum(r["after"]["rare_phase_mse"] for r in biased)/6)
    report = dict(hypothesis="Independent value fitting can use existing rollout data after policy-KL stopping, while biased targets can make extra fit harmful.",
                  summary=summary, records=records, wall_seconds=time.monotonic()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256(Path(__file__).with_name("critic_kl_completion_v1.py").read_bytes()).hexdigest(),
                  accounting=dict(toy_fixed_target_rows=6*2*256, critic_optimizer_updates=6*2*3*32,
                                  critic_training_presentations=6*2*3*256, actor_optimizer_updates=0,
                                  robot_transitions=0, simulator_queries=0, teacher_demo_rows=0, gpu_seconds=0, dollars_spent=0),
                  limitations=["Frozen-actor regression toy, not a PPO learning curve or sample-efficiency result.",
                               "The deliberate phase-target bias is a counterexample, not measured StackCube critic bias.",
                               "More optimizer work is additional compute; no new independent experience is created."])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(summary=summary, wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
