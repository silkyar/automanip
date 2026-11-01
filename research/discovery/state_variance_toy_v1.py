"""Two-stage reward-only exploration/precision toy, with competent controls.

The toy environment alone knows the target function. Learners only get ordinary
observations, sampled actions and scalar terminal rewards; no target labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from research.discovery.state_variance_v1 import LogStdResidual, gaussian


class Agent(nn.Module):
    def __init__(self, state_variance=False):
        super().__init__()
        self.actor_mean = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.critic = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        with torch.no_grad():
            self.actor_mean[-1].weight.mul_(.01)
            self.actor_mean[-1].bias.zero_()
        self.actor_logstd = nn.Parameter(torch.full((1, 1), -.5))
        self.residual = LogStdResidual(3, 1, width=32) if state_variance else None

    def distribution(self, obs):
        mean = self.actor_mean(obs)
        if self.residual is None:
            return torch.distributions.Normal(mean, self.actor_logstd.expand_as(mean).exp())
        return gaussian(mean, self.actor_logstd, self.residual, obs)


def target(context):
    """Environment-only reward computation; never a training label/loss."""
    return .3*context + .1*context.pow(3)


def rollout(agent, context, noise):
    n = len(context)
    obs0 = torch.stack((torch.zeros(n), context, torch.zeros(n)), -1)
    distribution0 = agent.distribution(obs0)
    action0 = distribution0.loc + distribution0.scale*noise[:, :1]
    reached = action0.squeeze(-1).clamp(-1, 1).abs() > .45
    obs1 = torch.stack((torch.ones(n), context, reached.float()), -1)
    distribution1 = agent.distribution(obs1)
    action1 = distribution1.loc + distribution1.scale*noise[:, 1:]
    error = action1.squeeze(-1).clamp(-1, 1)-target(context)
    reward = reached.float()*torch.exp(-.5*(error/.04).square())
    logprob = torch.stack((distribution0.log_prob(action0).squeeze(-1), distribution1.log_prob(action1).squeeze(-1)), 0)
    return torch.stack((obs0, obs1)), torch.stack((action0, action1)), logprob, reward


def evaluate(agent):
    context = torch.linspace(-1, 1, 512)
    with torch.no_grad():
        obs0 = torch.stack((torch.zeros(512), context, torch.zeros(512)), -1)
        reached = agent.actor_mean(obs0).squeeze(-1).clamp(-1, 1).abs() > .45
        obs1 = torch.stack((torch.ones(512), context, reached.float()), -1)
        precise = (agent.actor_mean(obs1).squeeze(-1).clamp(-1, 1)-target(context)).abs() < .04
        std0 = agent.distribution(obs0).scale.mean()
        std1 = agent.distribution(obs1).scale.mean()
    return dict(successes=int((reached & precise).sum()), episodes=512,
                coarse_std=float(std0), fine_std=float(std1), deterministic_reach=int(reached.sum()))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    start = time.monotonic()
    iterations, episodes, epochs = 1000, 256, 4
    methods = ("global-std", "global-std-entropy-decay", "state-std")
    records = []
    for seed in range(6):
        torch.manual_seed(seed)
        base = Agent()
        generator = torch.Generator().manual_seed(1000+seed)
        contexts = torch.rand(iterations, episodes, generator=generator)*2-1
        noises = torch.randn(iterations, episodes, 2, generator=generator)
        for method in methods:
            if time.monotonic()-start > 240:
                raise TimeoutError("Four-minute CPU gate allowance exceeded")
            agent = copy.deepcopy(base)
            if method == "state-std":
                agent.residual = LogStdResidual(3, 1, width=32)
            optimizer = torch.optim.Adam(agent.parameters(), lr=.0008, eps=1e-5)
            curve = [dict(iteration=0, **evaluate(agent))]
            for iteration in range(iterations):
                with torch.no_grad():
                    obs, actions, old_logprob, reward = rollout(agent, contexts[iteration], noises[iteration])
                    obs, actions, old_logprob = obs.flatten(0, 1), actions.flatten(0, 1), old_logprob.flatten(0, 1)
                    returns = reward.repeat(2)
                    advantages = returns-agent.critic(obs).squeeze(-1)
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                entropy_coef = .003*(1-iteration/iterations) if method == "global-std-entropy-decay" else .003
                for _ in range(epochs):
                    distribution = agent.distribution(obs)
                    logprob = distribution.log_prob(actions).sum(-1)
                    ratio = (logprob-old_logprob).exp()
                    policy_loss = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages).mean()
                    value_loss = .5*(agent.critic(obs).squeeze(-1)-returns).square().mean()
                    loss = policy_loss + .5*value_loss - entropy_coef*distribution.entropy().sum(-1).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), .5)
                    optimizer.step()
                if (iteration+1) % 200 == 0:
                    curve.append(dict(iteration=iteration+1, **evaluate(agent)))
            records.append(dict(seed=seed, method=method, curve=curve))
    summary = {method: sum(r["curve"][-1]["successes"] for r in records if r["method"] == method)/(6*512) for method in methods}
    wins = 0
    for seed in range(6):
        result = {r["method"]: r["curve"][-1] for r in records if r["seed"] == seed}
        wins += result["state-std"]["successes"] > max(result[m]["successes"] for m in methods[:2])
    passed = summary["state-std"] >= max(summary[m] for m in methods[:2])+.05 and wins >= 4
    report = dict(
        hypothesis="State-conditioned per-step variance resolves exploration-versus-precision conflict without changing deterministic actor.",
        prespecified_gate="At least5percentage-point mean final success gain over best conventional control and strict paired wins over both in at least4/6 seeds.",
        gate_passed=passed, summary=summary, paired_wins_over_both_controls=wins, records=records,
        wall_seconds=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exploration_source_sha256=hashlib.sha256(Path(__file__).with_name("state_variance_v1.py").read_bytes()).hexdigest(),
        accounting=dict(toy_transitions=6*3*iterations*episodes*2,
                        actor_updates=6*3*iterations*epochs, critic_updates=6*3*iterations*epochs,
                        training_minibatch_rows=6*3*iterations*epochs*episodes*2,
                        deterministic_toy_eval_actions=6*3*6*512*2,
                        robot_transitions=0, simulator_queries=0, gpu_seconds=0, dollars_spent=0),
        limitations=["Hand-designed two-step toy, not robotic dynamics or a StackCube performance prediction.",
                     "State-variance head adds parameters/forward/backward computation; transition count alone is not compute equality.",
                     "Common random variates across methods do not imply equal state/action trajectories.",
                     "No optimal-action labels, demonstrations, teacher or analytic environment gradient enters training."],
    )
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("gate_passed", "summary", "paired_wins_over_both_controls", "wall_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
