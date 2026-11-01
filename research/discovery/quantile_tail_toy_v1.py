"""Rare-failure reward tradeoff: learned mean, mean/std, and quantile-tail actors.

Learners see shared ordinary (context, action, scalar reward) data, all outcomes.
Analytic probabilities and task success are held-out toy diagnostics only.
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

from research.discovery.quantile_tail_v1 import QuantileCritic, lower_tail, quantile_loss


def net(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))


def context_samples(count, generator):
    absolute = .2+.8*torch.rand(count, 1, generator=generator)
    sign = torch.where(torch.rand(count, 1, generator=generator) < .5, -1., 1.)
    return absolute*sign


def environment(context, action, uniform):
    speed = context*action.clamp(-1, 1)
    payout = .9+.1*torch.sigmoid(4*speed)
    failure_probability = .001+.06*torch.sigmoid(6*speed)
    return payout*(uniform >= failure_probability)


def evaluate(actor):
    # Analytic expectation avoids misreading a lucky zero-failure toy sample.
    context = torch.cat((-torch.linspace(.2, 1., 512), torch.linspace(.2, 1., 512))).unsqueeze(-1)
    with torch.no_grad():
        action = actor(context).clamp(-1, 1)
        speed = context*action
        payout = .9+.1*torch.sigmoid(4*speed)
        probability = .001+.06*torch.sigmoid(6*speed)
        tail = payout*(1-probability/.1).clamp_min(0)
    return dict(analytic_mean_success_probability=float((1-probability).mean()),
                analytic_expected_reward=float((payout*(1-probability)).mean()),
                analytic_conditional_cvar10=float(tail.mean()),
                mean_absolute_action=float(action.abs().mean()))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    start = time.monotonic()
    methods = ("quantile-mean", "moment-mean-minus-std", "quantile-cvar10")
    records = []
    for seed in range(6):
        if time.monotonic()-start > 240:
            raise TimeoutError("Four-minute CPU budget exceeded")
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(3000+seed)
        context = context_samples(16384, generator)
        action = torch.rand(16384, 1, generator=generator)*2-1
        reward = environment(context, action, torch.rand(16384, 1, generator=generator))
        critic = QuantileCritic(1, 1, width=64, quantiles=64)
        moments = net(2, 2)
        optimizer = torch.optim.Adam(list(critic.parameters())+list(moments.parameters()), lr=.002)
        indices = torch.randint(16384, (2000, 256), generator=generator)
        for index in indices:
            prediction = critic(context[index], action[index])
            prediction_moments = moments(torch.cat((context[index], action[index]), -1))
            target_moments = torch.cat((reward[index], reward[index].square()), -1)
            loss = quantile_loss(prediction, reward[index], critic.taus)+(prediction_moments-target_moments).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        critic.requires_grad_(False)
        moments.requires_grad_(False)
        base = net(1, 1)
        with torch.no_grad():
            base[-1].weight.mul_(.01)
            base[-1].bias.zero_()
        for method in methods:
            actor = copy.deepcopy(base)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=.002)
            curve = [dict(update=0, **evaluate(actor))]
            for update, index in enumerate(indices[:500], 1):
                obs = context[index]
                act = actor(obs).clamp(-1, 1)
                if method == "moment-mean-minus-std":
                    first, second = moments(torch.cat((obs, act), -1)).split(1, -1)
                    value = first-.5*(second-first.square()).clamp_min(1e-6).sqrt()
                else:
                    values = critic(obs, act)
                    value = values.mean(-1, keepdim=True) if method == "quantile-mean" else lower_tail(values, .1)
                actor_optimizer.zero_grad()
                (-value.mean()).backward()
                actor_optimizer.step()
                if update % 100 == 0:
                    curve.append(dict(update=update, **evaluate(actor)))
            with torch.no_grad():
                predicted = critic(context, action)
                crossed = (predicted[:, 1:] < predicted[:, :-1]).float().mean()
                coverage = (reward < predicted).float().mean(0)
                calibration = (coverage-critic.taus).abs().mean()
            records.append(dict(seed=seed, method=method, curve=curve,
                                training_quantile_crossing_fraction=float(crossed),
                                training_mean_quantile_coverage_error=float(calibration),
                                observed_training_failures=int((reward == 0).sum())))
    summary = {method: {key: sum(r["curve"][-1][key] for r in records if r["method"] == method)/6
                        for key in ("analytic_mean_success_probability", "analytic_expected_reward", "analytic_conditional_cvar10")}
               for method in methods}
    wins = 0
    for seed in range(6):
        row = {r["method"]: r["curve"][-1] for r in records if r["seed"] == seed}
        wins += row["quantile-cvar10"]["analytic_mean_success_probability"] > row["quantile-mean"]["analytic_mean_success_probability"]+.02
    passed = (wins >= 4 and summary["quantile-cvar10"]["analytic_mean_success_probability"] >= .99
              and summary["quantile-cvar10"]["analytic_mean_success_probability"] >= summary["moment-mean-minus-std"]["analytic_mean_success_probability"]-.01)
    report = dict(
        hypothesis="A lower-tail return objective can trade expected reward for reduced stochastic failure using ordinary scalar-reward data.",
        gate="CVaR mean success at least99%, at least2pp paired gain over mean in4/6 seeds, and within1pp of conventional mean-minus-std risk control.",
        gate_passed=passed, summary=summary, paired_two_point_wins=wins, records=records,
        wall_seconds=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_source_sha256=hashlib.sha256(Path(__file__).with_name("quantile_tail_v1.py").read_bytes()).hexdigest(),
        accounting=dict(toy_terminal_transitions=6*16384, quantile_critic_updates=6*2000,
                        moment_critic_updates=6*2000, actor_updates=6*3*500,
                        critic_minibatch_rows=6*2000*256, actor_minibatch_rows=6*3*500*256,
                        analytic_toy_diagnostic_contexts=6*3*6*1024,
                        robot_transitions=0, simulator_queries=0, gpu_seconds=0, dollars_spent=0),
        limitations=["Constructed reward/reliability conflict; no evidence StackCube has this particular tradeoff.",
                     "Conditional one-step CVaR, not an exact static episodic-CVaR algorithm.",
                     "Risk parameters .1 and .5 are fixed prespecified conventional choices, not claimed equivalent objectives.",
                     "Both critic training diagnostics use training data; no held-out calibration guarantee.",
                     "Different critic parameter counts/computation; common transitions and update counts are reported.",
                     "Analytic success/probabilities are diagnostic only, never learner losses or labels."],
    )
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("gate_passed", "summary", "paired_two_point_wins", "wall_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
