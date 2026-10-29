"""Falsification gate for a learned Euclidean contraction auxiliary loss.

CPU toy only. Learned dynamics and direct-reward critics see the same ordinary
transitions. No analytic policy, optimal action, or dynamics Jacobian is a label.
This is a contraction-inspired regularizer, not a stability certificate.
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


def network(inputs, outputs, width=32):
    return nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width),
                         nn.Tanh(), nn.Linear(width, outputs))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = network(2, 1)
        with torch.no_grad():
            self.net[-1].weight.mul_(.01)
            self.net[-1].bias.zero_()

    def forward(self, state):
        return self.net(state).clamp(-1, 1)


def dynamics(state, action, noise, hybrid=False):
    x, velocity = state.unbind(-1)
    gain = torch.where(x < 0, -.7, 1.) if hybrid else torch.ones_like(x)
    next_velocity = .8*velocity + gain*action.squeeze(-1) + .002*noise
    return torch.stack((x + .2*next_velocity, next_velocity), -1)


def closed_loop_jacobian(model, actor, state):
    state = state.detach().requires_grad_(True)
    predicted = state + model(torch.cat((state, actor(state)), -1))
    return torch.stack([torch.autograd.grad(predicted[:, j].sum(), state,
                                           create_graph=True, retain_graph=True)[0]
                        for j in range(state.shape[-1])], 1)


def contraction_loss(model, actor, state):
    jacobian = closed_loop_jacobian(model, actor, state)
    largest_singular = torch.linalg.svdvals(jacobian)[:, 0]
    return (largest_singular-.99).clamp_min(0).square().mean()


def evaluate(actor, seed, hybrid):
    generator = torch.Generator().manual_seed(seed)
    state = (torch.rand(256, 2, generator=generator)-.5)*torch.tensor([.8, .4])
    noise = torch.randn(50, 256, generator=generator)
    with torch.no_grad():
        for t in range(50):
            state = dynamics(state, actor(state), noise[t], hybrid)
        energy = state[:, 0].square() + .1*state[:, 1].square()
        success = (state[:, 0].abs() < .025) & (state[:, 1].abs() < .03)
    return dict(successes=int(success.sum()), episodes=256,
                mean_final_energy=float(energy.mean()),
                final_energy_p95=float(torch.quantile(energy, .95)))


def run(output: Path):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    start = time.monotonic()
    records = []
    methods = ("direct-reward-Q", "Q+spatial-smoothness", "Q+learned-contraction")
    for hybrid in (False, True):
        for seed in range(6):
            if time.monotonic()-start > 240:
                raise TimeoutError("CPU gate exceeded four-minute compute allowance")
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(1000+seed)
            state = (torch.rand(4096, 2, generator=generator)-.5)*torch.tensor([.8, .4])
            action = torch.rand(4096, 1, generator=generator)-.5
            successor = dynamics(state, action, torch.randn(4096, generator=generator), hybrid)
            reward = -(successor[:, :1].square()+.1*successor[:, 1:].square()+.01*action.square())
            model, critic = network(3, 2), network(3, 1)
            optimizer = torch.optim.Adam(list(model.parameters())+list(critic.parameters()), lr=.002)
            inputs = torch.cat((state, action), -1)
            indices = torch.randint(4096, (600, 128), generator=generator)
            for index in indices:
                model_loss = (model(inputs[index])-(successor-state)[index]).square().mean()
                q_loss = (critic(inputs[index])-reward[index]).square().mean()
                optimizer.zero_grad()
                (model_loss+q_loss).backward()
                optimizer.step()
            model.requires_grad_(False)
            critic.requires_grad_(False)
            actor = Actor()
            initial = evaluate(actor, 2000+seed, hybrid)
            for method in methods:
                policy = copy.deepcopy(actor)
                actor_optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
                for index in indices[:400]:
                    batch = state[index]
                    loss = -critic(torch.cat((batch, policy(batch)), -1)).mean()
                    if method == "Q+spatial-smoothness":
                        differentiable = batch.detach().requires_grad_(True)
                        jacobian = torch.autograd.grad(policy(differentiable).sum(), differentiable, create_graph=True)[0]
                        loss = loss + .01*jacobian.square().sum(-1).mean()
                    elif method == "Q+learned-contraction":
                        loss = loss + .1*contraction_loss(model, policy, batch)
                    actor_optimizer.zero_grad()
                    loss.backward()
                    actor_optimizer.step()
                record = dict(hybrid=hybrid, seed=seed, method=method,
                              initial=initial, final=evaluate(policy, 2000+seed, hybrid))
                with torch.no_grad():
                    record["training_model_delta_mse"] = float((model(inputs)-(successor-state)).square().mean())
                    record["training_q_mse"] = float((critic(inputs)-reward).square().mean())
                records.append(record)
    summary = []
    for hybrid in (False, True):
        grouped = {method: [r for r in records if r["hybrid"] == hybrid and r["method"] == method] for method in methods}
        control = grouped[methods[0]]
        candidate = grouped[methods[2]]
        summary.append(dict(
            hybrid=hybrid,
            means={method: sum(r["final"]["successes"] for r in rows)/(6*256) for method, rows in grouped.items()},
            candidate_paired_wins=sum(c["final"]["successes"] > b["final"]["successes"] for c, b in zip(candidate, control)),
            candidate_five_percentage_point_wins=sum(c["final"]["successes"]-b["final"]["successes"] >= 13 for c, b in zip(candidate, control)),
        ))
    passed = all(s["candidate_five_percentage_point_wins"] >= 4 for s in summary)
    report = dict(
        hypothesis="Learned Euclidean closed-loop contraction improves local final-state reliability beyond direct reward optimization and spatial smoothness.",
        gate="Candidate gains at least13/256 successes versus direct reward Q in at least4/6 paired seeds, in both smooth and hybrid plants.",
        gate_passed=passed, decision="continue only if gate passed" if passed else "stop this Euclidean-contraction version",
        summary=summary, records=records, wall_seconds=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accounting=dict(toy_transition_data=2*6*4096, model_updates=2*6*600,
                        direct_reward_q_updates=2*6*600, actor_updates=2*6*3*400,
                        training_sample_rows=2*6*600*128,
                        actor_sample_rows=2*6*3*400*128,
                        heldout_toy_evaluation_steps=2*6*4*50*256,
                        robot_transitions=0, simulator_queries=0, gpu_seconds=0, dollars_spent=0),
        limitations=["Offline one-step local toy, not a full RL control.",
                     "Dynamics/Q fits shared across actor methods; contraction requires additional second-derivative computation.",
                     "Euclidean units/metric are fixed by this toy; no learned certificate or stability guarantee.",
                     "Hybrid actuation flip is a synthetic contact-style stress, not a model of StackCube contacts."],
    )
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(gate_passed=passed, summary=summary, wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
