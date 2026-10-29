"""Preregistered scalar-policy Bellman fit and actual-Critic mean constraint gate.

All data are synthetic scalar LQR transitions, embedded in48/8 for source reuse.
Analytic values are HELDOUT measurements only, never fitted labels or gradients.
No robots, actor learning, demonstration data or policy-performance claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from research.discovery.replay_learner import BatchRenorm, Critic

METHODS = ("target-sac", "joint-moments", "split-moments")
CONFIG = dict(width=64, heads=1, seeds=[1, 2, 3], steps=1000, batch_size=256,
              replay_rows=16384, heldout_rows=4096, learning_rate=.001,
              adam_betas=[.5, .999], target_tau=.005, gamma=.95,
              state_coefficient=.8, action_coefficient=.5, action_cost=.1,
              policy_gain=-.6, state_distribution="Uniform[-1,1]",
              behavior_action_distribution="independent Uniform[-1,1]",
              actor_updates=0, entropy_coefficient=0., warmup_updates=100000)


def transitions(count, generator):
    state = 2*torch.rand(count, 1, generator=generator)-1
    action = 2*torch.rand(count, 1, generator=generator)-1
    following = .8*state+.5*action
    reward = -(state.square()+.1*action.square())
    obs, acts = torch.zeros(count, 48), torch.zeros(count, 8)
    next_obs, next_acts = torch.zeros(count, 48), torch.zeros(count, 8)
    obs[:, :1], acts[:, :1] = state, action
    next_obs[:, :1], next_acts[:, :1] = following, -.6*following
    return dict(obs=obs, action=acts, next_obs=next_obs, next_action=next_acts,
                reward=reward, state=state, scalar_action=action, following=following)


def make_critic(method, linear_state):
    norm = None if method == "target-sac" else "split" if method == "split-moments" else "joint"
    critic = Critic(width=CONFIG["width"], normalization=norm)
    state = critic.state_dict()
    for name, value in linear_state.items():
        state[name].copy_(value)
    critic.load_state_dict(state)
    return critic


def prediction_and_target(critic, target, method, data):
    critic.train()
    if method == "target-sac":
        current = critic(data["obs"], data["action"])
        with torch.no_grad():
            future = target(data["next_obs"], data["next_action"])
    else:
        current, future = critic(torch.cat([data["obs"], data["next_obs"]]),
                                 torch.cat([data["action"], data["next_action"]])).chunk(2)
    return current, future, (data["reward"]+.95*future).detach()


@torch.no_grad()
def identity_gate(linear_state, data):
    result = {}
    for method in METHODS:
        critic = make_critic(method, linear_state)
        obs = torch.cat([data["obs"], data["next_obs"]])
        acts = torch.cat([data["action"], data["next_action"]])
        critic.train()
        first, second = critic(obs, acts).chunk(2)
        offset = critic.output.bias[0]
        if isinstance(critic.second_norm, BatchRenorm):
            offset = offset+(critic.output.weight[0]*critic.second_norm.bias).sum()
        row = dict(warm_training_mean_current=float(first.mean()), warm_training_mean_next=float(second.mean()),
                   warm_training_mean_difference=float((first.mean()-second.mean())),
                   final_affine_offset=float(offset),
                   warm_current_offset_error=float((first.mean()-offset).abs()),
                   warm_next_offset_error=float((second.mean()-offset).abs()))
        if method == "split-moments":
            if max(row["warm_current_offset_error"], row["warm_next_offset_error"]) > 1e-5:
                raise AssertionError("Actual warm split-Critic violates final affine mean identity")
        critic.eval()
        first, second = critic(obs, acts).chunk(2)
        row["running_inference_mean_difference"] = float(first.mean()-second.mean())
        for module in critic.modules():
            if isinstance(module, BatchRenorm):
                module.steps.fill_(module.warmup)
        critic.train()
        first, second = critic(obs, acts).chunk(2)
        row["postwarm_brn_training_mean_difference"] = float(first.mean()-second.mean())
        result[method] = row
    return result


@torch.no_grad()
def measure(critic, target, method, data):
    # These analytic quantities are computed ONLY here, after parameter updates.
    # No learner or minibatch function takes analytic labels as an input.
    coefficient = 1.036/(1-.95*.25)
    true_future = -coefficient*data["following"].double().square()
    true_current = data["reward"].double()+.95*true_future
    view = copy.deepcopy(critic)
    target_view = copy.deepcopy(target) if target is not None else None
    current, future, bellman_target = prediction_and_target(view, target_view, method, data)
    view.eval()
    current_eval = view(data["obs"], data["action"])
    future_eval = view(data["next_obs"], data["next_action"])
    def rmse(prediction, truth):
        return float((prediction.double()-truth).square().mean().sqrt())
    return dict(oracle_value_coefficient=coefficient,
                true_q_mean_current=float(true_current.mean()), true_q_mean_next=float(true_future.mean()),
                true_q_mean_difference=float(true_current.mean()-true_future.mean()),
                training_view_q_rmse=rmse(current, true_current),
                training_view_q_mean_bias=float((current.double()-true_current).mean()),
                actual_bootstrap_target_rmse=rmse(bellman_target, true_current),
                self_td_mse=float(F.mse_loss(current, bellman_target)),
                training_view_q_mean_difference=float(current.mean()-future.mean()),
                running_inference_q_rmse=rmse(current_eval, true_current),
                running_inference_q_mean_bias=float((current_eval.double()-true_current).mean()),
                running_inference_q_mean_difference=float(current_eval.mean()-future_eval.mean()))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(1)
    learner_path = Path("research/discovery/replay_learner.py")
    learner_sha = hashlib.sha256(learner_path.read_bytes()).hexdigest()
    generator = torch.Generator().manual_seed(812810)
    replay = transitions(CONFIG["replay_rows"], generator)
    heldout = transitions(CONFIG["heldout_rows"], generator)
    records, gate = [], None
    for seed in CONFIG["seeds"]:
        if time.monotonic()-started > 280:
            break
        torch.manual_seed(seed)
        initial = Critic(width=CONFIG["width"], normalization=None)
        linear_state = copy.deepcopy(initial.state_dict())
        if gate is None:
            gate = identity_gate(linear_state, {k: v[:256] for k, v in heldout.items()})
            print(json.dumps(dict(event="actual_critic_identity", values=gate)), flush=True)
        learners = {}
        for method in METHODS:
            critic = make_critic(method, linear_state)
            target = copy.deepcopy(critic).eval().requires_grad_(False) if method == "target-sac" else None
            opt = torch.optim.Adam(critic.parameters(), lr=.001, betas=(.5, .999))
            learners[method] = dict(critic=critic, target=target, optimizer=opt, recent_losses=[])
        index_generator = torch.Generator().manual_seed(812900+seed)
        indices = torch.randint(CONFIG["replay_rows"], (CONFIG["steps"], CONFIG["batch_size"]), generator=index_generator)
        updates = 0
        for step in range(CONFIG["steps"]):
            if time.monotonic()-started > 280:
                break
            batch = {k: v[indices[step]] for k, v in replay.items()}
            for method, learner in learners.items():
                q, _, target = prediction_and_target(learner["critic"], learner["target"], method, batch)
                loss = F.mse_loss(q, target)
                learner["optimizer"].zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in learner["critic"].parameters()):
                    raise FloatingPointError("Nonfinite fixed toy fit; stop, no rescue: "+method)
                learner["optimizer"].step()
                learner["recent_losses"].append(float(loss.detach()))
                if method == "target-sac":
                    with torch.no_grad():
                        for source, destination in zip(learner["critic"].parameters(), learner["target"].parameters()):
                            destination.lerp_(source, .005)
            updates += 1
        for method, learner in learners.items():
            row = dict(seed=seed, method=method, optimizer_calls=updates,
                       sampled_fitting_rows=updates*CONFIG["batch_size"],
                       critic_forward_rows=2*updates*CONFIG["batch_size"],
                       mean_last_100_fitting_td_mse=sum(learner["recent_losses"][-100:])/min(100, updates),
                       metrics=measure(learner["critic"], learner["target"], method, heldout))
            records.append(row)
            print(json.dumps(dict(event="fixed_toy_result", **row)), flush=True)
        if updates != CONFIG["steps"]:
            break
    elapsed = time.monotonic()-started
    complete = len(records) == 9 and all(row["optimizer_calls"] == CONFIG["steps"] for row in records)
    if hashlib.sha256(learner_path.read_bytes()).hexdigest() != learner_sha:
        raise RuntimeError("Owned Critic source changed during probe; record not eligible")
    report = dict(schema_version=1, state="complete" if complete else "budget_censored",
                  cpu_seconds=elapsed, config=CONFIG,
                  probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  actual_critic_source_sha256=learner_sha,
                  upstream_crossq_commit="a318a266679bec3f9d93c44e029860a6da077c89",
                  actual_critic_identity=gate, records=records,
                  total_optimizer_calls=sum(row["optimizer_calls"] for row in records),
                  total_sampled_fitting_rows=sum(row["sampled_fitting_rows"] for row in records),
                  unique_synthetic_training_transitions=CONFIG["replay_rows"],
                  synthetic_heldout_transitions=CONFIG["heldout_rows"],
                  new_robot_simulator_transitions=0, actor_updates=0, teacher_rows=0, gpu_seconds=0, dollars=0,
                  limitations=["One head, scalar deterministic fixed-policy toy; no entropy, exploration or contact dynamics.",
                               "Analytic values are measurement-only, never optimizer inputs.",
                               "Fixed replay and common minibatches; three initial-weight/sampler seeds, not robot training seeds.",
                               "No tuning, extension or retries permitted to rescue this fixed result."])
    if elapsed > 300:
        report["state"] = "time_cap_failed"
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "actual_critic_identity")}, indent=2))
