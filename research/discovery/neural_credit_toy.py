"""Fixed-data continuous-action neural value-learning gate, CPU only.

Known control: per-decision importance-sampled n-step TD residual targets.
Recursive candidate: established V-trace, rho untruncated, c=min(1,rho).
This explicitly does not propose a new algorithm or learn a robot policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

HORIZON = 12
GAMMA = .95
METHODS = ("one_step", "n_step_4", "n_step_full", "recursive_vtrace")


def make_data(seed, episodes=128):
    """Uniform continuous behavior actions, Gaussian dynamics/reward noise.

    Dataset is never filtered by reward, final outcome, policy or method.
    No simulator and no extra ground-truth training labels are used.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    states = torch.zeros(episodes, HORIZON + 1, device="cpu")
    states[:, 0] = .5 * torch.rand(episodes, generator=rng) - .25
    actions = 2 * torch.rand(episodes, HORIZON, generator=rng) - 1
    for t in range(HORIZON):
        states[:, t+1] = .9 * states[:, t] + .2 * actions[:, t] + .03 * torch.randn(episodes, generator=rng)
    rewards = torch.zeros_like(actions)
    rewards[:, -1] = 1 - (states[:, -1] - .4).square() + .2 * torch.randn(episodes, generator=rng)
    phase = torch.arange(HORIZON + 1).expand(episodes, -1) / HORIZON
    inputs = torch.stack((phase, states), dim=-1)
    return inputs, actions, rewards, (states[:, -1] - .4).abs() < .15


def oracle_values(inputs, target_k):
    """Analytic V under pi_k(a)=(1+k*a)/2, a in [-1,1].

    Means/variances follow the linear stochastic dynamics exactly. This oracle
    is ONLY a held-out toy diagnostic, never an optimization target.
    """
    steps = HORIZON - (inputs[..., 0] * HORIZON).round()
    mean = (.9 ** steps) * inputs[..., 1] + .2 * (target_k/3) * (1-.9**steps)/.1
    variance = (.04 * (1/3-target_k**2/9) + .03**2) * (1-.81**steps)/.19
    value = GAMMA ** (steps-1) * (1-variance-(mean-.4).square())
    return torch.where(steps > 0, value, torch.zeros_like(value))


def targets(values, rewards, rhos, method):
    """All methods consume the same immutable target-network snapshot.

    Vtrace G_t=V_t+rho_t*delta_t+gamma*c_t*(G_{t+1}-V_{t+1}).
    rho is NOT truncated (bounded by 1.75 here), so target policy remains pi.
    c truncates only traces. Terminal V=0; no cross-episode recursion.
    """
    if method not in METHODS: raise ValueError("unknown target method")
    delta = rhos * (rewards + GAMMA * values[:, 1:] - values[:, :-1])
    result = values[:, :-1].clone()
    if method == "recursive_vtrace":
        correction = torch.zeros_like(values[:, 0])
        for t in reversed(range(HORIZON)):
            correction = delta[:, t] + GAMMA * rhos[:, t].clamp(max=1) * correction
            result[:, t] += correction
    else:
        length = {"one_step": 1, "n_step_4": 4, "n_step_full": HORIZON}[method]
        for t in range(HORIZON):
            coefficient = torch.ones_like(values[:, 0])
            for j in range(t, min(HORIZON, t+length)):
                result[:, t] += coefficient * delta[:, j]
                coefficient *= GAMMA * rhos[:, j]
    return result


def critic(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32),
                         nn.Tanh(), nn.Linear(32, 1)).to(device="cpu")


def train(method, seed, target_k, data, updates=300):
    inputs, actions, rewards, success = data
    model = critic(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    # Identical index stream, weights, optimizer and target-refresh cadence.
    batch_rng = torch.Generator(device="cpu").manual_seed(10000+seed)
    rhos = 1 + target_k * actions  # pi/mu; mu(a)=1/2, full common support
    phase, x = torch.meshgrid(torch.arange(HORIZON)/HORIZON,
                              torch.linspace(-.6, .6, 41), indexing="ij")
    grid = torch.stack((phase, x), -1)
    truth = oracle_values(grid, target_k)
    history, frozen_targets = [], None
    with torch.no_grad():
        initial = float((model(grid).squeeze(-1)-truth).square().mean())
    for step in range(updates):
        if step % 20 == 0:
            with torch.no_grad():
                values = model(inputs).squeeze(-1)
                values[:, -1] = 0
                frozen_targets = targets(values, rewards, rhos, method).flatten()
        index = torch.randint(actions.numel(), (256,), generator=batch_rng)
        prediction = model(inputs[:, :-1].reshape(-1, 2)[index]).squeeze(-1)
        loss = (prediction - frozen_targets[index]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 20 == 0:
            with torch.no_grad():
                errors = (model(grid).squeeze(-1)-truth).square()
                history.append({"gradient_steps": step+1, "mse": float(errors.mean()),
                                "early_phase_mse": float(errors[:HORIZON//2].mean())})
    with torch.no_grad():
        errors = (model(grid).squeeze(-1)-truth).square()
    return {"method": method, "seed": seed, "target_k": target_k,
            "initial_grid_mse": initial, "final_grid_mse": float(errors.mean()),
            "early_phase_mse": float(errors[:HORIZON//2].mean()),
            "mean_checkpoint_mse": statistics.mean(v["mse"] for v in history),
            "gradient_steps": updates, "minibatch_presentations": updates*256,
            "target_snapshots": (updates+19)//20, "curve": history}


def run(seeds=6, updates=300):
    start, cpu_start = time.perf_counter(), time.process_time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rows, datasets = [], []
    for seed in range(seeds):
        data = make_data(seed)
        inputs, actions, rewards, success = data
        digest = hashlib.sha256(b"".join(t.numpy().tobytes() for t in data)).hexdigest()
        datasets.append({"seed": seed, "sha256": digest, "episodes": len(inputs),
                         "toy_transitions": actions.numel(), "successes": int(success.sum()),
                         "failures": int((~success).sum()), "outcomes_filtered": 0})
        for k in (0., .75):
            for method in METHODS:
                rows.append(train(method, seed, k, data, updates))
        print(f"completed toy seed {seed}", flush=True)
    aggregates = []
    for k in (0., .75):
        for method in METHODS:
            matching = [r for r in rows if r["method"] == method and r["target_k"] == k]
            aggregates.append({"target_k": k, "method": method,
                **{metric+suffix: fn(r[metric] for r in matching)
                   for metric in ("final_grid_mse", "early_phase_mse", "mean_checkpoint_mse")
                   for suffix, fn in (("_mean", statistics.mean), ("_sd", statistics.stdev))}})
    paired = []
    for k in (0., .75):
        for control in METHODS[:-1]:
            candidate = [r for r in rows if r["method"] == METHODS[-1] and r["target_k"] == k]
            baseline = [r for r in rows if r["method"] == control and r["target_k"] == k]
            diffs = [a["final_grid_mse"]-b["final_grid_mse"] for a,b in zip(candidate,baseline)]
            paired.append({"target_k": k, "control": control,
                "candidate_minus_control_mse_by_seed": diffs,
                "candidate_lower_mse_seeds": sum(v<0 for v in diffs),
                "mean_difference": statistics.mean(diffs)})
    return {"schema_version": 1, "scope": "fixed-data neural toy, not robotics or sample-efficiency proof",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": "cpu", "threads": 2,
        "design": {"horizon": HORIZON, "gamma": GAMMA, "training_seeds": list(range(seeds)),
            "updates_per_fit": updates, "target_refresh_updates": 20, "batch_size": 256,
            "behavior_density": "uniform[-1,1]", "target_density": "(1+k*a)/2 on [-1,1]",
            "policy_shift_k": [0., .75], "importance_ratio_range_shifted": [.25, 1.75],
            "rho_clipped": False, "vtrace_c": "min(1,rho)", "oracle_used_for_training": False,
            "target_network_schedule_matched": True, "data_and_initialization_paired": True},
        "datasets": datasets, "aggregate": aggregates, "paired_comparisons": paired, "runs": rows,
        "accounting": {"robot_environment_transitions": 0, "simulator_queries": 0,
            "robot_resets": 0, "toy_transitions_collected": sum(d["toy_transitions"] for d in datasets),
            "toy_episode_resets": sum(d["episodes"] for d in datasets),
            "critic_gradient_steps_total": len(rows)*updates,
            "minibatch_presentations_total": len(rows)*updates*256,
            "actor_gradient_steps": 0, "teacher_demo_planning_data": 0,
            "gpu_seconds": 0, "dollars_spent": 0, "paid_api_calls": 0,
            "wall_seconds": time.perf_counter()-start, "process_cpu_seconds": time.process_time()-cpu_start}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("aggregate", "paired_comparisons", "accounting")}, indent=2))
