"""Bounded, standard-library falsification tests, not a robotics benchmark.

Run: python3 -m research.discovery.credit_toy --output research/discovery/credit-results.json
No simulator, policy checkpoint, environment variable, or benchmark file is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time


def replay_chain(horizon=50, gamma=.9, mode="reverse", seed=0, target="online"):
    """Reuse ONE deterministic trajectory until V(0) reaches 90% of truth.

    Each sweep updates every transition once. Frozen targets are snapshotted
    once per sweep, isolating why order alone cannot improve such a learner.
    No new environment transitions are generated during replay.
    """
    if horizon < 1 or mode not in {"forward", "shuffle", "reverse"}:
        raise ValueError("invalid replay configuration")
    if target not in {"online", "frozen"}:
        raise ValueError("invalid target mode")
    values = [0.] * (horizon + 1)
    rng = random.Random(seed)
    truth = gamma ** (horizon - 1)
    first = None
    for sweep in range(1, horizon + 1):
        order = list(range(horizon))
        if mode == "reverse": order.reverse()
        if mode == "shuffle": rng.shuffle(order)
        bootstrap = values if target == "online" else values.copy()
        for t in order:
            values[t] = float(t == horizon - 1) + gamma * bootstrap[t + 1]
        if first is None: first = values[0] / truth
        if values[0] >= .9 * truth:
            return {"sweeps_to_90pct_value": sweep, "bellman_updates": sweep * horizon,
                    "first_sweep_fraction_of_truth": first,
                    "toy_environment_transitions": horizon, "trajectory_reuses": sweep}
    raise AssertionError("deterministic chain should converge within horizon sweeps")


def first_advantage(horizon, gamma, lam, success_probability, outcome, exact_critic):
    """One action at t=0 chooses a branch; rewards arrive at t=H-1.

    The state retains the chosen branch. Before acting V is the mean of two
    branches (p=.2/.8); afterwards exact V knows branch probability, not outcome.
    Enumerating both Bernoulli outcomes avoids Monte Carlo estimation error.
    """
    values = [0.] * (horizon + 1)
    if exact_critic:
        values[0] = .5 * gamma ** (horizon - 1)
        for t in range(1, horizon):
            values[t] = success_probability * gamma ** (horizon - 1 - t)
    advantage = 0.
    for t in reversed(range(horizon)):
        reward = float(outcome) if t == horizon - 1 else 0.
        delta = reward + gamma * values[t + 1] - values[t]
        advantage = delta + gamma * lam * advantage
    return advantage


def gae_check(horizon=50, gamma=.9, lam=.9):
    def expected(p, exact):
        return sum(prob * first_advantage(horizon, gamma, lam, p, y, exact)
                   for y, prob in ((0, 1-p), (1, p)))
    gap = .6 * gamma ** (horizon - 1)
    exact = expected(.8, True) - expected(.2, True)
    zero = expected(.8, False) - expected(.2, False)
    return {"horizon": horizon, "gamma": gamma, "lambda": lam,
            "true_action_value_gap": gap, "exact_critic_expected_gae_gap": exact,
            "zero_critic_expected_gae_gap": zero,
            "exact_critic_gap_fraction": exact / gap,
            "zero_critic_gap_fraction": zero / gap,
            "toy_environment_transitions": 0, "enumerated_branch_outcomes": 4}


def selection_bias(seeds=256, episodes=64, probability=.2):
    """A rare-success branch exposes the danger of success-only replay.

    This is an estimator experiment, not SIL (whose objective differs).
    At least one success is required for the success-only estimate to exist.
    """
    uniform, selected = [], []
    missing = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        outcomes = [float(rng.random() < probability) for _ in range(episodes)]
        uniform.append(statistics.mean(outcomes))
        positives = [y for y in outcomes if y]
        if positives: selected.append(statistics.mean(positives))
        else: missing += 1
    return {"true_probability": probability, "independent_toy_seeds": seeds,
            "toy_episodes_per_seed": episodes,
            "uniform_estimate_mean": statistics.mean(uniform),
            "uniform_estimate_sd_across_seeds": statistics.stdev(uniform),
            "success_only_estimate_mean": statistics.mean(selected),
            "seeds_with_no_success": missing,
            "toy_bernoulli_draws": seeds * episodes,
            "robot_environment_transitions": 0}


def run():
    start = time.perf_counter()
    cpu_start = time.process_time()
    replay = {}
    for target in ("online", "frozen"):
        for mode in ("forward", "shuffle", "reverse"):
            runs = [replay_chain(mode=mode, seed=s, target=target) for s in range(256)]
            counts = [r["bellman_updates"] for r in runs]
            replay[f"{target}_{mode}"] = {
                "independent_toy_seeds": len(runs),
                "bellman_updates_mean": statistics.mean(counts),
                "bellman_updates_min": min(counts), "bellman_updates_max": max(counts),
                "bellman_updates_sd": statistics.stdev(counts),
                "first_sweep_fraction_of_truth_mean": statistics.mean(
                    r["first_sweep_fraction_of_truth"] for r in runs),
                "toy_environment_transitions_per_seed": 50,
                "robot_environment_transitions": 0,
            }
    return {"schema_version": 1, "experiment": "credit-order-and-bias-falsification",
            "scope": "analytical/tabular toy only; not StackCube evidence",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(), "replay": replay,
            "gae": gae_check(), "selection_bias": selection_bias(),
            "accounting": {"robot_environment_transitions": 0, "robot_resets": 0,
                "simulator_queries": 0, "neural_gradient_steps": 0,
                "extra_teacher_demonstration_or_planning_data": 0,
                "gpu_seconds": 0, "paid_api_calls": 0, "dollars_spent": 0,
                "process_cpu_seconds": time.process_time()-cpu_start,
                "wall_seconds": time.perf_counter()-start}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
