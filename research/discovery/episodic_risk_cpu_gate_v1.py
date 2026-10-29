"""Exact score-gradient and payload gate; no simulator/evaluator imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from research.discovery.episodic_risk_v1 import build_trainer, score_weights


def outcomes(theta):
    p = 1/(1+math.exp(-theta))
    rows = []
    for first, second, reward, probability in ((0, 0, .9, 1.), (0, 1, 1.1, 1.),
                                             (1, 0, .8, 1.), (1, 1, 0., .1), (1, 1, 1.3, .9)):
        count = first+second
        mass = p**count*(1-p)**(2-count)*probability
        rows.append((reward, mass, count-2*p))
    return rows


def exact_cvar(theta, alpha):
    remaining, result = alpha, 0.
    for reward, mass, _ in sorted(outcomes(theta)):
        take = min(remaining, mass)
        result += take*reward
        remaining -= take
        if remaining <= 1e-15:
            break
    return result/alpha


def analytic_check():
    records = []
    for p in (.2, .4, .7):
        theta = math.log(p/(1-p))
        rows = outcomes(theta)
        # Exact rational probability mass represented by1000 rows. No random data
        # collection, optimization or favorable sample selection in this check.
        reward, joint_score = [], []
        for r, mass, score in rows:
            count = round(1000*mass)
            reward.extend([r]*count)
            joint_score.extend([score]*count)
        reward = torch.tensor(reward, dtype=torch.float64)
        joint_score = torch.tensor(joint_score, dtype=torch.float64)
        for alpha in (.2, .5, 1.):
            weights, threshold = score_weights(reward, alpha)
            estimated = float((weights*joint_score).mean())
            finite_difference = (exact_cvar(theta+1e-5, alpha)-exact_cvar(theta-1e-5, alpha))/2e-5
            error = abs(estimated-finite_difference)
            if error > 1e-8:
                raise AssertionError((p, alpha, estimated, finite_difference))
            if alpha == 1.:
                expected_mean_gradient = sum(r*mass*score for r, mass, score in rows)
                if abs(estimated-expected_mean_gradient) > 1e-12:
                    raise AssertionError("alpha1 did not reduce to mean-return score")
            wrong = float((joint_score*(reward-.5)*(reward <= threshold)/alpha).mean())
            records.append(dict(policy_probability=p, alpha=alpha, var=float(threshold),
                                score_gradient=estimated, exact_finite_difference=finite_difference,
                                absolute_error=error, incorrectly_tail_centered_gradient=wrong))
    return records


def run(output, official):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    records = analytic_check()
    payload = build_trainer(official.read_bytes())
    result = dict(
        check="Two-step stochastic Bernoulli policy, shared parameter; complete-episode joint score, exact probability enumeration.",
        records=records, maximum_absolute_error=max(r["absolute_error"] for r in records),
        alpha1_matches_conventional_mean=True,
        ordinary_baseline_inside_tail_counterexample=any(abs(r["score_gradient"]-r["incorrectly_tail_centered_gradient"]) > .01 for r in records if r["alpha"] < 1),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(__file__).with_name("episodic_risk_v1.py").read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,
        robot_transitions=0, simulator_queries=0, optimizer_updates=0, gpu_seconds=0, dollars_spent=0,
        limitations=["Population score identity at behavior policy only; empirical VaR, RMS scaling, clipping and multiple epochs remain approximations.",
                     "Analytic toy, not StackCube policy performance or a protected-evaluator test."],
    )
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("maximum_absolute_error", "payload_sha256", "wall_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.official)
