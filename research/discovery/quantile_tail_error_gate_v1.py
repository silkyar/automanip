"""Bounded critic-value error need not bound action-gradient error.

Analytic CPU counterexample only, not the error distribution of a learned robot
critic. It motivates validating derivatives or using observed-trajectory scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def check():
    action = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    payout = .9+.1*torch.sigmoid(4*action)
    failure_probability = .001+.06*torch.sigmoid(6*action)
    true_cvar = payout*(1-failure_probability/.1)
    approximate = true_cvar+.01*torch.sin(100*action)
    exact_gradient = torch.autograd.grad(true_cvar, action, retain_graph=True)[0]
    approximate_gradient = torch.autograd.grad(approximate, action)[0]
    if not (exact_gradient < 0 < approximate_gradient):
        raise AssertionError("Counterexample did not reverse the update direction")
    return dict(
        uniform_value_error_bound=.01, error_function=".01*sin(100*a)",
        evaluation_action=0., exact_cvar_action_gradient=float(exact_gradient),
        approximate_cvar_action_gradient=float(approximate_gradient),
        action_gradient_direction_reversed=True,
        conclusion="Small bounded value error alone does not certify a lower-tail actor gradient.",
        limitation="Analytic counterexample, not a measured error or failure prevalence in the neural toy or StackCube.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        robot_transitions=0, simulator_queries=0, optimizer_updates=0, gpu_seconds=0, dollars_spent=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))
