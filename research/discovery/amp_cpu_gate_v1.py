"""Small BC and transition-discriminator implementation gates; no robot claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from research.discovery.amp_bc_v1 import fit_bc
from research.discovery.amp_v1 import MotionPrior, StudentActor


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    records = []
    for seed in range(6):
        if time.monotonic()-started > 240:
            raise TimeoutError("Four-minute CPU gate exceeded")
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(41000+seed)
        obs = torch.randn(2048, 48, generator=generator)
        labels = torch.tanh(.3*obs[:, :8]+.1*obs[:, 8:16])
        validation = torch.randn(512, 48, generator=generator)
        val_labels = torch.tanh(.3*validation[:, :8]+.1*validation[:, 8:16])
        actor = StudentActor((32, 32, 32))
        with torch.no_grad():
            initial = float((actor(validation)-val_labels).square().mean())
        bc = fit_bc(actor, obs, labels, updates=400, batch_size=128, seed=seed)
        with torch.no_grad():
            final = float((actor(validation)-val_labels).square().mean())
        # Equal state marginals, opposite cross-transition dependence. The
        # discriminator must use transition structure rather than a label column.
        expert = torch.randn(2048, 48, generator=generator)
        expert_next = .8*expert+.1*torch.randn(2048, 48, generator=generator)
        policy = torch.randn(2048, 48, generator=generator)
        policy_next = -.8*policy+.1*torch.randn(2048, 48, generator=generator)
        prior = MotionPrior(expert, expert_next, seed=seed, capacity=4096)
        prior.reward_and_remember(policy, policy_next)
        stats = prior.update(rounds=200, batch_size=128)
        heldout = torch.randn(512, 48, generator=generator)
        with torch.no_grad():
            expert_style = prior.reward_and_remember(heldout, .8*heldout).mean()
            policy_style = prior.reward_and_remember(heldout, -.8*heldout).mean()
        records.append(dict(seed=seed, initial_validation_bc_mse=initial, final_validation_bc_mse=final,
                            bc_stats=bc, heldout_expert_style=float(expert_style), heldout_policy_style=float(policy_style),
                            discriminator_stats=stats))
    passed = all(r["final_validation_bc_mse"] < .02 and r["final_validation_bc_mse"] < .5*r["initial_validation_bc_mse"]
                 and r["heldout_expert_style"] > r["heldout_policy_style"]+.1 for r in records)
    report = dict(gate_passed=passed, gate="Each of6 seeds: BC heldout MSE below.02 and half initial, heldout expert style exceeds opposite-transition policy by.1.",
                  records=records, wall_seconds=time.monotonic()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  core_sha256=hashlib.sha256(Path(__file__).with_name("amp_v1.py").read_bytes()).hexdigest(),
                  bc_source_sha256=hashlib.sha256(Path(__file__).with_name("amp_bc_v1.py").read_bytes()).hexdigest(),
                  accounting=dict(synthetic_action_label_rows=6*2048, bc_updates=6*400,
                                  bc_minibatch_rows=6*400*128, synthetic_expert_transitions=6*2048,
                                  synthetic_policy_transitions=6*2048, discriminator_updates=6*200,
                                  discriminator_sample_rows=6*200*128*2,
                                  robot_transitions=0, simulator_queries=0, gpu_seconds=0, dollars_spent=0),
                  limitations=["Synthetic supervised action labels are disclosed toy demonstrations, not robot teacher data.",
                               "Discriminator classification and BC regression do not test closed-loop imitation or robotic sample efficiency.",
                               "No GPU or protected evaluator test was run."])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(gate_passed=passed, wall_seconds=report["wall_seconds"], records=records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
