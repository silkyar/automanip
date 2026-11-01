"""Random-initialized BC student; dataset actions, never teacher weights, are input."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import torch

from research.discovery.amp_v1 import StudentActor, export_actor, load_demos


def fit_bc(actor, obs, actions, *, updates, batch_size, seed, deadline=None):
    generator = torch.Generator(device=obs.device).manual_seed(seed+37001)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    initial = float((actor(obs[:min(4096, len(obs))])-actions[:min(4096, len(obs))]).square().mean().detach())
    completed = rows = 0
    loss = None
    for _ in range(updates):
        if deadline is not None and time.monotonic() >= deadline:
            break
        index = torch.randint(len(obs), (batch_size,), generator=generator, device=obs.device)
        prediction = actor(obs[index])
        objective = (prediction-actions[index]).square().mean()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite BC update")
        completed += 1
        rows += batch_size
        loss = float(objective.detach())
    return dict(initial_training_probe_mse=initial, last_training_batch_mse=loss,
                optimizer_updates=completed, demonstration_minibatch_rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-cpu-dataflow-smoke", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or not 1 <= args.max_seconds <= 600:
        raise ValueError("Invalid bounded BC budget")
    started = time.monotonic()
    external = float(os.environ["AUTOMANIP_TRAINING_DEADLINE"]) if args.device == "cuda" else started+args.max_seconds
    deadline = min(external, started+args.max_seconds)
    if not math.isfinite(deadline) or deadline <= started:
        raise ValueError("Expired BC lease")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.local_cpu_dataflow_smoke and args.device != "cpu":
        raise ValueError("Local CPU smoke data cannot be used in a production GPU student run")
    data = load_demos(args.dataset, allow_local_cpu_smoke=args.local_cpu_dataflow_smoke)
    obs, actions = data.behavior_rows(device=args.device)
    val_obs, val_actions = data.behavior_rows(validation=True, device=args.device)
    torch.manual_seed(args.seed)
    actor = StudentActor().to(args.device)
    initial = export_actor(actor, args.output/"initial-actor")
    stats = fit_bc(actor, obs, actions, updates=args.updates, batch_size=args.batch_size,
                   seed=args.seed, deadline=deadline-10)
    with torch.no_grad():
        validation_mse = float((actor(val_obs)-val_actions).square().mean())
    final = export_actor(actor, args.output/"final-actor")
    report = dict(
        track="prior-assisted random-student behavior cloning", teacher_weights_loaded=False,
        comparison_eligible=not args.local_cpu_dataflow_smoke,
        student_initialization="random", seed=args.seed, dataset_sha256=data.dataset_sha256,
        device=args.device, training_torch_version=torch.__version__, training_numpy_version=importlib.metadata.version("numpy"),
        teacher_prefix_transitions=data.metadata["teacher_prefix_transitions"],
        teacher_prefix_trainer_seconds=data.metadata["teacher_prefix_trainer_seconds"],
        teacher_prefix_dollars=data.metadata["teacher_prefix_dollars"],
        teacher_generation_transitions=data.metadata["environment_transitions"],
        total_dataset_episodes=data.metadata["completed_episodes"],
        failed_teacher_episodes_preserved=data.metadata["failed_completed_episodes"],
        training_demo_episodes=len(data.train_episode_indices), validation_demo_episodes=len(data.validation_episode_indices),
        train_episode_indices=data.train_episode_indices, validation_episode_indices=data.validation_episode_indices,
        validation_action_mse=validation_mse, validation_scope="public demonstration episode holdout, not protected robot scoring",
        bc_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_source_sha256=hashlib.sha256(Path(__file__).with_name("amp_v1.py").read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started, student_environment_transitions=0,
        dollars=None if args.device == "cuda" else 0., initial=initial, final=final, **stats)
    (args.output/"bc-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
