"""Bounded discriminator dataflow probe on explicitly supplied Mac CPU data.

All robot trajectories are pre-collected fresh PUBLIC TRAINING data. No new
simulator query, protected evaluator, upload or cloud operation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.discovery.amp_v1 import MotionPrior, amp_style_reward, load_demos


def summarize(prior, observations, successes=None):
    obs, following = observations[:, :-1].reshape(-1, 48), observations[:, 1:].reshape(-1, 48)
    with torch.no_grad():
        features = prior.discriminator.features(obs, following)
        score = prior.discriminator(features)
        style = amp_style_reward(score).reshape(len(observations), 50)
        result = dict(episodes=len(observations), transition_rows=len(features), mean_score=float(score.mean()),
                      fraction_score_above_zero=float((score > 0).float().mean()),
                      mean_style=float(style.mean()), last10_mean_style=float(style[:, -10:].mean()),
                      minimum_style=float(style.min()), maximum_style=float(style.max()))
        if successes is not None:
            for label, mask in (("success", successes), ("failure", ~successes)):
                result[label+"_episodes"] = int(mask.sum())
                result[label+"_last10_style"] = float(style[mask, -10:].mean()) if mask.any() else None
    return result


def run(teacher_directory, learner_directory, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    teacher = load_demos(teacher_directory, allow_local_cpu_smoke=True)
    collection = json.loads((learner_directory/"collection.json").read_text())
    if collection.get("comparison_eligible") is not False or collection.get("purpose") != "local-cpu-discriminator-dataflow-smoke":
        raise ValueError("Explicit fresh-training CPU probe data required")
    arrays = {}
    for arm in ("random", "bc"):
        path = learner_directory/(arm+".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != collection["arms"][arm]["sha256"]:
            raise ValueError("Learner data digest mismatch")
        with np.load(path, allow_pickle=False) as source:
            obs, success = source["observations"], source["final_success"]
            if obs.shape != (64, 51, 48) or obs.dtype != np.float32 or not np.isfinite(obs).all() or success.dtype != np.bool_:
                raise ValueError("Learner observation contract mismatch")
            arrays[arm] = (torch.from_numpy(obs.copy()), torch.from_numpy(success.copy()))
    digest = collection["arms"]["bc"]["sha256"]
    ordered = sorted(range(64), key=lambda i: hashlib.sha256(f"{digest}:probe-v1:{i}".encode()).digest())
    validation, training = sorted(ordered[:12]), sorted(ordered[12:])
    expert_obs, expert_next = teacher.training_pairs()
    torch.manual_seed(1)
    prior = MotionPrior(expert_obs, expert_next, seed=1, capacity=100000)
    bc_obs, bc_success = arrays["bc"]
    prior.reward_and_remember(bc_obs[training, :-1].reshape(-1, 48), bc_obs[training, 1:].reshape(-1, 48))
    expert_holdout = teacher.observations[teacher.validation_episode_indices]
    initial = dict(expert_holdout=summarize(prior, expert_holdout),
                   bc_holdout=summarize(prior, bc_obs[validation], bc_success[validation]))
    stats = prior.update(rounds=200, batch_size=256)
    if time.monotonic()-started > 50:
        raise TimeoutError("Local discriminator probe compute exceeded50seconds")
    final = dict(expert_holdout=summarize(prior, expert_holdout),
                 bc_training=summarize(prior, bc_obs[training], bc_success[training]),
                 bc_holdout=summarize(prior, bc_obs[validation], bc_success[validation]),
                 random_out_of_distribution=summarize(prior, *arrays["random"]))
    with torch.no_grad():
        shifted = expert_holdout[:, 1:].roll(25, 1).reshape(-1, 48)
        score = prior.discriminator(prior.discriminator.features(expert_holdout[:, :-1].reshape(-1, 48), shifted))
        shifted_style = float(amp_style_reward(score).mean())
    style_gap = final["expert_holdout"]["mean_style"]-final["bc_holdout"]["mean_style"]
    accuracy = .5*(final["expert_holdout"]["fraction_score_above_zero"]+1-final["bc_holdout"]["fraction_score_above_zero"])
    signal_gate = style_gap >= .05 and accuracy >= .6
    report = dict(
        purpose="local-cpu-discriminator-dataflow-smoke", comparison_eligible=False, formal_benchmark_result=False,
        teacher_dataset_sha256=teacher.dataset_sha256, learner_collection=collection,
        teacher_training_episodes=len(teacher.train_episode_indices), teacher_validation_episodes=len(teacher.validation_episode_indices),
        bc_training_episodes=training, bc_validation_episodes=validation,
        all_failed_bc_episodes_retained=True, actor_actions_or_success_as_discriminator_input=False,
        initial=initial, final=final, discriminator_training=stats,
        heldout_style_gap=style_gap, heldout_balanced_sign_accuracy=accuracy,
        shifted_successor_expert_style=shifted_style,
        real_minus_shifted_expert_style=final["expert_holdout"]["mean_style"]-shifted_style,
        prespecified_signal_gate="Held-out teacher-minus-BC style>=.05 and balanced sign accuracy>=.6; source/runtime dataflow only.",
        signal_gate_passed=signal_gate,
        recommendation=("After upload approval, admit a small pinned Linux/GPU BC+PPO versus BC+PPO+AMP interface/quality pilot; no performance claim from this CPU probe."
                        if signal_gate else "Dataflow works, but do not promote prior efficacy. First review representation/heldout signal or admit interface-only Linux smoke with no advantage claim."),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256(Path(__file__).with_name("amp_v1.py").read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,
        accounting=dict(new_robot_transitions=0, new_simulator_queries=0, discriminator_updates=200,
                        expert_minibatch_rows=200*256, learner_minibatch_rows=200*256,
                        teacher_prior_generation_transitions=teacher.metadata["environment_transitions"],
                        precollected_learner_transitions=sum(collection["arms"][arm]["environment_transitions"] for arm in ("random", "bc")),
                        teacher_prefix_transitions=teacher.metadata["teacher_prefix_transitions"],
                        gpu_seconds=0, dollars_spent=0),
        limitations=["Mac dev-SAPIEN CPU data differs from pinned Linux/GPU reference; neither data nor result is production-comparison eligible.",
                     "Classifier separation is not a policy-learning or sample-efficiency result.",
                     "Random-policy pairs are held-out stress only, not discriminator training negatives.",
                     "Outcome-stratified scores are diagnostics only, with small episode counts.",
                     "Shifted-successor diagnostic changes temporal dependence, not the training objective."])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("signal_gate_passed", "heldout_style_gap", "heldout_balanced_sign_accuracy", "real_minus_shifted_expert_style", "wall_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--learner-datasets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.teacher_dataset, args.learner_datasets, args.output)
