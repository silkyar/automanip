"""One-update Gaussian PPO actor diagnostic on public CPU BC observations.

Own on-policy Gaussian samples and fixed synthetic advantages only. No simulator,
task return, teacher reward, critic gradient or robotic-learning claim.
All frozen training payloads remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from research.discovery.amp_v1 import load_demos, read_actor
from research.discovery.bc_residual_exploration_v2 import DEFAULT_STD, residual_std


def gaussian_kl_parts(old_mean, old_std, new_mean, new_std):
    """Analytic KL(old||new), in the latent Gaussian used by the PPO KL check."""
    old_mean, old_std, new_mean, new_std = (x.double() for x in (old_mean, old_std, new_mean, new_std))
    mean = ((old_mean-new_mean).square()/(2*new_std.square())).sum(-1).mean()
    variance = (torch.log(new_std/old_std)+old_std.square()/(2*new_std.square())-.5).sum(-1).mean()
    return mean+variance, mean, variance


def one_actor_update(base_actor, obs, probe_obs, initial_std, noise, probe_noise, advantages, mean_lr):
    actor = copy.deepcopy(base_actor)
    logstd = nn.Parameter(initial_std.float().log().reshape(1, 8).clone())
    # Mean LR is the diagnostic intervention; std LR is held at the original.
    optimizer = torch.optim.Adam([dict(params=list(actor.parameters()), lr=mean_lr),
                                  dict(params=[logstd], lr=.0003)], eps=1e-5)
    with torch.no_grad():
        old_mean, old_probe_mean = actor(obs), actor(probe_obs)
        old_std = logstd.exp().clone()
        old = torch.distributions.Normal(old_mean, old_std)
        old_probe = torch.distributions.Normal(old_probe_mean, old_std)
        action = old_mean+old_std*noise
        probe_action = old_probe_mean+old_std*probe_noise
        old_logprob = old.log_prob(action).sum(-1)
        old_probe_logprob = old_probe.log_prob(probe_action).sum(-1)
        before = [parameter.detach().clone() for parameter in actor.parameters()]
    fixed_advantage = (advantages-advantages.mean())/(advantages.std()+1e-8)
    distribution = torch.distributions.Normal(actor(obs), logstd.exp())
    ratio = (distribution.log_prob(action).sum(-1)-old_logprob).exp()
    loss = torch.maximum(-fixed_advantage*ratio, -fixed_advantage*ratio.clamp(.8, 1.2)).mean()-.003*distribution.entropy().sum(-1).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*actor.parameters(), logstd], .5)
    optimizer.step()
    with torch.no_grad():
        new_mean, new_probe_mean, new_std = actor(obs), actor(probe_obs), logstd.exp()
        train_kl = gaussian_kl_parts(old_mean, old_std, new_mean, new_std)
        probe_kl = gaussian_kl_parts(old_probe_mean, old_std, new_probe_mean, new_std)
        new_probe = torch.distributions.Normal(new_probe_mean, new_std)
        logratio = new_probe.log_prob(probe_action).sum(-1)-old_probe_logprob
        sampled_proxy = ((logratio.exp()-1)-logratio).mean()
        parameter_step = sum((parameter-previous).double().square().sum() for parameter, previous in zip(actor.parameters(), before)).sqrt()
    return dict(
        mean_lr=mean_lr, logstd_lr=.0003, optimizer_calls=1,
        mean_parameter_step_l2=float(parameter_step), raw_joint_actor_gradient_norm=float(grad_norm),
        global_clip_multiplier=min(1., .5/(float(grad_norm)+1e-6)),
        train_mean_delta_rms=float((new_mean-old_mean).double().square().mean().sqrt()),
        probe_mean_delta_rms=float((new_probe_mean-old_probe_mean).double().square().mean().sqrt()),
        old_std=old_std.flatten().tolist(), new_std=new_std.flatten().tolist(),
        train_exact_kl=float(train_kl[0]), probe_exact_kl=float(probe_kl[0]),
        probe_mean_kl_component=float(probe_kl[1]), probe_variance_kl_component=float(probe_kl[2]),
        next_minibatch_sampled_kl_proxy=float(sampled_proxy),
        exact_probe_kl_exceeds_point1=bool(probe_kl[0] > .1),
        sampled_next_kl_exceeds_point1=bool(sampled_proxy > .1))


def run(dataset, actor_path, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    data = load_demos(dataset, allow_local_cpu_smoke=True)
    actor, manifest = read_actor(actor_path)
    if len(data.train_episode_indices) < 64:
        raise ValueError("Need64 complete public BC training episodes for disjoint minibatches")
    train_ids, probe_ids = data.train_episode_indices[:32], data.train_episode_indices[32:64]
    observations = data.observations[train_ids, :-1].reshape(-1, 48)
    probe_observations = data.observations[probe_ids, :-1].reshape(-1, 48)
    calibration_obs, calibration_actions = data.behavior_rows()
    sigma, _ = residual_std(actor, calibration_obs, calibration_actions)
    sigma_ratio = float((sigma/DEFAULT_STD).log().mean().exp())
    records = []
    for seed in range(6):
        for name, std, scale in (
            ("default-original-lr", torch.full((8,), DEFAULT_STD), 1.),
            ("residual-original-lr", sigma, 1.),
            ("residual-sigma-ratio-lr", sigma, sigma_ratio),
            ("residual-sigma-ratio-squared-lr", sigma, sigma_ratio**2),
        ):
            if time.monotonic()-started > 240:
                raise TimeoutError("Four-minute CPU diagnostic limit exceeded")
            generator = torch.Generator().manual_seed(71000+seed)
            noise = torch.randn(len(observations), 8, generator=generator)
            probe_noise = torch.randn(len(probe_observations), 8, generator=generator)
            # The same frozen synthetic advantage is used across all four sigma/LR
            # cases. It is a controlled score signal, not a task reward or GAE.
            advantages = noise[:, 0]+.5*noise[:, 1]-.25*noise[:, 2]
            measured = one_actor_update(actor, observations, probe_observations, std, noise, probe_noise,
                                        advantages, .0003*scale)
            records.append(dict(seed=seed, case=name, mean_lr_multiplier=scale, **measured))
    cases = sorted({record["case"] for record in records})
    summary = {case: dict(
        median_exact_probe_kl=statistics.median(r["probe_exact_kl"] for r in records if r["case"] == case),
        median_sampled_next_kl=statistics.median(r["next_minibatch_sampled_kl_proxy"] for r in records if r["case"] == case),
        median_mean_parameter_step=statistics.median(r["mean_parameter_step_l2"] for r in records if r["case"] == case),
        median_probe_mean_delta=statistics.median(r["probe_mean_delta_rms"] for r in records if r["case"] == case),
        exact_kl_overshoots=sum(r["exact_probe_kl_exceeds_point1"] for r in records if r["case"] == case),
        sampled_kl_overshoots=sum(r["sampled_next_kl_exceeds_point1"] for r in records if r["case"] == case)) for case in cases}
    report = dict(
        purpose="one-update Gaussian mean-step / trust-region diagnostic", comparison_eligible=False,
        dataset_sha256=data.dataset_sha256, bc_actor_weights_sha256=manifest["weights_sha256"],
        update_training_episode_indices=train_ids, probe_training_episode_indices=probe_ids,
        excluded_bc_validation_episode_indices=data.validation_episode_indices,
        calibration_training_episode_indices=data.train_episode_indices,
        residual_std=sigma.tolist(), geometric_sigma_ratio=sigma_ratio,
        records=records, summary=summary, wall_seconds=time.monotonic()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accounting=dict(new_robot_transitions=0, simulator_queries=0, task_reward_labels=0,
                        synthetic_advantage_rows=6*len(observations), optimizer_calls=24,
                        mean_parameter_optimizer_calls=24, logstd_parameter_optimizer_calls=24,
                        critic_optimizer_calls=0, actor_forward_rows=len(calibration_obs)+24*5*len(observations),
                        actor_backward_rows=24*len(observations), gpu_seconds=0, dollars_spent=0),
        limitations=["Real public Mac CPU BC observation inputs, but synthetic advantages/actions and no policy-learning experiment.",
                     "Uses the exact clipped PPO actor+entropy objective and Adam settings, but omits critic gradients to isolate actor step geometry.",
                     "Fresh optimizer state for a single update does not reproduce100-rollout production training.",
                     "Analytic KL is for the latent Gaussian checked by the trainer, not the clipped executed-action distribution.",
                     "Mean LR scaling may reduce KL while also suppressing useful learning; no optimal scale or robot benefit is established.",
                     "Both completed BC+PPO robot arms failed; this diagnostic cannot establish the cause or a solution."])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(geometric_sigma_ratio=sigma_ratio, summary=summary, wall_seconds=report["wall_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bc-actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.bc_actor, args.output)
