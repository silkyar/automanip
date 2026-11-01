"""Proposal-only mean LR scaling on the exact frozen residual-std v2 payload.

Known Gaussian trust-region geometry, not a novelty or robotics-gain claim.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time

import torch

BASE_SHA256 = "c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200"


def apply_mean_lr(optimizer, base_lr):
    """Called AFTER every original scheduler reset, never compounding scales."""
    if not math.isfinite(base_lr) or base_lr < 0:
        raise ValueError("Finite nonnegative base LR required")
    for group in optimizer.param_groups:
        group["lr"] = base_lr * group["automanip_mean_lr_multiplier"]


def configure_mean_lr(optimizer, agent, power):
    if power not in (1, 2) or len(optimizer.param_groups) != 1 or optimizer.state:
        raise ValueError("Expected fresh single-group Adam and ratio power 1 or 2")
    std = agent.actor_logstd.detach().double().exp().reshape(-1)
    ceiling = math.exp(-.5)
    if std.numel() != 8 or not torch.isfinite(std).all() or (std < .05-1e-7).any() or (std > ceiling+1e-7).any():
        raise ValueError("Expected exactly eight calibrated residual stds")
    ratio = float((std.log().mean()+.5).exp())
    multiplier = ratio ** power
    mean_parameters = list(agent.actor_mean.parameters())
    mean_ids = {id(parameter) for parameter in mean_parameters}
    original = optimizer.param_groups[0]
    all_ids = {id(parameter) for parameter in original["params"]}
    remaining = [parameter for parameter in original["params"] if id(parameter) not in mean_ids]
    expected_remaining = {id(parameter) for parameter in agent.critic.parameters()} | {id(agent.actor_logstd)}
    if not mean_ids or len(all_ids) != len(original["params"]) or {id(p) for p in remaining} != expected_remaining or not mean_ids <= all_ids:
        raise ValueError("Unexpected actor/critic parameter partition")
    base_lr = original["lr"]
    original["params"] = remaining
    original["automanip_role"] = "critic_and_logstd"
    original["automanip_mean_lr_multiplier"] = 1.
    optimizer.add_param_group(dict(params=mean_parameters, lr=base_lr,
                                   automanip_role="actor_mean", automanip_mean_lr_multiplier=multiplier))
    apply_mean_lr(optimizer, base_lr)
    return dict(ratio=ratio, ratio_power=power, mean_lr_multiplier=multiplier,
                initial_std=std.cpu().tolist(), initial_base_lr=base_lr,
                initial_mean_lr=base_lr*multiplier, initial_critic_lr=base_lr, initial_logstd_lr=base_lr)


def mean_lr_initialize(optimizer, agent, logger):
    started = time.monotonic()
    report = configure_mean_lr(optimizer, agent, AUTOMANIP_MEAN_LR_POWER)
    calibration = AUTOMANIP_RESIDUAL_LR_CALIBRATION
    report.update(
        schema_version=1, mechanism="training-demo residual sigma ratio scales actor_mean Adam LR only",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        patch_source_sha256=AUTOMANIP_MEAN_LR_PATCH_SHA256,
        frozen_residual_base_sha256=AUTOMANIP_MEAN_LR_BASE_SHA256,
        dataset_sha256=calibration["dataset_sha256"], bc_actor_weights_sha256=calibration["bc_actor_weights_sha256"],
        training_episode_indices=calibration["training_episode_indices"],
        excluded_validation_episode_indices=calibration["excluded_validation_episode_indices"],
        residual_forward_rows=calibration["residual_forward_rows"],
        calibration_wall_seconds=calibration["calibration_wall_seconds"],
        additional_bc_forward_rows=0, new_environment_transitions=0, new_optimizer_updates=0,
        residual_patch_source_sha256=calibration["residual_patch_source_sha256"],
        original_std=calibration["original_std"], residual_rms=calibration["residual_rms"],
        actor_mean_critic_weights_reward_schema_unchanged=True,
        note="Assisted track; teacher collection and BC prefix costs remain counted. No gain claimed.")
    report["initial_std_sha256"] = hashlib.sha256(json.dumps(report["initial_std"], separators=(",", ":")).encode()).hexdigest()
    report["group_initialization_wall_seconds"] = time.monotonic()-started
    for key in ("ratio", "mean_lr_multiplier", "initial_base_lr", "initial_mean_lr", "initial_critic_lr", "initial_logstd_lr",
                "additional_bc_forward_rows", "group_initialization_wall_seconds"):
        logger.add_scalar("audit/assist_mean_lr/"+key, report[key], 0)
    print("AUTOMANIP_MEAN_LR_INITIALIZATION "+json.dumps(report, sort_keys=True), flush=True)


def audit_mean_lr(optimizer, logger, global_step, iteration):
    by_role = {group["automanip_role"]: group for group in optimizer.param_groups}
    base, mean = by_role["critic_and_logstd"], by_role["actor_mean"]
    report = dict(schema_version=1, global_step=global_step, iteration=iteration,
                  mean_lr=mean["lr"], critic_lr=base["lr"], logstd_lr=base["lr"],
                  mean_lr_multiplier=mean["automanip_mean_lr_multiplier"])
    for key in ("mean_lr", "critic_lr", "logstd_lr", "mean_lr_multiplier"):
        logger.add_scalar("audit/assist_mean_lr/"+key, report[key], global_step)
    print("AUTOMANIP_MEAN_LR_ACTIVE "+json.dumps(report, sort_keys=True), flush=True)


def strip_additions(source):
    return re.sub(r"^[ \t]*# BEGIN BC_MEAN_LR ([\w-]+)\n.*?^[ \t]*# END BC_MEAN_LR \1\n", "", source, flags=re.M | re.S)


def build_payload(base, power):
    if hashlib.sha256(base).hexdigest() != BASE_SHA256 or power not in (1, 2):
        raise ValueError("Expected exact frozen residual-std v2 base and power 1 or 2")
    source = base.decode()
    helpers = (f"AUTOMANIP_MEAN_LR_POWER = {power}\n"
               f"AUTOMANIP_MEAN_LR_PATCH_SHA256 = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}\n"
               f"AUTOMANIP_MEAN_LR_BASE_SHA256 = {BASE_SHA256!r}\n\n"
               + "\n\n".join(inspect.getsource(f) for f in (apply_mean_lr, configure_mean_lr, mean_lr_initialize, audit_mean_lr))+"\n")
    edits = [
        ("# END BC_RESIDUAL_STD helpers\n", "helpers", helpers, ""),
        ('    print("AUTOMANIP_EXPLORATION_INITIALIZATION "+json.dumps(report, sort_keys=True), flush=True)\n',
         "calibration-binding", '    global AUTOMANIP_RESIDUAL_LR_CALIBRATION\n    AUTOMANIP_RESIDUAL_LR_CALIBRATION = json.loads(json.dumps(report))\n', "before"),
        ("    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)\n",
         "optimizer-groups", "    mean_lr_initialize(optimizer, agent, logger)\n", ""),
        ("        # END PRECISION apply-schedule\n", "constant-schedule", "        apply_mean_lr(optimizer, automanip_lr)\n", ""),
        ('            optimizer.param_groups[0]["lr"] = lrnow\n', "anneal-schedule", "            apply_mean_lr(optimizer, lrnow)\n", ""),
        ("        rollout_time = time.time()\n", "active-audit", "        audit_mean_lr(optimizer, logger, global_step, iteration)\n", "before"),
    ]
    for anchor, name, addition, position in edits:
        if source.count(anchor) != 1:
            raise ValueError("Frozen insertion anchor changed: "+name)
        indent = addition[:len(addition)-len(addition.lstrip())]
        marked = indent+"# BEGIN BC_MEAN_LR "+name+"\n"+addition+indent+"# END BC_MEAN_LR "+name+"\n"
        source = source.replace(anchor, marked+anchor if position == "before" else anchor+marked)
    if strip_additions(source).encode() != base:
        raise ValueError("Unmarked change to frozen residual v2 base")
    compile(source, "bc-mean-lr-proposal", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--power", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.base.read_bytes(), args.power)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
