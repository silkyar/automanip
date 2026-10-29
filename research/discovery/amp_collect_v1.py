"""Collect teacher demonstrations using fresh public TRAINING seeds only.

No evaluator imports, private seeds, environment-state replay or hidden planning.
Every completed episode is retained; success selection happens later in load_demos.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from research.discovery.amp_v1 import DATA_FORMAT, PRODUCTION_PINS, read_actor


def collect_batch(env, actor, seed, progress):
    obs, _ = env.reset(seed=seed)
    progress["explicit_reset_calls"] += 1
    progress["explicit_reset_episodes"] += len(obs)
    records = dict(observations=[obs.detach().cpu().numpy().astype(np.float32)], actions=[], task_rewards=[])
    progress["partial"] = records
    for step in range(50):
        if obs.shape != (len(obs), 48) or not torch.isfinite(obs).all():
            raise ValueError("Teacher collector observation interface mismatch")
        with torch.no_grad():
            action = actor.deterministic(obs)
        if action.shape != (len(obs), 8) or not torch.isfinite(action).all() or action.abs().max() > 1:
            raise ValueError("Invalid bounded teacher action")
        following, reward, terminal, truncated, infos = env.step(action)
        progress["vector_step_calls"] += 1
        progress["environment_transitions"] += len(obs)
        # Collector uses auto_reset=False, so following is the actual successor.
        if terminal.bool().any() or (step < 49 and truncated.bool().any()) or (step == 49 and not truncated.bool().all()):
            raise ValueError("Collector requires full50-step public task episodes")
        records["actions"].append(action.detach().cpu().numpy().astype(np.float32))
        records["task_rewards"].append(reward.detach().cpu().numpy().astype(np.float32))
        records["observations"].append(following.detach().cpu().numpy().astype(np.float32))
        obs = following
    success = infos["episode"]["success_at_end"]
    if success.shape != (len(obs),) or success.dtype != torch.bool:
        raise ValueError("Pinned public wrapper final-success vector missing")
    batch = {key: np.stack(value, axis=1) for key, value in records.items()}
    batch["final_success"] = success.detach().cpu().numpy()
    batch["public_training_seed"] = np.full(len(obs), seed, dtype=np.int64)
    progress["partial"] = None
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-actor", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--teacher-prefix-transitions", type=int, required=True)
    parser.add_argument("--teacher-prefix-trainer-seconds", type=float, required=True)
    parser.add_argument("--teacher-prefix-dollars", type=float)
    parser.add_argument("--max-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.episodes < 4 or args.num_envs < 1 or args.episodes % args.num_envs or args.teacher_prefix_transitions < 1:
        raise ValueError("Invalid generation or teacher-prefix budget")
    if args.teacher_prefix_trainer_seconds < 0 or not 1 <= args.max_seconds <= 600:
        raise ValueError("Invalid prefix time or collector lease")
    started = time.monotonic()
    deadline = min(float(os.environ["AUTOMANIP_TRAINING_DEADLINE"]), started+args.max_seconds)
    if not np.isfinite(deadline) or deadline <= started:
        raise ValueError("Expired absolute collector lease")
    if any(importlib.metadata.version(key) != version for key, version in PRODUCTION_PINS.items()):
        raise ValueError("Collector requires the exact declared Linux/GPU software pins")
    args.output.mkdir(parents=True, exist_ok=False)
    actor, manifest = read_actor(args.teacher_actor, "cuda")
    if manifest["weights_sha256"] != args.expected_teacher_sha256:
        raise ValueError("Collector teacher differs from the reviewed policy")
    actor.eval()
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = ManiSkillVectorEnv(gym.make(
        "StackCube-v1", robot_uids="panda_wristcam", obs_mode="state",
        control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
        sim_backend="physx_cuda", render_mode="rgb_array", num_envs=args.num_envs),
        auto_reset=False, ignore_terminations=True, record_metrics=True)
    progress = dict(explicit_reset_calls=0, explicit_reset_episodes=0, vector_step_calls=0,
                    environment_transitions=0, partial=None)
    batches, status, error = [], "completed", None
    try:
        for index in range(args.episodes//args.num_envs):
            if time.monotonic() >= deadline-15:
                status = "budget-stopped-before-next-complete-batch"
                break
            batches.append(collect_batch(env, actor, args.seed_start+index, progress))
            print("AMP_COLLECT_PROGRESS", progress["environment_transitions"], sum(len(b["final_success"]) for b in batches), flush=True)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
        partial = progress.pop("partial")
        if partial is not None:
            np.savez_compressed(args.output/"incomplete-batch.npz",
                                **{key: np.stack(value, axis=1) for key, value in partial.items() if value})
        if batches:
            data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            np.savez_compressed(args.output/"trajectories.npz", **data)
            digest = hashlib.sha256((args.output/"trajectories.npz").read_bytes()).hexdigest()
            complete, successes = len(data["final_success"]), int(data["final_success"].sum())
        else:
            digest, complete, successes = None, 0, 0
        metadata = dict(
            format=DATA_FORMAT, sha256=digest, status=status, error=error,
            env_id="StackCube-v1", observation_dim=48, action_dim=8, horizon=50,
            control_mode="pd_joint_delta_pos", obs_mode="state", reward_mode="normalized_dense",
            simulation_backend="physx_cuda", source_partition="public-training",
            public_training_seed_start=args.seed_start, num_envs=args.num_envs,
            requested_episodes=args.episodes, completed_episodes=complete,
            successful_episodes=successes, failed_completed_episodes=complete-successes,
            incomplete_transition_rows=progress["environment_transitions"]-complete*50,
            teacher_actor_weights_sha256=manifest["weights_sha256"],
            teacher_prefix_transitions=args.teacher_prefix_transitions,
            teacher_prefix_trainer_seconds=args.teacher_prefix_trainer_seconds,
            teacher_prefix_dollars=args.teacher_prefix_dollars,
            collection_wall_seconds=time.monotonic()-started, collection_dollars=None,
            versions={key: importlib.metadata.version(key) for key in PRODUCTION_PINS},
            collector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            action_labels="deterministic executed clipped teacher actions",
            selection="all complete episodes preserved; later BC/AMP selects final-successful episodes explicitly",
            reset_accounting="Explicit reset calls/episodes below; constructor work included in wall time but not asserted as an extra measured reset.",
            **progress)
        (args.output/"dataset.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(json.dumps(metadata, sort_keys=True))
    if status == "failed":
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
