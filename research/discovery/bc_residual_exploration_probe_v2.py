"""Offline sampler diagnostic on existing, explicitly nonbenchmark CPU demos."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from research.discovery.amp_v1 import load_demos, read_actor
from research.discovery.bc_residual_exploration_v2 import DEFAULT_STD, clipped_normal_squared_error, lower_noise_counterexample, residual_std


def run(dataset, actor_path, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    started = time.monotonic()
    data = load_demos(dataset, allow_local_cpu_smoke=True)
    actor, manifest = read_actor(actor_path)
    obs, actions = data.behavior_rows()
    validation, labels = data.behavior_rows(validation=True)
    before_rng = torch.get_rng_state().clone()
    std, rms = residual_std(actor, obs, actions)
    with torch.no_grad():
        mean = actor(validation)
        deterministic_mse = (mean.clamp(-1, 1)-labels).square().mean()
        metrics = {}
        for name, scale in (("original", torch.full((8,), DEFAULT_STD)), ("residual", std)):
            mse, clipping = clipped_normal_squared_error(mean, scale[None], labels)
            metrics[name] = dict(std=scale.tolist(),
                                 exact_expected_executed_action_mse_to_demo=float(mse.mean()),
                                 exact_expected_action_clip_fraction=float(clipping.mean()),
                                 per_coordinate_expected_action_mse=mse.mean(0).tolist())
    if not torch.equal(before_rng, torch.get_rng_state()):
        raise AssertionError("Residual estimator changed the rollout RNG stream")
    report = dict(
        purpose="offline action-distribution diagnostic", comparison_eligible=False, formal_benchmark_result=False,
        dataset_sha256=data.dataset_sha256, actor_weights_sha256=manifest["weights_sha256"],
        training_episode_indices=data.train_episode_indices, excluded_validation_episode_indices=data.validation_episode_indices,
        calibration_rows=len(obs), heldout_action_diagnostic_rows=len(validation), rms_residual=rms.tolist(),
        floor=.05, ceiling=DEFAULT_STD, exact_std=std.tolist(),
        heldout_deterministic_action_mse_to_demo=float(deterministic_mse), metrics=metrics,
        lower_noise_counterexample=lower_noise_counterexample(),
        decision="Ready for one paired sampler-initialization test; offline action distance is not an RL gain.",
        unchanged=["deterministic mean", "critic", "reward", "entropy coefficient", "architecture", "state/action interface", "PPO update code"],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mechanism_source_sha256=hashlib.sha256(Path(__file__).with_name("bc_residual_exploration_v2.py").read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,
        accounting=dict(new_robot_transitions=0, new_simulator_queries=0, optimizer_updates=0,
                        additional_bc_forward_rows=len(obs)+len(validation), gpu_seconds=0, dollars_spent=0),
        limitations=["Existing Mac dev-SAPIEN training data is not production benchmark data.",
                     "BC residual RMS is not irreducible/environment uncertainty; clipped labels and regression bias affect it.",
                     "Smaller noise can suppress corrective exploration, as the counterexample demonstrates.",
                     "A random critic and PPO updates may cause other warm-start failures; this test does not identify their causes."])
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("exact_std", "metrics", "lower_noise_counterexample", "wall_seconds")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.actor, args.output)
