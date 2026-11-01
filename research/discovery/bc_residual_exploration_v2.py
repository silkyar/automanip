"""Train-demo residual-scaled initialization of existing Gaussian std parameters.

Conventional exploration heuristic, not Bayesian uncertainty or a novelty claim.
No v1 file is edited. Student mean/critic/architecture/reward remain untouched.
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

DEFAULT_STD = math.exp(-.5)
AUTOMANIP_RESIDUAL_PATCH_SHA256 = None  # Filled as literals in self-contained payloads.
AUTOMANIP_RESIDUAL_BASE_SHA256 = None
BASE_PAYLOADS = {
    "assist-bc-ppo-residual-std": "7a4de8deecf0adc15d9f8beaef1bf6b9874ca9eb7b0122efbab5d9b1bc80d427",
    "assist-bc-ppo-amp-residual-std": "c2af05677a3a3652b0bd6b27f482a4946a51cad72b0ce0106e6b2b7b90f457ad",
}


@torch.no_grad()
def residual_std(actor_mean, observations, actions, floor=.05, ceiling=math.exp(-.5), batch_size=1024):
    if observations.ndim != 2 or observations.shape[1] != 48 or actions.shape != (len(observations), 8):
        raise ValueError("Expected existing state48 and bounded action8 demonstration rows")
    if len(observations) < 1 or not 0 < floor <= ceiling or batch_size < 1:
        raise ValueError("Invalid residual exploration contract")
    squared = torch.zeros(8, dtype=torch.float64, device=observations.device)
    for start in range(0, len(observations), batch_size):
        obs, target = observations[start:start+batch_size], actions[start:start+batch_size]
        mean = actor_mean(obs)
        if not torch.isfinite(mean).all() or not torch.isfinite(target).all() or target.abs().max() > 1.000001:
            raise ValueError("Invalid residual calibration rows")
        squared += (mean.double()-target.double()).square().sum(0)
    rms = (squared/len(observations)).sqrt()
    return rms.clamp(floor, ceiling), rms


@torch.no_grad()
def initialize_residual_std(agent, args, logger):
    # load_demos is the already embedded v1 production reader. It still rejects
    # local CPU smoke data by default. Only its existing TRAIN demo split is used.
    started = time.monotonic()
    data = load_demos(args.assist_dataset)
    obs, actions = data.behavior_rows(device=agent.actor_logstd.device)
    before = agent.actor_logstd.detach().exp().clone()
    if not torch.allclose(before, torch.full_like(before, math.exp(-.5)), atol=1e-7, rtol=0):
        raise ValueError("Residual initialization requires the reviewed original logstd=-.5 start")
    standard_deviation, rms = residual_std(agent.actor_mean, obs, actions)
    agent.actor_logstd.copy_(standard_deviation.log().to(agent.actor_logstd.dtype).reshape_as(agent.actor_logstd))
    report = dict(
        mechanism="training-demo residual RMS initializes existing per-coordinate Gaussian std only",
        dataset_sha256=data.dataset_sha256, training_episode_indices=data.train_episode_indices,
        bc_actor_weights_sha256=json.loads((Path(args.assist_bc_actor)/"manifest.json").read_text())["weights_sha256"],
        residual_patch_source_sha256=AUTOMANIP_RESIDUAL_PATCH_SHA256,
        frozen_assisted_base_sha256=AUTOMANIP_RESIDUAL_BASE_SHA256,
        excluded_validation_episode_indices=data.validation_episode_indices,
        residual_forward_rows=len(obs), residual_rms=rms.cpu().tolist(),
        original_std=before.reshape(-1).cpu().tolist(), initial_std=agent.actor_logstd.exp().reshape(-1).cpu().tolist(),
        floor=.05, ceiling=math.exp(-.5), trainable_after_initialization=True,
        new_environment_transitions=0, new_optimizer_updates=0,
        mean_critic_reward_architecture_unchanged=True,
        note="Residual scale is a heuristic using clipped action labels, not calibrated posterior uncertainty.")
    for index in range(8):
        logger.add_scalar(f"audit/assist_init/original_std_dim_{index}", float(before.reshape(-1)[index]), 0)
        logger.add_scalar(f"audit/assist_init/residual_std_dim_{index}", float(agent.actor_logstd.exp().reshape(-1)[index]), 0)
    report["calibration_wall_seconds"] = time.monotonic()-started
    logger.add_scalar("audit/assist_init/calibration_forward_rows", len(obs), 0)
    logger.add_scalar("audit/assist_init/calibration_wall_seconds", report["calibration_wall_seconds"], 0)
    print("AUTOMANIP_EXPLORATION_INITIALIZATION "+json.dumps(report, sort_keys=True), flush=True)


def clipped_normal_squared_error(mean, std, target):
    """Exact E[(clip(N(mean,std),-1,1)-target)^2] and clipping probability.

    Offline action-space diagnostic only, not a robot success estimator.
    """
    mean, std, target = mean.double(), std.double(), target.double()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.isfinite(target).all() or (std <= 0).any():
        raise ValueError("Finite Gaussian parameters and positive std required")
    a, b = (-1-mean)/std, (1-mean)/std
    phi_a, phi_b = torch.exp(-.5*a.square())/math.sqrt(2*math.pi), torch.exp(-.5*b.square())/math.sqrt(2*math.pi)
    lower, upper = torch.special.ndtr(a), torch.special.ndtr(-b)
    interior = (1-lower-upper).clamp(0., 1.)
    first = mean*interior+std*(phi_a-phi_b)
    second = (mean.square()+std.square())*interior+2*mean*std*(phi_a-phi_b)+std.square()*(a*phi_a-b*phi_b)
    mse = second-2*target*first+target.square()*interior+(-1-target).square()*lower+(1-target).square()*upper
    return mse.clamp_min(0), (lower+upper).clamp(0., 1.)


def lower_noise_counterexample():
    # A cloned mean0 needs an exploratory action in(.3,.6) to find a correction.
    # Reduced imitation deviation cannot certify that such a correction is found.
    values = {}
    for name, std in (("original", math.exp(-.5)), ("floor", .05)):
        probability = torch.special.ndtr(torch.tensor(.6/std, dtype=torch.float64))-torch.special.ndtr(torch.tensor(.3/std, dtype=torch.float64))
        values[name] = float(probability)
    return dict(correction_interval=[.3, .6], cloned_mean=0., correction_probability=values,
                limitation="Constructed counterexample only; no assertion StackCube needs this action interval.")


def strip_residual_additions(source):
    return re.sub(r"^[ \t]*# BEGIN BC_RESIDUAL_STD ([\w-]+)\n.*?^[ \t]*# END BC_RESIDUAL_STD \1\n", "", source, flags=re.M | re.S)


def build_payload(base, method):
    if method not in BASE_PAYLOADS or hashlib.sha256(base).hexdigest() != BASE_PAYLOADS[method]:
        raise ValueError("Frozen assisted-v1 payload or residual experiment ID mismatch")
    source = base.decode()
    helpers = ("import math\n"
               f"AUTOMANIP_RESIDUAL_PATCH_SHA256 = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}\n"
               f"AUTOMANIP_RESIDUAL_BASE_SHA256 = {hashlib.sha256(base).hexdigest()!r}\n\n"
               +inspect.getsource(residual_std)+"\n\n"+inspect.getsource(initialize_residual_std)+"\n")
    anchor = "# END DISCOVERY_ASSIST helpers\n"
    source = source.replace(anchor, anchor+"# BEGIN BC_RESIDUAL_STD helpers\n"+helpers+"# END BC_RESIDUAL_STD helpers\n")
    anchor = "    automanip_prior = assist_initialize(agent, args, AUTOMANIP_USE_PRIOR, device)\n"
    if source.count(anchor) != 1:
        raise ValueError("Reviewed BC initializer anchor changed")
    source = source.replace(anchor, anchor+"    # BEGIN BC_RESIDUAL_STD initialize\n    initialize_residual_std(agent, args, logger)\n    # END BC_RESIDUAL_STD initialize\n")
    if strip_residual_additions(source).encode() != base:
        raise ValueError("Unmarked change to assisted-v1 payload")
    compile(source, method+"-payload", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--method", choices=list(BASE_PAYLOADS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.base.read_bytes(), args.method)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
