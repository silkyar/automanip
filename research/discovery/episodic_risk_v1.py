"""Finite-episode CVaR score adapter and self-contained pinned PPO payload builder.

Published risk-policy-gradient mechanism, adapted to a clipped PPO surrogate.
No ordinary state baseline is subtracted inside a selected tail. Learned
quantile models are deliberately absent from runtime and agent serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import torch

OFFICIAL_SHA256 = "3fa4818861d428480244aae889bc679b2c8e89b59d6289014003c266a84dbab0"
TASK_FINGERPRINT = "b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"


def episode_returns(rewards, gamma):
    if rewards.ndim != 2 or not 0 < gamma <= 1 or not torch.isfinite(rewards).all():
        raise ValueError("Expected finite time-by-episode rewards and valid gamma")
    discount = gamma**torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)
    return (discount[:, None]*rewards).sum(0)


def score_weights(returns, fraction):
    if returns.ndim != 1 or len(returns) < 2 or not 0 < fraction <= 1 or not torch.isfinite(returns).all():
        raise ValueError("Expected complete ordinary episode returns and valid alpha")
    threshold = torch.quantile(returns.detach(), fraction)
    weights = (returns.detach()-threshold)*(returns.detach() <= threshold)/fraction
    return weights, threshold


def training_diagnostics(returns, success, fraction):
    if success.shape != returns.shape or success.dtype != torch.bool:
        raise ValueError("Training final-success flags must be one boolean per episode")
    weights, threshold = score_weights(returns, fraction)
    strict = returns < threshold
    inclusive = returns <= threshold
    failure = ~success
    failure_count = int(failure.sum())
    overall = float(failure.float().mean())
    tail_rate = float(failure[inclusive].float().mean()) if inclusive.any() else None
    return dict(
        episodes=len(returns), fraction=fraction, return_quantile=float(threshold),
        mean_discounted_return=float(returns.mean()),
        strict_tail_episodes=int(strict.sum()), inclusive_tail_episodes=int(inclusive.sum()),
        training_final_successes=int(success.sum()), training_failures=failure_count,
        inclusive_tail_failures=int((failure & inclusive).sum()),
        strict_tail_failures=int((failure & strict).sum()),
        inclusive_tail_failure_rate=tail_rate,
        failure_capture_fraction=(float((failure & inclusive).sum())/failure_count if failure_count else None),
        failure_enrichment=(tail_rate/overall if overall > 0 and tail_rate is not None else None),
        nonzero_score_episodes=int((weights != 0).sum()),
        score_weight_rms=float(weights.square().mean().sqrt()),
        scope="Ordinary training episodes only; success is diagnostic, never an actor-loss label",
    )


def actor_score_advantages(rewards, dones, next_done, final_infos, gamma, mode):
    if rewards.shape[0] != 50 or dones.shape != rewards.shape or next_done.shape != rewards.shape[1:]:
        raise ValueError("Risk adapter requires complete synchronized 50-step episodes")
    if dones[1:].bool().any() or not next_done.bool().all():
        raise ValueError("Partial or cross-episode rollout rejected")
    if not torch.isfinite(dones).all() or not torch.isfinite(next_done).all():
        raise ValueError("Nonfinite episode masks")
    final_mask = final_infos.get("_final_info")
    if final_mask is None or not final_mask.bool().all():
        raise ValueError("Every final transition must supply its pre-reset final_info")
    success = final_infos["final_info"]["episode"]["success_at_end"]
    returns = episode_returns(rewards, gamma)
    if mode not in ("ordinary", "mean", "cvar"):
        raise ValueError("Unregistered risk mode")
    alpha = .1 if mode == "cvar" else 1.
    # Always log the proposed10% tail, including ordinary and mean controls.
    diagnostics = training_diagnostics(returns, success, .1)
    weights, threshold = score_weights(returns, alpha)
    scale = weights.square().mean().sqrt()
    normalized = weights/scale.clamp_min(1e-8)
    diagnostics.update(actor_mode=mode, actor_alpha=alpha, actor_var_baseline=float(threshold),
                       actor_raw_weight_rms=float(scale), score_rms_normalization=True,
                       actor_score_used=(mode != "ordinary"))
    return normalized[None, :].expand_as(rewards).clone(), diagnostics


def risk_configure(args):
    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset:
        raise ValueError("Only the fixed public StackCube full-horizon interface is supported")
    if args.evaluate or args.track or args.capture_video or args.risk_mode not in ("ordinary", "mean", "cvar"):
        raise ValueError("Training-only payload: no evaluator, tracking or video mode")
    if args.num_envs < 32 or not 1 <= args.risk_max_seconds <= 1320:
        raise ValueError("At least32 episodes and a bounded trainer lease required")
    if args.checkpoint and (args.risk_prefix_transitions <= 0 or args.risk_prefix_trainer_seconds is None):
        raise ValueError("Continuation must declare all prefix transitions/time")
    if args.risk_prefix_transitions < 0 or (args.risk_prefix_trainer_seconds is not None and args.risk_prefix_trainer_seconds < 0):
        raise ValueError("Invalid prefix accounting")
    if args.checkpoint:
        actual = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if actual != args.risk_expected_checkpoint_sha256:
            raise ValueError("Reviewed continuation checkpoint digest mismatch")
    else:
        actual = None
        if args.risk_prefix_transitions or args.risk_prefix_trainer_seconds:
            raise ValueError("Prefix declared without reviewed continuation input")
    now = time.monotonic()
    lease = float(os.environ["AUTOMANIP_TRAINING_DEADLINE"])
    if not math.isfinite(lease) or lease <= now:
        raise ValueError("Explicit trusted absolute deadline missing, invalid or expired")
    output = Path(args.risk_output_dir)
    output.mkdir(parents=True, exist_ok=False)
    configuration = dict(vars(args))
    configuration.update(
        upstream_ppo_sha256="3fa4818861d428480244aae889bc679b2c8e89b59d6289014003c266a84dbab0",
        trainer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_checkpoint_sha256=actual, optimizer_restart=True,
        initial_policy="own reviewed reference continuation" if actual else "random initialization",
        scientific_track="continuation" if actual else "from-scratch",
        no_teacher_demos_pretraining_or_planning=True,
        risk_note="CVaR finite-batch score with VaR baseline; RMS scaling, per-step PPO clipping and repeated epochs are approximations, not exact episodic-CVaR optimization.",
        dollars=None, gpu_seconds=None,
        effective_norm_adv=(args.norm_adv if args.risk_mode == "ordinary" else False),
    )
    (output/"configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    if args.risk_mode != "ordinary":
        args.norm_adv = False  # No minibatch centering/state baseline of tail weights.
    args.save_model = False  # Own numeric actor export and ordinary state_dict below.
    return min(lease, now+args.risk_max_seconds), now


def risk_export(agent, args, label):
    output = Path(args.risk_output_dir)
    linear = [layer for layer in agent.actor_mean if isinstance(layer, torch.nn.Linear)]
    hidden = [layer.out_features for layer in linear[:-1]]
    if len(hidden) != 3 or hidden != [256, 256, 256] or linear[0].in_features != 48 or linear[-1].out_features != 8:
        raise ValueError("Public pinned actor shape changed")
    tensors = {}
    for i, layer in enumerate(linear):
        for name in ("weight", "bias"):
            tensor = getattr(layer, name).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite policy")
            tensors[f"layers.{i}.{name}"] = dict(shape=list(tensor.shape), values=tensor.flatten().tolist())
    raw = json.dumps(tensors, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = output/"actors"/label
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(format="automanip-json-f32-v1", task_fingerprint="b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f",
                    observation_dim=48, action_dim=8, hidden_sizes=hidden, activation="tanh", weights_sha256=digest)
    (directory/"weights.json").write_bytes(raw)
    (directory/"manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    # Standard Agent tensor state_dict only; never passed to the trusted evaluator.
    state = agent.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite ordinary actor/critic checkpoint")
    native = output/"native"
    native.mkdir(exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in state.items()}, native/f"{label}.pt")
    return dict(actor_directory=str(directory), actor_weights_sha256=digest,
                native_state_dict_sha256=hashlib.sha256((native/f"{label}.pt").read_bytes()).hexdigest())


def build_trainer(official):
    if hashlib.sha256(official).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Pinned public PPO source differs")
    source = official.decode()
    helpers = "import hashlib, json, math\nfrom pathlib import Path\n\n"+"\n\n".join(
        inspect.getsource(fn) for fn in (episode_returns, score_weights, training_diagnostics,
                                       actor_score_advantages, risk_configure, risk_export))+"\n\n"
    entry = 'if __name__ == "__main__":\n'
    if source.count(entry) != 1:
        raise ValueError("Public PPO entry anchor changed")
    source = source.replace(entry, helpers+entry)
    args_anchor = "    finite_horizon_gae: bool = False\n"
    source = source.replace(args_anchor, args_anchor+"""
    risk_mode: str = "cvar"
    risk_output_dir: str = "risk-continuation-output"
    risk_max_seconds: int = 120
    risk_prefix_transitions: int = 0
    risk_prefix_trainer_seconds: Optional[float] = None
    risk_expected_checkpoint_sha256: Optional[str] = None
""")
    source = source.replace("    # env setup\n", "    automanip_deadline, automanip_started = risk_configure(args)\n    automanip_optimizer_steps = 0\n    # env setup\n")
    for line in (
        "    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)\n",
        "        eval_envs = FlattenActionSpaceWrapper(eval_envs)\n",
        "    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)\n",
        "    eval_obs, _ = eval_envs.reset(seed=args.seed)\n",
        "    eval_envs.close()\n",
    ):
        if source.count(line) != 1:
            raise ValueError("Public PPO evaluation-removal anchor changed")
        source = source.replace(line, "")
    source = source.replace("        if iteration % args.eval_freq == 1:\n", "        if False:  # Owner-controlled external scoring only.\n")
    source = source.replace("        agent.load_state_dict(torch.load(args.checkpoint))\n", "        agent.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))\n")
    loop = "    for iteration in range(1, args.num_iterations + 1):\n"
    source = source.replace(loop, "    automanip_initial_export = risk_export(agent, args, 'initial')\n"+loop+
                            "        if time.monotonic() >= automanip_deadline - 10:\n            break\n")
    anchor = "            returns = advantages + values\n"
    replacement = anchor+"""        automanip_score, automanip_diagnostic = actor_score_advantages(
            rewards, dones, next_done, infos, args.gamma, args.risk_mode)
        if args.risk_mode != "ordinary":
            advantages = automanip_score
        automanip_diagnostic.update(new_transitions=global_step, training_seed=args.seed)
        with (Path(args.risk_output_dir)/"training-risk.jsonl").open("a") as automanip_log:
            automanip_log.write(json.dumps(automanip_diagnostic)+"\\n")
"""
    if source.count(anchor) != 1:
        raise ValueError("Public PPO actor-advantage anchor changed")
    source = source.replace(anchor, replacement)
    source = source.replace("            for start in range(0, args.batch_size, args.minibatch_size):\n",
                            "            for start in range(0, args.batch_size, args.minibatch_size):\n                if time.monotonic() >= automanip_deadline - 10:\n                    break\n")
    source = source.replace("                optimizer.step()\n", "                optimizer.step()\n                automanip_optimizer_steps += 1\n")
    source = source.replace("        clipfracs = []\n", """        clipfracs = []
        v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = torch.tensor(0., device=device)
""")
    source = source.replace('np.mean(clipfracs), global_step)', '(np.mean(clipfracs) if clipfracs else 0.), global_step)')
    final = """    automanip_final_export = risk_export(agent, args, 'final')
    automanip_elapsed = time.monotonic() - automanip_started
    automanip_report = dict(new_transitions=global_step,
        prefix_transitions=args.risk_prefix_transitions,
        total_prefix_plus_new_transitions=args.risk_prefix_transitions+global_step,
        trainer_seconds=automanip_elapsed, prefix_trainer_seconds=args.risk_prefix_trainer_seconds,
        prefix_plus_new_trainer_seconds=(args.risk_prefix_trainer_seconds or 0)+automanip_elapsed,
        optimizer_steps=automanip_optimizer_steps, optimizer_restart=True,
        vector_step_calls=global_step//args.num_envs,
        completed_training_episodes=global_step//50,
        reset_episodes_including_initial=global_step//50+args.num_envs,
        seed=args.seed, mode=args.risk_mode, initial=automanip_initial_export,
        final=automanip_final_export, dollars=None, gpu_seconds=None,
        note="Training accounting only; no protected scoring or performance claim.")
    (Path(args.risk_output_dir)/'report.json').write_text(json.dumps(automanip_report, indent=2)+"\\n")
"""
    source = source.replace("    envs.close()\n", final+"    envs.close()\n")
    compile(source, "episodic-risk-payload", "exec")
    return source.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_trainer(args.official.read_bytes())
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps(dict(path=str(args.output), sha256=hashlib.sha256(payload).hexdigest())))
