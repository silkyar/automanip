"""One frozen CPU fixed-policy LQR offset diagnostic; no robot or actor updates.

Analytic values/gradients are heldout measurements only. D is an explicitly
informative oracle initialization, not a learned centering method or PopArt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from research.discovery.replay_learner import Critic

CONFIG = dict(seeds=[1, 2, 3], steps=1000, batch_size=256, width=64,
              replay_rows=16384, heldout_rows=4096, learning_rate=.001,
              adam_betas=[.5, .999], gamma=.95, tau=.005,
              state_coefficient=.8, action_coefficient=.5, action_cost=.1,
              policy_gain=-.6, dtype="float64", cpu_threads=1,
              dataset_seed=812810, index_seed_base=812900,
              data_draws="existing float32 Uniform[-1,1] then exact float64 cast",
              initial_weights="existing float32 Critic initialization then float64 cast",
              arms={"A": dict(reward_shift=0., coordinate=0., internal_bias_shift=0.),
                    "B": dict(reward_shift=1., coordinate=0., internal_bias_shift=0.),
                    "C": dict(reward_shift=1., coordinate=20., internal_bias_shift=-20.),
                    "D": dict(reward_shift=1., coordinate=20., internal_bias_shift=0.)},
              parity_tolerance=1e-7, cpu_wall_cap_seconds=300)
MEASUREMENT_COST = dict(critic_forward_rows=0, action_gradient_rows=0, analytic_heldout_rows=0)


def digest_tensors(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        x = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def transitions(count, generator):
    # Same RNG stream/distribution as replay_value_probe; all fit tensors double.
    state = (2*torch.rand(count, 1, generator=generator)-1).double()
    action = (2*torch.rand(count, 1, generator=generator)-1).double()
    following = .8*state+.5*action
    reward = -(state.square()+.1*action.square())
    obs, acts = torch.zeros(count, 48, dtype=torch.float64), torch.zeros(count, 8, dtype=torch.float64)
    next_obs, next_acts = torch.zeros_like(obs), torch.zeros_like(acts)
    obs[:, :1], acts[:, :1] = state, action
    next_obs[:, :1], next_acts[:, :1] = following, -.6*following
    return dict(obs=obs, action=acts, next_obs=next_obs, next_action=next_acts,
                reward=reward, state=state, scalar_action=action, following=following)


def fit_update(learner, batch):
    current = learner["critic"](batch["obs"], batch["action"])
    c = learner["coordinate"]
    with torch.no_grad():
        future = learner["target"](batch["next_obs"], batch["next_action"])
        target = batch["reward"]+learner["reward_shift"]+.95*(c+future)-c
    loss = F.mse_loss(current, target)
    learner["optimizer"].zero_grad(set_to_none=True); loss.backward()
    if not torch.isfinite(loss) or any(not p.grad.isfinite().all() for p in learner["critic"].parameters()):
        raise FloatingPointError("Nonfinite fit: preserve, no retry")
    learner["optimizer"].step()
    with torch.no_grad():
        for online, lagged in zip(learner["critic"].parameters(), learner["target"].parameters()):
            lagged.lerp_(online, .005)
    return float(loss.detach())


def parity(left, right, bias_adjustment=0.):
    """Compare left to right after left output-bias coordinate adjustment."""
    maxima = dict(online_parameter=0., target_parameter=0., parameter_gradient=0., adam_state=0.)
    for role, key in (("critic", "online_parameter"), ("target", "target_parameter")):
        for (name, a), (other, b) in zip(left[role].named_parameters(), right[role].named_parameters()):
            assert name == other
            value = a.detach()+bias_adjustment if name == "output.bias" else a.detach()
            maxima[key] = max(maxima[key], float((value-b.detach()).abs().max()))
    for a, b in zip(left["critic"].parameters(), right["critic"].parameters()):
        if a.grad is not None and b.grad is not None:
            maxima["parameter_gradient"] = max(maxima["parameter_gradient"], float((a.grad-b.grad).abs().max()))
        sa, sb = left["optimizer"].state.get(a, {}), right["optimizer"].state.get(b, {})
        assert sa.keys() == sb.keys()
        for key in sa:
            maxima["adam_state"] = max(maxima["adam_state"], float((sa[key]-sb[key]).abs().max()))
    return maxima


def measured_outputs(learner, data):
    critic = learner["critic"]
    before = digest_tensors(critic.state_dict().items())
    with torch.no_grad():
        q = critic(data["obs"], data["action"])+learner["coordinate"]
    action = data["action"].detach().clone().requires_grad_(True)
    predicted = critic(data["obs"], action)
    gradient = torch.autograd.grad(predicted.sum(), action)[0][:, :1].detach()
    MEASUREMENT_COST["critic_forward_rows"] += 2*len(action)
    MEASUREMENT_COST["action_gradient_rows"] += len(action)
    assert digest_tensors(critic.state_dict().items()) == before
    return q, gradient


def measure(learner, data):
    # All oracle quantities are local to this heldout-only measurement function.
    coefficient = 1.036/(1-.95*.25)
    MEASUREMENT_COST["analytic_heldout_rows"] += len(data["obs"])
    truth = data["reward"]-.95*coefficient*data["following"].square()+learner["reward_shift"]/(1-.95)
    gradient_truth = -.2*data["scalar_action"]-.95*coefficient*data["following"]
    q, gradient = measured_outputs(learner, data)
    denominator = gradient.norm()*gradient_truth.norm()
    return dict(q_rmse=float((q-truth).square().mean().sqrt()),
                q_mean_bias=float((q-truth).mean()),
                scalar_action_gradient_rmse=float((gradient-gradient_truth).square().mean().sqrt()),
                scalar_action_gradient_cosine=float((gradient*gradient_truth).sum()/denominator) if denominator > 0 else None,
                represented_q_mean=float(q.mean()), true_q_mean=float(truth.mean()),
                fitted_state_sha256=digest_tensors(learner["critic"].state_dict().items()),
                measurement_preserves_fitted_state=True)


def run(output):
    if output.exists():
        raise FileExistsError("One receipt only; never overwrite or retune")
    started = time.monotonic()
    process_started = time.process_time()
    torch.set_num_threads(1)
    source_paths = [Path(__file__), Path("research/discovery/replay_learner.py"),
                    Path("research/discovery/sac_baseline.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report = dict(state="started", config=CONFIG, source_sha256=hashes,
                  torch_version=torch.__version__, device="cpu", records=[], parity=[],
                  optimizer_calls=0, sampled_fitting_rows=0, robot_transitions=0,
                  actor_updates=0, evaluator_queries=0, teacher_data=0,
                  interpretation="Synthetic fixed-policy diagnostic; D is informative oracle initialization")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    def persist():
        report["cpu_wall_seconds"] = time.monotonic()-started
        report["cpu_process_seconds"] = time.process_time()-process_started
        report["measurement_work"] = MEASUREMENT_COST.copy()
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)
    try:
        generator = torch.Generator().manual_seed(CONFIG["dataset_seed"])
        replay = transitions(16384, generator); heldout = transitions(4096, generator)
        report.update(replay_sha256=digest_tensors(replay.items()), heldout_sha256=digest_tensors(heldout.items()),
                      unique_synthetic_training_transitions=16384, heldout_synthetic_transitions=4096)
        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            initial = Critic(width=64, normalization=None).double()
            learners = {}
            for arm, spec in CONFIG["arms"].items():
                critic = copy.deepcopy(initial)
                with torch.no_grad(): critic.output.bias.add_(spec["internal_bias_shift"])
                learners[arm] = dict(critic=critic, target=copy.deepcopy(critic).eval().requires_grad_(False),
                                     optimizer=torch.optim.Adam(critic.parameters(), lr=.001, betas=(.5, .999)),
                                     reward_shift=spec["reward_shift"], coordinate=spec["coordinate"])
            indices = torch.randint(16384, (1000, 256), generator=torch.Generator().manual_seed(812900+seed))
            report.setdefault("index_sha256", {})[str(seed)] = digest_tensors([("indices", indices)])
            report.setdefault("initial_linear_state_sha256", {})[str(seed)] = digest_tensors(initial.state_dict().items())
            checks = {"C_vs_B": parity(learners["C"], learners["B"], 20.),
                      "D_vs_A_residual": parity(learners["D"], learners["A"])}
            initial_metrics = {arm: measure(value, heldout) for arm, value in learners.items()}
            # Confirm matching represented initialization and oracle20 difference BEFORE fitting.
            qa, _ = measured_outputs(learners["A"], heldout)
            qb, _ = measured_outputs(learners["B"], heldout)
            qc, _ = measured_outputs(learners["C"], heldout)
            qd, _ = measured_outputs(learners["D"], heldout)
            initial_checks = dict(A_equals_B_max_abs=float((qa-qb).abs().max()),
                                  C_equals_B_max_abs=float((qc-qb).abs().max()),
                                  D_minus_A_minus20_max_abs=float((qd-qa-20).abs().max()))
            if max(initial_checks.values()) > 1e-7: raise AssertionError("Invalid preregistered initialization")
            losses = {arm: [] for arm in learners}
            for step in range(1000):
                if time.monotonic()-started > 300: raise TimeoutError("CPU cap; preserve partial, no repeat")
                batch = {k: v[indices[step]] for k, v in replay.items()}
                for arm, value in learners.items():
                    losses[arm].append(fit_update(value, batch))
                    report["optimizer_calls"] += 1; report["sampled_fitting_rows"] += 256
                for name, left, right, shift in (("C_vs_B", "C", "B", 20.), ("D_vs_A_residual", "D", "A", 0.)):
                    for key, value in parity(learners[left], learners[right], shift).items():
                        checks[name][key] = max(checks[name][key], value)
            for arm, value in learners.items():
                report["records"].append(dict(seed=seed, arm=arm, optimizer_calls=1000,
                    sampled_fitting_rows=256000, initial_metrics=initial_metrics[arm],
                    mean_last100_td_mse=sum(losses[arm][-100:])/100, metrics=measure(value, heldout)))
            for name, left, right, shift in (("C_vs_B", "C", "B", 0.), ("D_vs_A_residual", "D", "A", 20.)):
                ql, gl = measured_outputs(learners[left], heldout); qr, gr = measured_outputs(learners[right], heldout)
                checks[name].update(final_represented_q_adjusted_max_abs=float((ql-qr-shift).abs().max()),
                                    final_action_gradient_max_abs=float((gl-gr).abs().max()))
                checks[name]["within_1e_minus7"] = max(v for k, v in checks[name].items() if k != "within_1e_minus7") <= 1e-7
            report["parity"].append(dict(seed=seed, initial_checks=initial_checks, full_fit_maxima=checks))
            persist()
        by_key = {(r["seed"], r["arm"]): r for r in report["records"]}
        ratios = [by_key[s, "B"]["metrics"]["scalar_action_gradient_rmse"]/
                  by_key[s, "A"]["metrics"]["scalar_action_gradient_rmse"] for s in CONFIG["seeds"]]
        report["B_over_A_gradient_error_ratios"] = ratios
        report["primary_verdict"] = ("supports_constant_mode_interference" if all(x > 2 for x in ratios)
            else "weakens_constant_mode_interference" if all(x <= 1.25 for x in ratios)
            else "partial_or_inconclusive")
        report["coordinate_parity_passed"] = all(v["within_1e_minus7"] for r in report["parity"] for v in r["full_fit_maxima"].values())
        report["state"] = "complete"
    except Exception as exc:
        report.update(state="failed_no_retry", error=type(exc).__name__+": "+str(exc))
        raise
    finally:
        report["critic_forward_fitting_rows"] = 2*report["sampled_fitting_rows"]
        report["sources_unchanged"] = hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        persist()
    print(json.dumps(dict(state=report["state"], seconds=report["cpu_wall_seconds"],
                         optimizer_calls=report["optimizer_calls"], ratios=report["B_over_A_gradient_error_ratios"],
                         verdict=report["primary_verdict"], parity=report["coordinate_parity_passed"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
