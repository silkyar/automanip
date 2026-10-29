"""Bounded CPU algebra gate for the joint/split replay-moment hypothesis.

No robots, fitting, optimizer, oracle training labels or policy improvement.
Textbook BRN and the pinned author's finite-epsilon expression are distinguished.
This is a Torch mathematical transcription, not a Flax execution/parity claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

UPSTREAM_COMMIT = "a318a266679bec3f9d93c44e029860a6da077c89"
PINNED_FILES = {
    "sbx/sac/policies.py": "985a4e76bcaacc055b9be312d6ce813260f72792235e6d7d50fbcebf10b8e3af",
    "sbx/sac/sac.py": "ea6497a8fe49549de757ccc1531b619295fd012188afbb86f03ffe56014af30c",
    "train.py": "1c4ea95b04bd9947fdfd2fc79c97a8dedbed6ca5dd7cd43ded41403f90e79247",
}


def moments(x, split):
    if x.ndim != 1 or len(x) % 2 or len(x) < 4:
        raise ValueError("One scalar feature, two equal groups, at least two rows/group")
    groups = x.chunk(2) if split else [x]
    means, variances = [], []
    for group in groups:
        mean = group.mean()
        # The author's Flax module selects use_fast_variance=True by default.
        variance = (group.square().mean()-mean.square()).clamp_min(0)
        means.append(mean.expand_as(group))
        variances.append(variance.expand_as(group))
    return torch.cat(means), torch.cat(variances)


def normalize(x, *, split, kind="bn", epsilon=.001, running_mean=.25,
              running_variance=2., corrections_clipped=False):
    mean, variance = moments(x, split)
    standard_deviation = (variance+epsilon).sqrt()
    if kind == "bn":
        return (x-mean)/standard_deviation
    running_std = torch.as_tensor(running_variance+epsilon, dtype=x.dtype).sqrt()
    r = (standard_deviation/running_std).detach()
    d = ((mean-running_mean)/running_std).detach()
    if corrections_clipped:
        r, d = r.clamp(1/3, 3), d.clamp(-5, 5)
    if kind == "textbook_brn":
        return (x-mean)/standard_deviation*r+d
    if kind == "pinned_brn":
        # policies.py BatchRenorm: finite-epsilon custom moments, followed by
        # Flax _normalize, which adds epsilon to custom_var once more.
        custom_var = variance/r.square()
        custom_mean = mean-d*variance.sqrt()/r
        return (x-custom_mean)/(custom_var+epsilon).sqrt()
    raise ValueError("Unknown normalization expression")


def run(upstream, output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = json.loads((upstream/"manifest.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise ValueError("CrossQ commit changed")
    for name, digest in PINNED_FILES.items():
        if manifest["files"][name] != digest or hashlib.sha256((upstream/name).read_bytes()).hexdigest() != digest:
            raise ValueError("CrossQ source changed: "+name)

    template = torch.tensor([-3., -1., 1., 3.], dtype=torch.float64)
    shifted = torch.cat([template-2, template+2])
    same_distribution = torch.cat([template, template])
    joint = normalize(shifted, split=False, epsilon=0)
    split = normalize(shifted, split=True, epsilon=0)
    torch.testing.assert_close(split[:4], split[4:], atol=0, rtol=0)
    assert float((joint[:4]-joint[4:]).abs().mean()) > 1
    torch.testing.assert_close(normalize(same_distribution, split=False),
                               normalize(same_distribution, split=True), atol=0, rtol=0)
    alias = dict(current_feature_mean=-2., next_feature_mean=2.,
                 within_half_variance=5., pooled_variance=9.,
                 joint_normalized_half_means=[float(joint[:4].mean()), float(joint[4:].mean())],
                 split_normalized_half_means=[float(split[:4].mean()), float(split[4:].mean())],
                 split_halves_become_identical=True, no_shift_control_identical=True,
                 interpretation="Separate centering erases a between-group shift; this is not a target-error measurement.")

    # One shared EMA must include between-group variance. Averaging two variances
    # or applying EMA twice is a different operator, even at identical weights.
    momentum, old_mean, old_var = .99, .25, 2.
    pooled_mean, pooled_var = shifted.mean(), shifted.square().mean()-shifted.mean().square()
    shared_mean = momentum*old_mean+(1-momentum)*float(pooled_mean)
    shared_var = momentum*old_var+(1-momentum)*float(pooled_var)
    wrong_average_var = momentum*old_var+(1-momentum)*5.
    sequential_mean = momentum*(momentum*old_mean+(1-momentum)*(-2))+(1-momentum)*2
    assert shared_var != wrong_average_var and shared_mean != sequential_mean
    ema = dict(momentum_old_weight=momentum, correct_union_mean=shared_mean,
               correct_union_variance=shared_var, wrong_average_half_variance=wrong_average_var,
               wrong_twice_updated_mean=sequential_mean, updates_per_normalizer=1)

    postwarm = {}
    for epsilon in (0., .001):
        a = normalize(shifted, split=False, kind="textbook_brn", epsilon=epsilon)
        b = normalize(shifted, split=True, kind="textbook_brn", epsilon=epsilon)
        expected = (shifted-.25)/(2.+epsilon)**.5
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
        torch.testing.assert_close(a, expected, atol=1e-12, rtol=0)
        postwarm[str(epsilon)] = dict(joint_split_maxabs=float((a-b).abs().max()),
                                     common_running_normalization_maxabs=float((a-expected).abs().max()))
    # Identical values do NOT imply identical autodiff Jacobians: r and d are
    # explicitly stopped, but the mean/variance derivatives in BN remain live.
    jacobians = {}
    for name, split_flag in (("joint", False), ("split", True)):
        jacobians[name] = torch.autograd.functional.jacobian(
            lambda x: normalize(x, split=split_flag, kind="textbook_brn"), shifted)
    cross_joint = float(jacobians["joint"][:4, 4:].norm())
    cross_split = float(jacobians["split"][:4, 4:].norm())
    assert cross_joint > .1 and cross_split == 0
    derivatives = dict(joint_current_output_wrt_next_half_jacobian_frobenius=cross_joint,
                       split_current_output_wrt_next_half_jacobian_frobenius=cross_split,
                       full_joint_split_jacobian_maxabs=float((jacobians["joint"]-jacobians["split"]).abs().max()),
                       jacobian_evaluations=2, scalar_reverse_mode_backwards=16,
                       interpretation="Stop-gradient corrections allow equal forward values but different training Jacobians.")

    # Check exactly what the finite-epsilon pinned expression supports. These
    # counterexamples are expected negative conditions, never tuned away.
    negative = {}
    for name, x, epsilon in (
        ("ordinary_variance_zero_epsilon", shifted, 0.),
        ("ordinary_variance_pinned_epsilon", shifted, .001),
        ("low_within_group_variance_pinned_epsilon",
         torch.cat([template*.0001-1, template*.0001+1]), .001),
    ):
        a = normalize(x, split=False, kind="pinned_brn", epsilon=epsilon)
        b = normalize(x, split=True, kind="pinned_brn", epsilon=epsilon)
        clipped_a = normalize(x, split=False, kind="pinned_brn", epsilon=epsilon, corrections_clipped=True)
        clipped_b = normalize(x, split=True, kind="pinned_brn", epsilon=epsilon, corrections_clipped=True)
        reference = (x-.25)/(2.+epsilon)**.5
        negative[name] = dict(unclipped_joint_split_maxabs=float((a-b).abs().max()),
                              clipped_joint_split_maxabs=float((clipped_a-clipped_b).abs().max()),
                              unclipped_joint_vs_common_running_maxabs=float((a-reference).abs().max()),
                              unclipped_split_vs_common_running_maxabs=float((b-reference).abs().max()))
    assert negative["ordinary_variance_zero_epsilon"]["unclipped_joint_split_maxabs"] < 1e-12
    assert negative["ordinary_variance_pinned_epsilon"]["unclipped_joint_split_maxabs"] > 1e-8
    assert negative["low_within_group_variance_pinned_epsilon"]["unclipped_joint_split_maxabs"] > .1
    seconds = time.monotonic()-started
    if seconds > 300:
        raise TimeoutError("CPU gate exceeded five minutes")
    report = dict(schema_version=1, state="passed", cpu_seconds=seconds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_commit=UPSTREAM_COMMIT, upstream_files=PINNED_FILES,
        scope="CPU scalar-feature algebra/autodiff; no Flax numerical parity or robot-learning claim",
        shifted_bn_aliasing=alias, shared_ema=ema,
        textbook_unclipped_postwarm_forward=postwarm, stopped_correction_jacobians=derivatives,
        pinned_expression_negative_conditions=negative,
        new_environment_transitions=0, optimizer_calls=0, fitted_rows=0, teacher_rows=0,
        gpu_seconds=0, dollars=0,
        conclusion="Joint moments preserve distribution differences during BN warmup; BRN values/Jacobians/epsilon must be distinguished. No claim of accurate Bellman targets or useful policy gradients.")
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("research/discovery/upstream/crossq"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream, args.output), indent=2, allow_nan=False))
