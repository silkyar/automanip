# Clipped-action likelihood experiment

Status: CPU mathematics and patch checks passed; not yet trained or validated on
a GPU. Existing Runpod experiments continue unchanged. Modal must pass simulator,
renderer, frozen-development parity and trainer-boundary checks before any trial.

## Evidence and hypothesis

The first completed timed-control seed clips 33.0% of sampled action components
in the final training quarter. The full-horizon variant clips 21.6%. These rates
do not establish a defect: a Gaussian latent-action policy followed by clipping
can yield a valid policy-gradient estimator. However, distinct out-of-range
samples all execute the same physical boundary action, potentially adding
unnecessary score-function variance.

The per-dimension audit is a useful counterexample to treating saturation itself
as a failure: full-horizon training clips its gripper component 73.9% of the
time late in training, versus 53.5% for control, while most arm-joint components
clip much less. Appropriate opening/closing can involve a boundary command.
The proposed estimator conditions on those executed boundary events; it does
not penalize them or assume that less clipping always means a better policy.

Fujita and Maeda's [Clipped Action Policy Gradient, ICML 2018](https://proceedings.mlr.press/v80/fujita18a.html)
uses the probability mass of the clipped tail instead of the density at the
particular latent sample. Their result motivates this adaptation; this is not
a new algorithm invented by AutoManip, nor evidence that it improves PPO here.

For each independent action dimension with Gaussian mean μ and standard
deviation σ, use log Φ((-1−μ)/σ) at the lower bound, the normal log density in
the interior, and log Φ((μ−1)/σ) at the upper bound. Stable `log_ndtr` avoids
subtracting a CDF from one in the far tail. The rollout stores this likelihood;
the update recomputes the same executed-action likelihood for PPO ratios.

The candidate retains Gaussian sampling, the existing latent Gaussian entropy
regularizer, initialization, actor/critic architecture, optimizer and all task
settings. Keeping the entropy term fixed isolates the likelihood change; it is
not a claim to have corrected the entropy of the clipped mixture distribution.
PPO's clipped surrogate adds another approximation, so the original paper's
variance result is not a guarantee of better PPO returns or speed.

## Proposed controlled trial

- Reference: full-horizon training, gamma 0.9, entropy 0.003, learning rate
  0.0003, eight PPO epochs, 1,024 environments, 32 minibatches.
- Candidate: identical, with executed-action likelihoods in rollout and update.
- Three paired random-initialization training seeds, 1/2/3.
- At most 1,320 trainer seconds per run; same 20-second graceful-save reserve
  and independent hard timeout; 25M requested-transition ceiling.
- Final checkpoint only; record steps, rollout/update time, action clipping,
  gradient diagnostics and frozen development outcomes.
- Use same-platform controls on Modal. Provider timings are reported separately
  from Runpod; new platform speed is not attributed to the algorithm.
- No confirmation queries, task edits, reward edits, horizon changes, policy
  output post-processing changes or evaluator modifications.

The deterministic action function at fixed weights is exactly the same clipped
MLP. Therefore the current safe actor serialization and frozen evaluator remain
applicable without extensions.

Implementation: `src/automanip/capg.py`. Four CPU tests check Gaussian tail masses,
finite extreme-tail gradients, an analytic bounded-control gradient with reduced
Monte Carlo variance, and the exact scope of the source patch. All changes are
generated from the pinned official PPO file plus common budget/audit additions.
These are mathematical and interface checks, not simulator training evidence.

## Related findings that changed prioritization

The critic-gradient-domination hypothesis is unsupported in the completed first
seed: critic gradients exceed actor gradients on only 1/488 sampled rollout
updates, with median critic/actor norm ratios 0.028 (control) and 0.042 (full
horizon). Splitting gradient clipping is therefore not the first proposed fix.

Full-horizon seed 1 reached 120/128 final stacks in 1,088.77 trainer seconds,
versus 114/128 in 1,298.89 seconds for the control, both at 24,985,600 transitions.
Rollout time decreased from 1,117.06 to 887.47 seconds, while optimization time
increased from 116.66 to 137.15 seconds. The improvement is mostly in rollout
time, consistent with reduced reset overhead, but reset time was not isolated
directly. Remaining training seeds are pending.

At the preselected 15.36M checkpoint, full horizon scored 88.3% versus 60.2% for
control. The first sampled checkpoint above 80% occurred at logged elapsed times
696.28 versus 1,065.75 seconds. These are sampled observations, not measured
first-crossing times. See `runs/stackcube-settling-20261027/seed1-learning-comparison.png`.
