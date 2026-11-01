# Can a better action estimate remove enough gradient noise?

Registered before robot outcomes,2026-10-28. Follow-up to the negative
[replay-moment screen](replay-moment-results-20261028.md), within the user's
16:32:33–20:32:33UTC autonomous investigation/$100 shared Modal cap. No new
Runpod use. This is a diagnostic, not another GPU training sweep.

## Mechanism and contrary evidence

The stable joint critic did not produce grasping. A possible bridge is using
replay-trained action values as policy-gradient control variates rather than
trusting their gradients directly. This is established prior art, not our
invention: [Q-Prop](https://arxiv.org/abs/1611.02247) uses an off-policy critic's
Taylor expansion. However, [Mirage](https://proceedings.mlr.press/v80/tucker18a.html)
found important implementation biases and limited unbiased variance gains.
The [Stein method](https://arxiv.org/abs/1710.11198) also requires care about
sample reuse; its KL-penalty surrogate is not clipped PPO. No such method is
implemented or admitted for robot training here.

For a fixed state and first action, future stochastic actions can leave return
noise that any baseline depending only on the current state/action cannot
remove. By total covariance, for local gradient sample g:

`trace Cov(g|s) = trace Cov_a(E[g|s,a]) + E_a[trace Cov(g|s,a)]`.

Even hypothetical perfect removal of the first term leaves the second. We ask
whether a10× variance gain is already implausible in the measured local tangent.
Variance improvement is not automatically interaction-efficiency improvement.

Two cheap gates passed before a robot probe:

- `gradient_noise_budget_probe.py`: .02843 CPU seconds,1,048,576 synthetic
  reward evaluations, zero optimizer/robot calls. Analytic quadratic example
  with stronger continuation noise limits ideal improvement to1.875× or1.097×;
  same-sample gating of a zero-mean variate gives−.48394 exact gradient bias.
- `continuation_variance_gate.py`: .24131 CPU seconds,2,000 synthetic experiments,
  2,048,000 synthetic returns. It checks independent baseline calibration,
  within-continuation variance and subtraction of within/R from between-group
  sample covariance. It distinguishes cases with approximately1× versus100×
  theoretical headroom. Bootstrap intervals are approximate, not certified.

## Fixed public robot diagnostic

Use only two already-created **from-scratch** actor snapshots: common random
initial actor and joint-moment256k actor. They are fixed because they bracket
the stable-but-unsuccessful recipe; neither is a teacher, new candidate or
competent baseline. No critic is used to choose actions or labels. The experiment
does not fit a policy or use its findings as free training data.

For each actor and each of four fresh declared public initial seeds93274000–3:

1. Reset1024 environments to the same seed. Verify all48 initial observations
   match within1e−6 before any branch; otherwise stop, preserve failure, no retry.
   After the repeated first action, require within-action successor observations
   to match within1e−5; record both residuals. These finite-tolerance checks are
   evidence of replication, not a proof that all hidden simulator state matches.
2. Draw64 independent pre-tanh standard Gaussian first-action noises z. Repeat
   each action across16 continuations. Use the snapshot's exact mean/logstd and
   tanh action transform; keep all behavior parameters fixed.
3. Execute50 steps, with independent Gaussian noise in each continuation from
   step2 onward. Record every actual simulator transition and all rewards.
   Task/reward/controller/physics/horizon remain pinned. No state restoration,
   planning, reset search, protected episode seeds or evaluator service.
4. Compute finite50-step reward returns with gamma.99 primary and.9 sensitivity
   from the same rewards. No learned tail value or Monte Carlo claim about the
   continuing target is made.

This is8192 public diagnostic episodes /409,600 explicitly counted extra
simulator transitions,8 synchronized reset calls,400 vector step calls. These
queries are diagnostic compute, not free samples or part of a from-scratch
learning claim. If incomplete/censored, report only collected data; no new
lease, resampling or replacement policy.

Local score coordinates are z: Fisher-whitened **pre-tanh Gaussian mean**
coordinates at fixed variance and fixed state, not bounded-action coordinates
or the whole neural network's Fisher. The first16 action groups estimate the
trace-optimal state-only scalar baseline
`b=E[||z||²G]/E[||z||²]`. The remaining48 groups measure variance independently.
With16 futures per action, estimate within variance directly and subtract
within/16 from covariance of group means. Preserve negative finite-sample
between estimates without silently clamping them. Percentile intervals resample
independent action groups, preserving their complete continuation clusters;
they are exploratory small-sample uncertainty, not formal guarantees.

## Falsification and action

- If the estimated ideal reduction's approximate95% upper endpoint is below10
  for all4 states under gamma.99, reject a10× local variance promise from
  eliminating current-action noise for that actor/context. Do not generalize
  this into a global neural-policy bound or impossibility of10× learning gains.
- An endpoint above10 means only unresolved/headroom; it does not approve an
  algorithm, prove a critic can realize the gain, or justify a training sweep.
- If both components/returns are nearly zero, the measurement is uninformative
  about useful action distinctions; record this rather than declare low noise.
- The policies' initial-state behavior is a narrow slice. State visitation,
  later decisions, GAE bias, temporal covariance and shared-network Jacobians
  remain outside this diagnostic. Its purpose is to reject an unwarranted
  mechanism early, not establish a benchmark result.

One L40S function only: maximum500 process seconds,550 function seconds plus30
startup, fixed409,600-transition ceiling, no optimizer calls or protected
evaluations. Reserve$2 under shared cap (last reported total$13.50352218).
Admit only after source/policy hashes, CPU gates and independent method review.
Archive raw rewards/noises and source, then verify app stopped/zero tasks.
