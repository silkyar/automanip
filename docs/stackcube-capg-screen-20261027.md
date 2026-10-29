# CAPG learning screen — 2026-10-27

The user authorized returning to learning experiments rather than spending more
time on GPU utilization. This is a two-run development screen, not a broad sweep.

The preserved full-horizon PPO reference has now passed its once-only promotion
rule: three independently trained policies scored 478/512, 483/512 and 484/512,
mean 94.0755%, minimum 93.3594%. Those results establish reference eligibility;
they are not a private final holdout or evidence of a new sample-efficient method.
Confirmation reports remain separate from candidate-side development feedback.

## Question and intervention

Does conditioning PPO likelihoods on the executed clipped action improve the
learning curve compared with scoring the original unbounded Gaussian sample?
The measured late action-component clipping motivates testing gradient variance;
it does not establish a defect. This adapts published
[Clipped Action Policy Gradient](https://proceedings.mlr.press/v80/fujita18a.html),
not a new algorithm. CPU boundary-mass/gradient/variance checks and the restricted
400-transition simulator smoke already passed. Neither demonstrates learning.

Use the two previously reviewed payloads, unchanged:

| Method | Trainer SHA-256 |
|---|---|
| Full-horizon PPO control | `68e09965db1378616f380eb763862149d8c28232075e06741426caba3dfb5064` |
| Full-horizon PPO + CAPG | `8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971` |

Both use random-initialization training seed 1, dedicated L40S GPUs, gamma .9,
entropy coefficient .003, learning rate .0003, eight PPO epochs, 1,024 parallel
environments, 50-step rollouts and 32 minibatches. The Gaussian sampling, latent
Gaussian entropy term, architecture, optimizer, official training reward, task,
physics and observation/action interface remain fixed. Training runs to the
ordinary 50-step horizon rather than resetting immediately after success.

## Budget, artifacts and evaluation

- Two runs only, at most two concurrent trainers, no GPU packing work.
- Maximum 1,320 trainer seconds per run, including Python startup and built-in
  evaluation, with a 20-second graceful-save reserve and an external hard stop.
- Requested ceiling 25M training transitions; the rollout-aligned maximum is
  24,985,600. Log actual steps if the deadline stops a run earlier.
- Initial pair budget $3 including development evaluation and overhead. Price,
  capacity wait, setup, training, conversion and evaluation are accounted for
  separately. No silent restart or fresh time allowance after an interruption.
- Retain logs, TensorBoard events, native checkpoints and reviewed safe actor
  artifacts. Native checkpoints are decoded only by the restricted converter;
  the trusted evaluator accepts the existing safe numeric actor schema.
- The unchanged payload logs KL, clip fractions and optimization time but not
  exact optimizer-step counts. KL can end updates early: eight epochs times
  32 minibatches is an upper bound per rollout, not an observed update count.
  Explicitly report that limitation rather than invent counts. CAPG also changes
  the likelihood ratio used by KL stopping; a learning-curve gain alone cannot
  isolate score-function variance from changed update stopping. A later matched
  update-count ablation would be required for that mechanism claim.
- The trainer cannot access evaluator files, evaluation results or credentials.
- Frozen development evaluation: 128 episodes at preselected training checkpoints
  5.12M, 10.24M, 15.36M and 20.48M, plus the final actual-step checkpoint. These
  already exist under the unchanged official checkpoint schedule. If training
  stops early, report absent later points as censored; do not invent them.
- Built-in eight-environment training evaluations are auxiliary diagnostics.
  They do not establish the protected development success threshold.
- No confirmation queries, private holdout queries or evaluator modifications.

## Screen decision, declared before viewing results

Compare the paired curves on their common measured transition domain. Report
trapezoidal AUC divided by that domain's length, final success and observed
80/85/90% crossing brackets. The first measured threshold crossing does not imply
the policy remained above threshold, and no interpolation is a measured crossing.
Report training elapsed time at checkpoints separately from evaluator runtime.

A candidate warrants additional training seeds if its normalized development AUC
is at least .03 above control without losing more than 3/128 final successes, or
if it first reaches 85% at least one common checkpoint earlier with the same final
success guard. This is a practical screening rule, not a significance test.
No single-seed result establishes a reliable sample-efficiency improvement.
If neither gate passes, stop this branch and record the negative result. Crashes,
invalid artifacts or budget violations are failures, not opportunities to retry
with a fresh budget. Any subsequent paired seeds receive a separate bounded plan.

In parallel, the already delegated research agent may implement a conventional
off-policy control and cheap CPU tests. Its rejected recursive-target toy branch
remains stopped. No additional GPU training is authorized by that implementation
work, and no protected confirmation episodes are passed to the candidate agent.
