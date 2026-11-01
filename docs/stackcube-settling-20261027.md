# StackCube: fixed-time settling and PPO mechanism investigation

User authorization on 2026-10-27 covers the reference confirmation and settling
experiments, with later instructions requiring no increase in training time,
deeper code-level research, consideration of alternatives to PPO, and no
benchmark cheating.

## Completed reference confirmation

The existing discount-0.9 / entropy-0.003 policies were evaluated once each on
the frozen confirmation split: 465/512, 347/512 and 449/512 successful final
stacks. Pooled mean: **1261/1536 = 82.0964%**, compared with 321/384 = 83.59375%
development success. No training was needed for these evaluations. These are
historical fixed confirmation seeds independent of development, not a private
final holdout. Confirmation is not queried again in this search.

An orchestration pause applied while the user introduced the wall-clock
constraint. The active evaluator could finish normally; no reference was
re-evaluated. Any confirmation process-time measurement containing that pause
is unsuitable for a performance comparison.

## Effective experiment and budgets

The original nine-run plan was explicitly amended before scientific training.
There are now four methods, three random-initialization seeds each: **12 runs**.

| Method | Change from the paired reference |
|---|---|
| Timed control | Discount 0.9, entropy 0.003, original success resets |
| Full-horizon training | Ignore success terminations while training until the normal horizon |
| Entropy decay | Keep 0.003 through 15M transitions, linearly reach zero at 25M |
| Discount/entropy interaction | Discount 0.925 with entropy 0.003 |

Every run is limited to **at most 1,320 seconds (22 minutes) in the trainer
process**, including startup and its built-in evaluations, and at most the
25M requested transition ceiling (24,985,600 complete-rollout transitions).
A graceful deadline reserves 20 seconds for finishing the current iteration,
saving the final checkpoint and flushing logs. An independent process timeout
enforces the outer cap; exceeding it is an execution failure. Candidate
selection uses the final checkpoint, never a retrospectively selected best
development checkpoint. Independent evaluation time is recorded separately.

Actual transitions can differ under this limit. The new control provides the
paired comparison under identical upper budgets and instrumentation. Historical
25M results are context; a run stopping early is labelled transition-unmatched.
The original controller's recorded configuration is preserved, and each runner
records the effective amended configuration in `time-budget-amendment.json`.

The official PPO file is unchanged. Root constructs deterministic trainer
derivatives by adding marked blocks; removing them must recover the entire
official file byte for byte. Additions enforce the deadline and record audit
metrics; the entropy candidate adds its specified schedule. The original task,
reward, evaluator, episode seeds, 50-step horizon and safe policy loader remain
frozen. Separate trainer/converter/evaluator identities and Landlock restrictions
remain active, with the documented shared-kernel/GPU limitations.

## Deeper research: observations versus hypotheses

The current overnight loop was a bounded sweep, not an unrestricted autonomous
code-research loop. The next step is measured mechanism investigation. Each
proposal needs a predicted signature, a narrowly scoped change, and an equal
budget comparison; a plausible story is not a discovered improvement.

1. **Actor/critic optimization coupling.** The pinned PPO clips one norm across
   both networks. Record separate raw actor and critic gradient norms on the
   first optimization minibatch of each rollout and the resulting shared clip
   multiplier. Large critic gradients could suppress actor gradients, but the
   first control samples do **not** support critic domination: around 1.08–1.18M
   transitions, actor norms were 1.38–1.99, critic norms 0.74–0.95, and shared
   multipliers 0.23–0.31. Adam rescales gradients adaptively, so the clip
   multiplier is not itself the parameter-update multiplier. Inspect actual
   update norms in a later diagnostic before changing clipping or optimizers.
2. **Bounded-action exploration.** The trainer samples an unbounded Gaussian,
   stores its latent log probability, and clips executed actions. This is not
   automatically an incorrect policy-gradient estimator. However, latent
   entropy can increase without useful executed-action diversity. Record
   clipping overall and per action dimension; early control samples were about
   10.4–10.7% clipped components. Check late placement/release behavior before
   concluding this is pathological or testing a squashed distribution.
3. **Termination/bootstrap semantics.** The official learner populates final
   values at resets and bootstraps with them. Distinguish success terminations
   from timeouts, and determine the intended episodic versus continuing
   objective before calling this a bug. Full-horizon training is the first
   controlled test of the observed training/evaluation reset difference.
4. **Feature and value scales.** Inspect feature magnitudes, tanh saturation,
   value residuals and explained variance before trying normalization or
   initialization changes. An affine observation normalizer could be folded
   exactly into the actor's first-layer weights at export, retaining the frozen
   evaluator interface; nonlinear preprocessing cannot be silently substituted.

All these are hypotheses or diagnostics. No gradient-clipping, normalization,
initialization, optimizer or algorithm improvement has yet been established.

## Algorithm challenger

PPO is a reference, not a permanent restriction. SAC is the first proposed
alternative because replay may improve sample efficiency in continuous control.
That does not guarantee better wall-clock efficiency with massively parallel
simulation; a pilot must fit the same 22-minute limit and preserve task/reward.

A tanh-hidden SAC actor can represent its squashed deterministic mean in the
existing safe format: two 256-wide tanh layers, the mean layer as an eight-wide
tanh hidden layer, then an eight-dimensional identity output. This fits the
validator's existing three-hidden-layer bound, without changing evaluator code.
An exporter would need exact numerical parity and schema tests. This is a
feasibility design, not an implemented or trained SAC result, and differs from
an upstream actor that uses ReLU. Pin and document the recipe before a pilot.

Sources: [official PPO baseline](../third_party/maniskill/ppo.py),
[SAC paper](https://arxiv.org/abs/1801.01290),
[ManiSkill baselines](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html),
[autoresearch](https://github.com/karpathy/autoresearch).

## Execution and reproducibility

Pod `ltaxmrjm99y9dd`: NVIDIA L4, EU-RO-1, read-back rate $0.59/hour.
Existing volume `zvx3i1sjnx` is mounted at `/workspace`; no new volume was created.
The stopped prior Pod could not resume because its host's GPUs were occupied;
a newly available slot allowed the same GPU class and pinned image to be used.

Pinned image digest:
`sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`.

Persistent results: `/workspace/automanip-settling-20261027`.
Supervisor: `.runtime/settling-20261027-supervisor.json`.
Backup target: `.runtime/settling-20261027-results.tar.gz`.
The supervisor enforces an eight-hour/$6 compute envelope, verifies backup and
stops the Pod. It runs on this laptop: keep it open, powered and online.

All four GPU smoke methods passed. Local checks passed 58 tests with one
Linux-only skip; the amended allowlisted Linux suite passed all 52 tests.
