# PPO implementation and usage

`train.py` is the editable training entry point. It adapts ManiSkill
v3.0.0b21's official state PPO algorithm rather than depending on an external
RL training package. ManiSkill supplies physics, PickCube, and training rewards;
AutoManip owns optimization, artifact export, provenance, and evaluation.

## What is implemented

- Reproducible random initialization and independent actor/critic MLPs with
  three 256-unit tanh hidden layers by default; learned diagonal Gaussian noise.
- Normalized dense PickCube training rewards with the frozen state/controller
  interface (42 observations, seven normalized actions, Panda end-effector deltas).
- GAE, clipped policy loss, value loss, entropy regularization, gradient clipping,
  Adam, advantage normalization, optional learning-rate annealing, and KL stopping.
- Vectorized ManiSkill rollouts, success-triggered partial resets by default,
  time-limit resets, and bootstrapping from pre-reset final observations.
- Environment transition cap and a soft wall-time budget; checks happen between
  simulator steps/minibatches, with CUDA synchronization at completion.
- Safe deterministic actor exports compatible with `scripts/evaluate.py`.
- JSON run records, training progress, source snapshots and checkpoint digests.

## Preview or execute

Configuration preview (no simulator construction or training):

```sh
PYTHONPATH=src python train.py --config configs/baseline.json --dry-run
PYTHONPATH=src python scripts/run_baseline.py
```

After setting up the pinned simulator environment on Linux/NVIDIA:

```sh
PYTHONPATH=src python train.py --config configs/baseline.json --seed 1 --output runs/reference/seed-1
# Or execute three independent seeds sequentially:
PYTHONPATH=src python scripts/run_baseline.py --execute
```

Defaults stop at 600 seconds or 9,984,000 transitions. The latter is the official
10M-step recipe rounded down to complete 512 x 50 rollouts. Time budget includes
simulator/model setup, rollout and optimizer work; it excludes final artifact
serialization and evaluation. This is a cooperative budget, not an external
process watchdog or a guarantee that each run finishes within exactly 600 seconds.
The first L4 batch completed about 5.6–5.9M transitions per ten-minute run;
see [the reference report](runpod-reference-20261025.md). Other GPU types need
their own throughput measurement.

The output directory must not already exist. Each run writes:

- `run.json`: training config/seed, task ID, git commit and diff, source digest,
  software versions, device/backend, elapsed training time, transitions, optimizer
  updates, errors, and artifact digest. Decision remains `unevaluated`.
- `train_source.py`: snapshot of the editable training entry point.
- `progress.jsonl`: losses, KL, training-only episode summaries, compute counters.
- `policy/manifest.json` and `policy/weights.json`: validated actor-only weights.

These artifacts support evaluation, not optimizer resume. There is no arbitrary
pickle loading or automatic candidate promotion. Measured reference performance
is reported separately from the trainer's own records.
A run that runs out of time before its first optimizer step is marked `trained:
false`. Failed runs retain an error record; they do not emit a candidate policy.

## Evaluate separately

```sh
PYTHONPATH=src python scripts/evaluate.py runs/reference/seed-1/policy --episodes 128 --num-envs 8 --backend physx_cuda
```

The trusted evaluator does not import `train.py`, use training rewards as scores,
or load critic/optimizer state. The current inference implementation is a slow
CPU correctness reference; GPU policy inference can be added after parity checks.
Public development seeds are useful for paired comparisons, not a private final
holdout. Confirmation/repeated-seed promotion and holdout service remain later
work. Save an evaluated checkpoint as the reference only after actual training
and independent evaluation.

## Differences from the official baseline

Source: https://github.com/haosulab/ManiSkill/blob/v3.0.0b21/examples/baselines/ppo/ppo.py

The requested `pd_ee_delta_pose` differs from upstream's default
`pd_joint_delta_pos`; results cannot be described as an identical official recipe.
Our GAE supports explicit terminal bootstrapping. By default it matches upstream's
continuing-task convention: bootstrap even after success termination, but stop
GAE recursion at resets. Disable `bootstrap_on_termination` in config for episodic
zero terminal values; truncations always use their final observations.
We use population-standard-deviation advantage normalization to handle singleton
minibatches; there is no finite-horizon GAE variant or clipped value loss in v0.
Evaluation, videos, TensorBoard and W&B are omitted from the training loop.

## Checks actually run

CPU synthetic tests cover known GAE/reset behavior, final-observation selection,
clipped objective, deterministic initialization, gradient updates, budgets, full
mock-training bookkeeping, and numerical parity between exported actor inference
and the trusted evaluator. Real CPU simulation checks a 50-step zero-action
training-wrapper episode including its time-limit autoreset and final observation.
On 2026-10-25, the same 31 tests passed on Runpod Linux, followed by CUDA wrapper
checks, smoke training/evaluation and three ten-minute random-init PPO runs.
Each exported policy was evaluated on 128 paired development episodes and passed
local artifact validation after download. Full scores, limitations and costs are
in [the reference report](runpod-reference-20261025.md).

PickCube still does not require release. The strict table-placement settling rule
is separate; a full custom placement environment is not part of this PPO change.
