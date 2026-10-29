# Standalone conventional SAC control — CPU gate and smoke handoff

Status: implementation and CPU gates complete; **no simulator/GPU training run**.
The trusted owner has declared full-horizon PPO the eligible reference. This
candidate agent did not read protected promotion episodes or diagnostics.
The rejected recursive-target branch remains stopped.

## Choice and source audit

Choose SAC because ManiSkill supplies an official state-based off-policy baseline
with the exact vector-environment API in the pinned release. This gives a more
defensible initial conventional control than assembling an unrelated TD3 trainer.
This is not evidence that SAC outperforms TD3 or the eligible PPO reference on
StackCube. The upstream README only supplies lightly tuned PushCube/PickCube
examples, using end-effector delta-position control; it supplies no comparable
StackCube result for our fixed joint controller.

Sources inspected in full:

- [Official pinned SAC source](https://github.com/haosulab/ManiSkill/blob/6226567ce326625fa6eacf9aeb3f4008e4339f7e/examples/baselines/sac/sac.py).
  Tag `v3.0.0b21` resolves to commit
  `6226567ce326625fa6eacf9aeb3f4008e4339f7e`. Exact file SHA256:
  `27e5d7b90a431ed9af1a6a05e593d6404a67a8cce96f8ae066ddf95f0ab6c51a`.
- [Pinned README](https://github.com/haosulab/ManiSkill/blob/6226567ce326625fa6eacf9aeb3f4008e4339f7e/examples/baselines/sac/README.md),
  and the installed `mani_skill==3.0.0b21` public vector-wrapper source.
- [SAC primary paper](https://proceedings.mlr.press/v80/haarnoja18b.html).

Exact source, README, upstream license and provenance are preserved under
`research/discovery/upstream/`. The candidate trainer neither imports nor executes
the vendored baseline. Its implementation is `research/discovery/sac_baseline.py`.

Important pinned wrapper fact: auto-reset calls `self.reset(options=...)`. Thus
instrumenting this wrapper instance's `reset` catches automatic resets in this
release. Final observations are full-vector tensors with `_final_observation`
boolean masks. The trainer cross-checks measured reset episodes against the sum
of terminal/truncated masks plus the initial reset batch, failing on disagreement.

## Recipe and explicit adaptations

The trainer implements squashed-Gaussian reparameterized SAC, twin Q critics,
minimum target Q minus entropy, Polyak target averaging every update, and learned
entropy temperature with target entropy `-action_dim`. It preserves the pinned
alpha objective and initial alpha 1. No demonstrations, checkpoints, teacher,
planning lookahead, reward changes, or extra observations are accepted.

| Setting | Candidate default | Relationship to source |
|---|---|---|
| Robot task | StackCube-v1, panda_wristcam | Public fixed task/interface |
| Observation/action | state48 / bounded action8 | Fixed comparable track |
| Controller/horizon | pd_joint_delta_pos / 50 | Fixed comparable track |
| Reward | normalized_dense | Official environment reward, no shaping |
| Success resets | disabled | Upstream SAC default; eligible PPO behavior |
| Actor | two 256-wide tanh hidden layers, Gaussian mean/std heads, tanh actions | **Adapted** from upstream three 256-wide ReLU layers to fit current safe actor schema |
| Critics | separate three-256-wide ReLU MLPs | Same shape/activation as upstream |
| Gamma | .9 | **Adapted** from .8 to eligible PPO anchor; not claimed globally optimal SAC choice |
| Tau / actor,Q,alpha LR | .01 / .0003 | Upstream defaults |
| Replay / batch | 500,000 / 1,024 | Replay from README example; batch from source |
| Environments | 32 | README example |
| Warmup | 4,000 transitions | Source default, uniform random actions |
| Collection block / UTD | 64 transitions / .5 optimizer rounds per transition | Source defaults; 32 updates per full block |
| Budget | 500,000 transitions, at most 1,320 trainer seconds | Bounded initial control, not a 25M run |

Additional implementation differences: stable analytic tanh-Jacobian log density
instead of subtracting nearly equal floats with an epsilon; freeze critic
parameters during actor backpropagation; bounded uniform transition replay
without unused logprob/value arrays; numeric safe actor artifacts instead of
pickle; explicit accounting, deadline, and seed/version metadata. The critic
freeze preserves gradients with respect to actions and avoids accumulating
unused Q parameter gradients. The stable Jacobian computes the exact squashed
Gaussian density; near extreme saturation this differs from the upstream epsilon
approximation and is declared rather than treated as identical code.

This is an **adapted conventional SAC baseline**, not the unmodified official
recipe and not yet a competent *measured* StackCube control. In particular, the
actor capacity/activation and control-mode changes need robot validation. Do not
call a failure of this one recipe a failure of off-policy RL in general.

## Replay, timing and deadline contract

One-step replay stores `(obs, actual_next_obs, action, reward, true_terminal)`.
At a truncation, `actual_next_obs` is the final pre-reset observation and the
Bellman target bootstraps. A true terminal stops bootstrap. With full-horizon
training, the pinned wrapper ignores success terminations, so episode limits
normally cause truncations. Missing/mismatched final-observation masks at reset
are errors; a reset observation is never silently substituted. Uniform sampling
does not join trajectories or construct cross-episode sequences.

Each run records source/upstream/git hashes, pinned versions, seed, random
initialization, public interface, actual transitions, vector-step calls, reset
calls and reset episodes, completed episodes, joint critic optimizer steps, actor
steps, alpha steps, replay minibatch rows, alpha/losses and checkpoint hashes.
Each joint critic optimizer step updates **two** Q networks. Minibatch rows count
replay samples drawn, not every repeated neural forward/backward presentation.

Wall times cover inference, replay, joint update work, reset work, export, and
environment step work **including** auto-resets. Reset time must not be added to
inclusive step time again. CUDA timing synchronizes work at measurement bounds;
these are accounting hooks, not a throughput optimization or speed claim.
Provider costs/hardware remain for the trusted launcher to append; absent costs
are null, never invented as zero for a GPU run. This CPU phase spent zero.

The trainer uses the minimum of its local budget minus a 20-second save reserve
and the explicitly supplied absolute monotonic
`AUTOMANIP_TRAINING_DEADLINE`. It rejects an already expired lease before simulator
creation. SIGTERM requests a final safe actor export. A launcher **must impose an
independent hard stop** because Python cannot bound a hung simulator/kernel or
filesystem call. Never give a queued process a fresh absolute deadline on start.
The only environment variable read by this trainer is that explicit deadline.

No evaluation environment is created. External trusted evaluation receives only
numeric actor files. Candidate code is not imported into that process.

## CPU evidence

`research/discovery/sac_cpu_gate.py` runs a contextual quadratic task with ordinary
scalar reward, three observations/two actions, and terminal one-transition
episodes. The actor only receives SAC gradients through learned Q; it never gets
optimal-action labels or a behavior-cloning loss. Analytic actions are used solely
for the held-out toy diagnostic. This is a basic actor/critic implementation gate,
not a test of contact dynamics, reset behavior in the actual simulator, or robotic
sample efficiency.

After 1,200 equal SAC rounds on 4,096 ordinary random-action samples per seed:

| Toy seed | Initial held-out action MSE | Final MSE |
|---|---:|---:|
| 0 | .243910 | .046224 |
| 1 | .188288 | .050387 |
| 2 | .196774 | .051377 |
| 3 | .215393 | .047272 |
| 4 | .173160 | .051226 |
| 5 | .186404 | .049695 |

All six improved. A preliminary seed-7 debug run, before the six-seed gate, also
improved .223433→.048278 in 1,200 rounds; it was not used to choose a recipe change.
The six-seed run used 12.61 wall / 24.37 process CPU seconds, 24,576 toy
transitions, 7,200 joint-Q/actor/alpha optimizer rounds each and 1,843,200 sampled
minibatch rows. No GPU, robot transitions, simulator queries, teacher/demonstration
data or paid APIs. Full curves and source digests are in
`research/discovery/sac-cpu-gate-v1/report.json`.

Sixteen discovery tests pass, including final-observation/truncation semantics,
true-terminal target masks, replay wraparound pairing, Polyak updates, extreme
squashed-Gaussian gradient finiteness, bounded budgets and existing counterexample
tests, plus a sanitized runtime with no git executable. Tests do not import the
simulator. A missing repository/git binary records `git_sha=null` without failure
or stderr; the trusted launcher supplies its own revision externally.

The exact learner source used for the CPU result is preserved as
`sac-cpu-gate-v1/learner-source.py` and matches that report's source digest. The
final trainer subsequently received only runtime fixes: absent-git handling,
starting its local lease before metadata work, and retaining uniform warmup until
the first completed learning block (matching the upstream block schedule).
No SAC update/export mathematics changed; the focused tests pass on the final
trainer. Existing fixture weights and trusted parity receipt remain applicable.

## Safe export evidence

Deterministic action is `tanh(mean(obs))`, not a noisy sample or mean of sampled
actions. The safe MLP has hidden sizes `[256,256,8]`: the third tanh applies to the
SAC mean layer, then an identity 8×8 linear layer preserves its output. Logstd and
critics are not exported. JSON keys, shapes, finite float32 values, exact SHA256
and manifest fields follow the public contract supplied by the trusted owner.

The separate **random, untrained robot-schema fixture** is under
`research/discovery/sac-cpu-gate-v1/robot-schema-fixture/`; it is not a robot-trained
candidate. `fixture-parity.json` supplies 130 state48 vectors and expected actions,
including saturated ±1,000 inputs. Candidate-side public numerical inference
matches Torch within max absolute error **4.47e-7**.

The trusted owner separately reports that the unchanged current `load_policy`
accepted the fixture and trusted NumPy inference matched every action within
**3.8048654e-7**. This agent did not inspect the trusted loader or receipt. This is
serialization validation only, not adversarial isolation or policy performance.

## Smallest root-reviewable robot smoke — not launched

Use the already verified Linux lock and existing trainer isolation. Only stage
the candidate module/files and fixed public assets; no evaluator files, protected
seeds, credentials or private outputs in the worker mount. No deploy integration
changes or resource writes were performed here.

```sh
python -m research.discovery.sac_baseline \
  --seed 1 --device cuda --num-envs 8 \
  --total-transitions 800 --buffer-size 512 --batch-size 32 \
  --learning-starts 64 --training-freq 8 --utd 0.125 \
  --gamma 0.9 --max-seconds 120 --checkpoint-every 400 \
  --output /workspace/output/sac-robot-smoke
```

The trusted wrapper supplies a fixed absolute deadline and an independent hard
stop no later than 120 seconds from the allotted trainer lease. This tiny smoke
changes update frequency/batch only to cheaply exercise code; it is not the
baseline performance recipe. Maximum charged compute before startup costs is
120 GPU seconds; actual credits/dollars require the trusted owner's current rate.
No spending is authorized by this document.

If completed normally, expect exactly 800 transitions, 100 vector step calls,
16 completed episodes (two 50-step boundaries per eight environments), 24 reset
episodes including eight initial resets, three reset calls, replay size 512,
**93 joint-Q, 93 actor and 93 alpha optimizer steps**, and 2,976 replay minibatch
rows. Expected artifacts: step-0, step-400, step-800 and final actor directories,
plus `run.json`, `training.jsonl`, `report.json`. The duplicated step-800/final actor
is deliberate; hashes should agree. The default 500k recipe only exports initial
and final actors because default checkpoint interval is 512k; no early curve is
implied. Explicit cadence is required for subsequent performance screens.

Smoke acceptance: correct actual state/action shapes and bounds; both reset
boundaries retain finite final observations; reset counters agree; finite SAC
losses and weights; expected completed update counts; all actor files pass the
trusted artifact loader; candidate and trusted action inference agree at saved
weights; no protected code/data visible to the candidate. A trusted development
rollout may be used as an interface/crash check, with no policy quality criterion
for an 800-transition actor. Record its evaluation transitions separately.

If smoke passes, root can propose a single bounded 500k-transition conventional
SAC control with explicit intermediate exports and the eligible PPO reference.
That performance screen needs its own approved time/spend/evaluation plan. Do not
launch it automatically or interpret this CPU gate as sample-efficiency evidence.

## Reproduction

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_*.py' -v
.venv311/bin/python -m research.discovery.sac_cpu_gate --output /tmp/automanip-sac-cpu-new-run
```

The gate intentionally refuses to overwrite an existing run directory. Earlier
negative discovery results and journals remain intact. All implementation changes
are limited to candidate-owned discovery paths and discovery tests.
