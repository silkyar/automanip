# AutoManip

A small foundation for autonomous experiments on robot policies. ManiSkill and
PyTorch are the intended simulator/training stack. No Tinker service is needed.

The [experiment ledger](docs/experiment-ledger.md) covers all 72 completed
scientific training runs, including changes, tasks, time, steps, per-seed results
and current work. The 50M StackCube recipe reaches 98.70% development and 95.90%
independent confirmation success across three seeds. Current research targets
at least 10× fewer interactions to 80–90% success against strong matched
baselines, with total compute and assistance accounted for. The
[scientific diagnosis](docs/research-strategy-diagnosis-20261028.md) led to one
approved mechanism probe, followed by a user-authorized autonomous investigation.
The [37-paper literature map](docs/literature/survey-20261028.md) covers world
models, replay, credit assignment, exploration, representations and plasticity.
The approved [first-update probe](docs/first-update-probe-results-20261028.md)
is complete: large actor steps damaged the assisted policy, while a useful
small-step direction remains unproven. The subsequent
[replay-moment screen](docs/replay-moment-results-20261028.md) completed three
fresh runs after CPU mechanism checks: all14 early development scores were0/128.
Its promotion gate failed; diagnostics distinguish numerical stabilization from
useful learning. All associated cloud apps are stopped.
The subsequent [two-factor sufficiency investigation](docs/replay-regression-results-20261028.md)
tested one setting at a time: discount restoration failed; lower initial entropy
weight recovered grasping (121/128 at256k) but no lifts/stacks. The primary gate
was partial, so no stacking-efficiency improvement is claimed. Autonomous
scientific work continues under the user's explicit until-stop authorization.

A separate [minimal StackCube publication repo](publication/stackcube-ppo/README.md)
contains the unchanged official PPO, a three-setting improvement diff and the
historical 32.23% → 94.08% three-seed confirmation comparison, with a
[25M progress graph](publication/stackcube-ppo/assets/stackcube-progress-25m.png),
[22-minute checkpoint availability graph](publication/stackcube-ppo/assets/stackcube-observed-by-22min.png)
and [successful final25M-policy video](publication/stackcube-ppo/assets/stackcube-full-horizon-25m-public-success.mp4).

[Runpod preparation](deploy/runpod/README.md) includes a digest-pinned container
recipe, hashed Linux dependency resolution, suggested Pod settings, and GPU smoke
and training commands. The custom image build remains unverified. A Runpod L4
batch using an official image plus the pinned bootstrap completed three
ten-minute training runs: 97.66%, 99.22% and 99.22% stock PickCube development
success. [Results and resource state](docs/runpod-reference-20261025.md).

The separate [StackCube baseline](docs/stackcube-baseline.md) uses the unchanged
official ManiSkill 25M-step PPO example, three training seeds, and frozen
development and confirmation evaluation. Its task, policies and output paths are
separate from PickCube. These original baseline results remain preserved.
[Completed StackCube results and diagnostic videos](docs/stackcube-results-20261026.md):
22.5%, 47.7% and 26.6% confirmation success, averaging 32.2% across three seeds.

## Run the CPU checks

From this directory, with Python 3.11 or newer:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The core tests require no third-party packages; PPO tests run when PyTorch is
installed. All 44 tests passed locally. The GPU training upload passed 38 tests;
cloud-supervisor and later diagnostic-selection checks were run locally. macOS CPU smoke
simulation and Linux CUDA smoke/training/evaluation have passed. The hashed
Linux lock is `requirements/linux-cu124.txt`; optional versions in
`pyproject.toml` alone are not a complete environment lock.

## Code

- `src/automanip/task_spec.py`: frozen task fields, fingerprint, backend constraints.
- `src/automanip/policy_io.py`: bounded data-only policy validation.
- `src/automanip/metrics.py`: stock Panda PickCube rule, separate placement settling rule,
  and Wilson confidence intervals.
- `src/automanip/evaluator.py`: policy/action/result contracts and synchronous vector batch rollouts.
- `tests/test_core.py`: isolated temporary artifacts and CPU unit tests.
- `docs/upstream.md`: verified APIs, reference differences, and open verification work.

## Artifact interface

An artifact directory has fixed names `manifest.json` and `weights.json`.
The manifest has exactly these fields:

```json
{
  "format": "automanip-json-f32-v1",
  "task_fingerprint": "<trusted task fingerprint>",
  "observation_dim": 42,
  "action_dim": 7,
  "hidden_sizes": [256, 256, 256],
  "activation": "tanh",
  "weights_sha256": "<SHA-256 of exact weights.json bytes>"
}
```

**42 was verified** in the macOS CPU smoke test and is frozen in `TaskSpec`.
The evaluator checks the installed environment against that interface. It supplies
`PolicySchema`; candidates cannot choose its dimensions.

Weights map `layers.N.weight` and `layers.N.bias` to `{"shape": [...], "values": [...]}`.
Weights are row-major `[output, input]`; biases are `[output]`. Hidden layers use
tanh, the final layer is linear. Trusted inference must clip means to normalized
action bounds before `validate_actions`; nonfinite outputs must fail the episode.
Only deterministic actor parameters are accepted, not critic/optimizer state.

The loader checks exact fields and tensor names, dimensions, finite float32 values,
byte/parameter limits, and digest. It never executes candidate code. JSON was chosen
for dependency-free CPU validation; safetensors is the intended efficient replacement
before full training. Digests identify bytes, not trustworthy provenance.

## Boundaries and next milestones

This is a local prototype, not an adversarial security boundary. A research agent
with unrestricted shell access could modify these files. Unattended runs require
separate trainer/evaluator identities, mounts, and processes or VMs, with private
seeds inaccessible to the trainer.

PPO training from random initialization, the Linux dependency lock and GPU
development evaluation are now exercised; see the dated reference report.
Next: independent confirmation seeds, learning-efficiency experiments and a
separately versioned placement task. The custom placement checker
is a tested rule only: an evaluator-owned table-placement task and contact-state
adapter still need implementation. The autonomous research loop and private
holdout evaluation remain later milestones.

## Simulator integration

```sh
PYTHONPATH=src python3 scripts/smoke_simulator.py
PYTHONPATH=src python3 scripts/evaluate.py checkpoints/candidate --episodes 8
# On a configured NVIDIA machine:
PYTHONPATH=src python3 scripts/evaluate.py checkpoints/candidate --episodes 128 --num-envs 8 --backend physx_cuda
```

The adapter checks the installed ManiSkill version and action space, measures
state dimensions from the evaluator-owned environment, and reads cube/goal/joint
state directly for scoring. The smoke test compares seeded reset observations.
The reference Python MLP implementation accepts validated tensors only; it is
intended for correctness checks, not training throughput.

Evaluation uses synchronized full-length episodes without autoreset. Success
termination signals are retained but do not reset an environment; the evaluator
ends each episode at 50 steps and reports both ever-success and final success.
Unexpected early truncations, invalid actions, and caught exceptions fail the
batch. Normal horizon completion is distinct from an execution timeout. A hung
native simulator requires a separate supervisor; this MVP has no hard wall-time
watchdog. The CLI uses public development seeds only and is not a holdout service.
Reports deliberately omit seeds and exception messages.

Local status: 44 CPU tests pass, including mock vector rollouts, failure paths,
and known-network inference. The macOS simulator initializes PickCube, steps it,
and reproduces seeded reset observations. The separate Runpod reference batch
verified CUDA physics, training and development evaluation.
`requirements/macos-py311.txt` records the resolved macOS environment; the Linux
environment is pinned separately in `requirements/linux-cu124.txt`.

## Working macOS setup

The project now has a local Python 3.11 environment in `.venv311`, installed using
uv's standalone Python download. Run the simulator with the project-local Vulkan
runtime (no shell profile changes):

```sh
sh scripts/macos_python.sh scripts/smoke_simulator.py
sh scripts/macos_python.sh scripts/evaluate.py checkpoints/candidate --episodes 8
```

Resolved package versions are in `requirements/macos-py311.txt`. They include
ManiSkill 3.0.0b21, PyTorch 2.6.0, NumPy 1.26.4, OpenCV 4.11.0.86, Pinocchio
(`pin`) 2.7.0, and the official SAPIEN 3.0.4 development macOS wheel. The older
SAPIEN wheel linked in ManiSkill's release guide is no longer downloadable.
The replacement works for this smoke test; broader API compatibility is unproven.
The nightly release asset may be replaced upstream; this snapshot is versioned
but is not an archival artifact store or a hash-verified lock.

The Vulkan runtime is MoltenVK v1.4.2, downloaded from the official KhronosGroup
release into `.runtime`. `scripts/macos_python.sh` selects its ICD for the current
command only. Runtime files, uv caches, and virtual environments are Git-ignored.

To recreate the Python dependencies in an existing Python 3.11 environment:

```sh
python -m pip install -r requirements/macos-py311.txt
python -m pip install --no-deps -e .
```

For a fresh runtime, download `MoltenVK-macos.tar` from
https://github.com/KhronosGroup/MoltenVK/releases/tag/v1.4.2 and extract
`MoltenVK/MoltenVK/dynamic` under `.runtime`. CPU physics runs locally; Vulkan
initialization uses the Apple GPU. This does not enable CUDA parallel simulation.

## PPO training code

The trainer is implemented in `train.py`, adapted from ManiSkill's official PPO
baseline. It includes random initialization, GAE, clipped optimization, bounded
training time/steps, run provenance and safe policy export. Three random-init
GPU reference runs are saved in `checkpoints/reference-20261025`. Full commands and algorithm differences are in
[docs/training.md](docs/training.md).

```sh
# Preview the GPU recipe without starting training:
PYTHONPATH=src .venv311/bin/python train.py --config configs/baseline.json --dry-run
# Preview three independently seeded GPU runs:
PYTHONPATH=src .venv311/bin/python scripts/run_baseline.py
# Synthetic PPO tests plus core tests:
PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -v
```
