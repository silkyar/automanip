# First GPU reference batch — 2026-10-25

Three independent random-initialization PPO policies were trained and evaluated
successfully on one NVIDIA L4 in Runpod Secure Cloud. All 31 CPU tests passed on
the Linux installation before training. The GPU wrapper check, 800-transition
smoke training, and eight-episode smoke evaluation passed before the full batch.

## Measured results

Each policy used the same 128 public development episode seeds and a 50-step
horizon. Scores below are stock ManiSkill PickCube success at episode end.
The cube may still be grasped; this is not a release or strict-placement metric.

| Training seed | Training seconds | Transitions | Optimizer updates | Successes | Rate | 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 600.043 | 5,853,696 | 15,060 | 125/128 | 97.66% | 93.34–99.20% |
| 2 | 600.020 | 5,657,600 | 15,939 | 127/128 | 99.22% | 95.71–99.86% |
| 3 | 600.043 | 5,621,760 | 15,090 | 127/128 | 99.22% | 95.71–99.86% |

Mean success across training seeds: **98.70%**. Total training time: 1,800.105
seconds; total environment transitions: 17,133,056. All 384 episode rollouts
completed their horizons without failed episodes. These are 128 paired episode
seeds repeated across three policies, not 384 distinct environment seeds.
Success-ever counts were 126, 127 and 127, respectively.

All runs stopped on the time budget before the 9,984,000-transition cap. This
measures the requested ten-minute recipe on this L4, not the complete official
10M-step recipe. Our end-effector controller also differs from upstream's default
joint-delta controller. There was no confirmation-seed or private holdout test,
and no autonomous candidate promotion.

## Training and policy interface

ManiSkill `PickCube-v1`, Panda, `state` observations, `pd_ee_delta_pose`, 512
training environments, 50 rollout steps, independent actor and critic networks
with three 256-unit tanh layers. PPO uses the pinned baseline configuration.

The actor receives 42 privileged state values: robot joint positions/velocities,
TCP pose, cube pose, target position, relative positions and a grasp flag. It
emits seven normalized end-effector/gripper actions. No camera input or physical
robot was used. The normalized dense training reward comes from ManiSkill.
Evaluation scores are recomputed from evaluator-owned simulator state.

Runtime: Python 3.11.10, ManiSkill 3.0.0b21, PyTorch 2.6.0+cu124, SAPIEN 3.0.0b1,
NumPy 1.26.4 and Gymnasium 0.29.1. The 98-package hashed Linux lock installed and
passed `pip check`; OS package versions are recorded separately.

## Artifacts and validation

Local reference policies and their evaluation records:
`checkpoints/reference-20261025/seed-{1,2,3}/`. Full source, training logs, runtime
evidence and machine-readable `summary.json`: `runs/runpod-20261025/`.
These directories are Git-ignored.

Every policy passed the trusted artifact loader locally after download. Weight
digests matched its manifest, training record and evaluation report:

| Seed | SHA-256 of weights.json |
| --- | --- |
| 1 | `fec1691e39ed4104f4a9c7ecc36284ec1e5c22a4d72197bf7c7fbb78d48d1d47` |
| 2 | `44bf86e51d92dec007656e1ca8bc320815145da1f32b5730cde04c22dec41c4c` |
| 3 | `7234ac96abb37a510cfff27927121cf510717772384f9e10b7a7b5a919e99b1b` |

Training source SHA-256:
`b64658434caf26e681143c40966ac1a169b498957b077cb49df8051c4d8e64b6`.
All 36 original source-package files were checked against the sealed source
manifest before execution and after download. Git metadata was excluded from
the upload, so individual `run.json` files report Git revision/diff unavailable;
the supplemental source manifest and source snapshot identify the actual inputs.

Downloaded results archive SHA-256:
`5bb2375c068e561c39b868e9c07301a56eb1b5c13e3ce53e3b080d13f385b0d8`.
The remote and local archive checksums matched.

The runtime-downloaded PhysX 5.3.1 library was recorded with SHA-256
`4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706`.
Its hash was recorded after download, not enforced before first use. OS packages
and runtime assets are not covered by the Python wheel lock. SAPIEN emitted a
Vulkan ICD warning and supplied a fallback; state-based CUDA simulation passed.
Camera rendering was not tested.

## What actually ran on Runpod

The prepared custom Dockerfile was not built. We used the official catalog image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, whose resolved digest
from the Pod boot log was
`sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`.

`deploy/runpod/bootstrap.sh` created `/opt/automanip-venv` and installed our pinned
PyTorch 2.6 stack there. `deploy/runpod/run-reference.sh` ran the smoke checks,
three sequential training runs and separate development evaluations. The source
and all outputs were under `/workspace/automanip`, on the attached network volume.
SSH exposed TCP port 22 only. A task-specific key was supplied directly to this
Pod; the account's SSH key list was not changed.

Per-process deadlines bounded setup/training/evaluation processes. They did not
stop the cloud Pod automatically. The batch ended at 17:03:28 UTC; the Pod was
stopped at approximately 17:25:20 UTC after local backup. That includes about
22 minutes of avoidable post-run idle time while other chat/terminal work was
handled. A future external supervisor should sync artifacts and stop compute
as soon as a batch completes, independently of the conversational agent. Cloud
control credentials must stay outside the trainer.

## Resource state and costs

Fresh reads after the stop confirmed:

| Resource | ID | Final state |
| --- | --- | --- |
| L4 Pod, EU-RO-1 | `2tulwuoonz2aqm` | `EXITED`; compute stopped |
| Standard 20 GB network volume | `zvx3i1sjnx` | Retained with checkpoints/source/logs |

The Pod read quoted $0.59/hour and 3,707 seconds of runtime, implying about
**$0.61 GPU rental**, plus roughly $0.007 container disk and $0.002 storage.
These are estimates, not settled charges. Posted billing was still behind the
observed runtime at the first post-stop read:

| Category | Posted USD, 2026-10-25 daily billing bucket |
| --- | ---: |
| Pod GPU | 0.06041223183274269 |
| Pod disk | 0.0005787037080153823 |
| Pod CPU | 0 |
| Standard network storage | 0.0019444444915279746 |
| Serverless | 0 |
| Public Endpoints | 0 |
| Clusters | 0 |
| High-performance storage | 0 |
| **Total** | **0.06293538003228605** |

These figures come from `list-pod-billing`, `list-network-volume-billing` and
`list-billing`, not a charge inferred from the deposit. The costliest posted
component is Pod GPU. GPU billing ended with the stop. The retained standard
volume continues at $1.40/month (20 GB × $0.07/GB-month), prorated hourly, until
deleted. Container disk is ephemeral and is not billed while stopped, per
[Runpod storage pricing](https://docs.runpod.io/pods/pricing).

## Recommended next experiments

1. Record representative successful and failed rollouts, then add an independent
   confirmation-seed evaluation path. The current CLI always uses public
   development seeds; increasing its episode count alone does not create an
   independent confirmation set. Freeze the new evaluation protocol before
   selecting variants.
2. Measure learning efficiency: repeat the unchanged recipe at 60 and 180
   seconds for training seeds 1, 2 and 3. This is six new runs and 12 minutes of
   training. Reuse the existing 600-second points. Compare paired development
   success at equal budgets and log transitions as well as wall time.
3. At the 180-second budget, test learning-rate annealing and a small entropy
   coefficient separately, through the existing config fields. Keep task,
   controller, observations, evaluation seeds and scorer fixed. Use the same
   three training seeds. Confirm only promising changes with fresh evaluation
   seeds; reserve a private final holdout for final selection.
4. Build a separately versioned table-placement task requiring release and
   stability. The pure settling rule exists, but the custom environment and
   contact adapter still need implementation. Preserve stock PickCube as v0.

The near-ceiling ten-minute development score leaves little room to demonstrate
accuracy improvements on this easy distribution. Faster learning and a clearly
defined harder task are more useful next targets than a broad hyperparameter
search. Before unattended research, add filesystem/identity isolation, an
external budget/stop supervisor, confirmation gates and experiment history.
