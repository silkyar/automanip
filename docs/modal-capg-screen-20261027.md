# Modal CAPG seed-1 screen

Scientific admission and interpretation are fixed in
`docs/stackcube-capg-screen-20261027.md`. This adapter launches only the reviewed
full-horizon control and CAPG source hashes, training seed 1, dedicated L40S.
It does not run confirmation or authorize further training seeds.

`deploy/modal/screen_contract.py` fixes commands, checkpoint selection, bounded
artifact reads and cost accounting. `screen.py` provides restricted trainers,
a trusted CPU coordinator, and the CPU-only output transport preflight.
`capg-screen-plan.json` contains the root-reviewed admission and scientific-plan
SHA256. The launch record freezes its admitted content hash.

## Isolation and durability

Each trainer sees only its own new output Volume mounted at `/trainer-output`
and the previously validated read-only PhysX cache subpath. It sees no evaluator
source, benchmark distribution, evaluator result mount, other trainer output or
long-lived credentials. Modal RPC and external networking remain restricted.
The source payload is checked against its reviewed hash before execution.

The trainer writes checkpoints, TensorBoard events, raw logs and a bounded file
inventory directly to its own Volume. A durable attempt marker is kernel-synced
before starting the subprocess. A restarted input refuses another training
process if that marker exists. Small metadata receipts avoid asynchronous Blob
uploads. The trusted coordinator reads through bounded regular-file descriptors,
refusing final and ancestor symlinks and changing/oversized artifacts. Only the
existing restricted `weights_only=True` converter decodes native PyTorch data.
The evaluator receives safe numeric actor JSON, never a native checkpoint.

A real restricted CPU preflight passed RPC/network denial, absence of evaluator
and result mounts/long-lived credentials, exact persistence of a 3 MiB random
artifact through Volume-v2 kernel sync, and refusal of replay. It used zero
simulator episodes and no training. Proof:
`.runtime/modal-capg-screen-transport-proof.json`. Nine Modal unit tests passed,
including deadline preservation, strict artifact reads and cadence guards.

## Time and measurement

Each trusted CPU dispatch fixes one absolute 1,320-second lease. Capacity queue
wait consumes that conservative lease and is logged separately. The trainer
maps the remaining time to its process monotonic deadline with a 20-second save
reserve. An external process timeout uses the same original lease. No automatic
attempt receives a fresh allowance. The wrapper has a bounded postprocessing
reserve; its time is reported separately from subprocess time.

The unchanged source saves native cadence checkpoints. Frozen development
scores are requested only at 5.12M, 10.24M, 15.36M, 20.48M, and the final actual
step count. Later absent points are marked censored. Future-step artifacts are
rejected, coincident final/cadence steps are merged, and identical safe-weight
hashes are evaluated once. All native cadence checkpoints remain preserved.

Normalized curve rows contain `steps`, `successes`, `episodes` (128), and
`trainer_seconds`, plus artifact/report digests and separate evaluation runtime,
queue time and actual hardware metadata. Checkpoint trainer time is measured
from subprocess start, so it includes Python startup and built-in evaluation.
Raw TensorBoard data and all scalars are retained. Optional nonfinite diagnostic
scalars become JSON null with tag counts; nonfinite policy tensors still fail
conversion/evaluation. Exact optimizer-step count is unavailable because of KL
early stopping; configured epochs/minibatches give only an upper bound.

The ten development evaluations have 120-second GPU-function and 100-second
native-process limits. The frozen evaluator script, task, physics, success rules,
seeds and policy schema remain unchanged. GPU evaluation writes validated results
to its trusted Volume before returning a small receipt. Its global
protocol/weights/split once-key is independent of the training job claim.

## Budget and active launch

At the checked Modal rates, the conservative container-time calculation is
about $2.75 for GPU/CPU/RAM, conversion and coordination, leaving over $0.24
under the shared $3 cap for preflight, I/O and interruptions. This is a resource
ceiling estimate; actual metering and infrastructure interruptions are reported.
The plan admits at most two trainers and two independent evaluators concurrently.

Root reviewed the complete adapter and independently passed the nine tests.
It enabled exactly this pair. App: `ap-ptx3nCf4npDjFOHmCCmtvS`.
Coordinator: `fc-01M5ZGB916AHVJ5AH8GEPPK41T`.
The admitted plan SHA256 is
`b58e00c5bf971586f046ef2db4da335280e8b764b6b8b5a4e764d8ff67c011ab`.
Do not rerun the entrypoint to inspect progress; durable launch/job claims prevent
silent replay.

Raw output Volumes are `automanip-modal-capg-control-seed1-20261027` and
`automanip-modal-capg-treatment-seed1-20261027`, each under `/trial`.
The trusted summary is
`automanip-modal-validation/stackcube-capg-screen-20261027/summary.json`.
Local launch/progress records use ignored `.runtime/modal-capg-screen-*` paths.

Sources: [Modal Volume persistence and kernel sync](https://modal.com/docs/guide/volumes),
[preemption](https://modal.com/docs/guide/preemption),
[resource prices](https://modal.com/pricing).
