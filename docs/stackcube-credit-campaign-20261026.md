# First StackCube research campaign

Authorized on 2026-10-26. This is a bounded first batch, testing longer credit
assignment while preserving the official task and reward. The current launch
is on Pod `2c1h36dlfn0t1f`, a Secure Cloud L4 in EU-RO-1 at the catalog rate of
$0.59/hour. The image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, resolved in boot logs
to digest `sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`.

## Registered experiment

- Candidates: PPO discount **0.9** and **0.95**, versus the existing **0.8** baseline.
- Training seeds: **1, 2, 3**, interleaved by seed: 0.9/seed 1, 0.95/seed 1, etc.
- Random initialization; unchanged official PPO source and stock reward.
- Same 1,024 environments, 50 rollout steps, eight epochs and 32 minibatches.
- Requested 25,000,000 transitions; actual **24,985,600 per run**, six runs total.
- Frozen 48-state/eight-action interface, joint-delta controller and evaluator horizon.
- Independent development curves at the same preselected checkpoints as baseline.
- **No confirmation queries or automatic promotion in this batch.**

Configuration: [`configs/stackcube-credit-20261026.json`](../configs/stackcube-credit-20261026.json).
Runner: [`scripts/run_stackcube_experiments.py`](../scripts/run_stackcube_experiments.py).
The runner records paired development wins/losses against each corresponding
baseline training seed. A candidate reaching 60% mean development success with
nonnegative changes on all three seeds is marked for confirmation review;
that flag does not constitute promotion or a holdout result.

## Isolation and durable outputs

Training, conversion and evaluation use different Linux UIDs. Root-owned
Landlock policies constrain filesystem access, including denying `/workspace`
to these worker processes. The trainer receives only the unchanged PPO program
and fixed command arguments. The converter uses restricted weights-only loading
and exact tensor-schema validation. The evaluator receives copied, immutable
actor-only JSON files; it never loads native checkpoints or candidate Python.

The network volume rejects ownership changes and does not enforce the tested
Unix access restrictions. Accordingly, identity-protected source and working
directories live under `/opt/automanip-credit-20261026`, on local POSIX storage.
Root, outside the worker sandboxes, mirrors stable checkpoints to the persistent
volume every five seconds and flushes final files before evaluation. The final
durable checkpoint hash must match the checkpoint used for evaluation. Only
known training output filenames and bounded regular files are copied.

This provides tested UID and kernel filesystem boundaries on a shared kernel
and GPU. It is not VM isolation or protection from a root-capable controller.
Cloud API credentials remain on the local machine, outside the GPU Pod.
CUDA also requires permission to set its own thread names. The policy permits
`/proc/self/task` writes for the worker process, without granting writes to
other processes or benchmark files. A direct restricted ManiSkill GPU check
passed after this correction. Device group membership was investigated and
reverted because it did not resolve the failure.

The first attempt (`765tbx0xbtfcal`) stopped safely at its ownership check,
before training. Its logs were backed up. The corrected launch directly tested Landlock denial of normal and symlink
access, then verified actual read/write operations for all three identities
before dependency setup. Earlier setup-only attempts also exposed an absent
optional asset cache and a permission-metadata probe that needed replacement
with real file operations. Their failure logs were backed up; no full training
runs were produced by those attempts. The original PickCube and StackCube baseline artifacts remain
separate from research outputs.

The end-to-end GPU smoke completed successfully: 400 training transitions,
restricted checkpoint export, and eight independent development episodes. Its
zero success rate is expected at this tiny budget and is not an experiment
result. The six-run production launcher started after this check passed.

## Monitoring and shutdown

Persistent campaign directory:
`/workspace/automanip-research-20261026-credit-v7` on volume `zvx3i1sjnx`.
The launcher runs detached from SSH. The independent local supervisor enforces
an eight-hour runtime ceiling and a $5 rate-based envelope, backs up the batch
when it finishes, verifies the archive digest, stops the Pod and confirms its
stopped state. It depends on this workstation remaining powered and connected.

Local supervisor state:
`.runtime/research-credit-v7-supervisor.json`.
Archive target:
`.runtime/research-credit-v7-20261026-results.tar.gz`.

From this project directory, connect with:

```sh
ssh -i .runtime/runpod-session-20261025_ed25519 -p 18997 root@213.173.105.12
```

Read `experiments.log`, `records/*/run.json`, `ledger.jsonl` and `summary.json`
under the campaign directory. `training/runs/*/durable-files.json` lists copied
checkpoint digests. Evaluator reports live under `evaluations/`, and safe actors
under `artifacts/`. SSH mappings can change after a stop/start.
