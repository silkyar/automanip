# AutoManip on a Runpod Pod

This prepares an interactive Linux NVIDIA Pod for the existing random-init PPO
trainer. Nothing starts training at container startup. The custom image build
remains unverified. The pinned Linux runtime has now passed 31 CPU tests, CUDA
smoke checks and three training/evaluation runs on an official Runpod image;
see [the reference report](../../docs/runpod-reference-20261025.md).

The tested route used the official Runpod Python 3.11 image, uploaded source to
`/workspace/automanip/source`, then ran `bootstrap.sh` to create
`/opt/automanip-venv` and install the Linux lock. `run-reference.sh` ran smoke
checks plus three seeded training/evaluation pairs. This avoids a custom image
build, but the `/opt` installation must be recreated after a Pod stop. The
Dockerfile route below bakes dependencies into a reusable image instead.

## Build and publish when ready

Docker is not installed on the current Mac. Use a Linux Docker builder or Docker
Desktop with an amd64 builder. The build needs package-index access, but no GPU.
The registry image must remain private if the source is private.

From the repository root, set `AUTOMANIP_IMAGE` to an image repository you own:

```sh
export AUTOMANIP_IMAGE=your-registry/your-repository:automanip-v0
docker build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -f deploy/runpod/Dockerfile -t "$AUTOMANIP_IMAGE" .
docker push "$AUTOMANIP_IMAGE"
docker image inspect "$AUTOMANIP_IMAGE" --format '{{json .RepoDigests}}'
```

Use the pushed `repository@sha256:...` image identity on Runpod. Commit the reviewed
source first for a meaningful revision label. The image also records hashes of
the actual copied files, including uncommitted files. `.git`, credentials, local
environments, checkpoints and run outputs are excluded from the build context.
The installed Python runtime and source stay under `/opt`, outside volume mounts.

## Pod settings

Create a **GPU Pod**, with these suggested settings after rechecking stock/price:

| Setting | Value |
| --- | --- |
| Cloud | Secure Cloud |
| GPU | 1 × `NVIDIA RTX A5000` (24 GB); A40 (48 GB) fallback |
| Data center | A location with current stock; match any attached network volume |
| Host CUDA floor | 12.8 or newer; the image's PyTorch runtime is CUDA 12.4 |
| Image | Your pushed immutable image digest |
| Container disk | 30 GB, provisional; measure after the first build |
| Network volume | Suggested 20 GB mounted at `/workspace`, for checkpoints/logs |
| TCP port | `22` only; enable a public TCP mapping for direct SSH |
| HTTP ports | None |
| Environment | `PUBLIC_KEY`: your SSH **public** key; no private key/API key |
| Driver capabilities | `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` |
| Startup override | Leave empty; the image starts an SSH server |

Hardware/disk sizes are provisional, not measured minima. An A5000 is an
economical first smoke-test choice; no throughput or 10-minute training target
has been validated on A5000. The first L4 batch completed about 5.6–5.9M
transitions per ten-minute run. This PyTorch 2.6/CUDA 12.4 recipe targets Ampere/Ada cards;
do not use it on Blackwell GPUs without revisiting the framework/runtime pins.

The standard Runpod PyTorch templates read earlier use other framework versions.
This image preserves `torch==2.6.0`, `mani_skill==3.0.0b21`, the 42-dimensional
state interface, and `pd_ee_delta_pose`. It uses a published PyTorch base whose
amd64 digest was checked against Docker Hub on 2026-10-23.

Use Runpod's displayed SSH address and port. After connecting interactively:

```sh
. /etc/profile.d/automanip.sh
cd /opt/automanip
mkdir -p /workspace/automanip/evidence
nvidia-smi
/opt/conda/bin/python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(16,device="cuda"); assert x.sum().item()==16; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
cp /opt/automanip-installed.txt /opt/automanip-source-revision.txt \
   /opt/automanip-source-sha256.txt /opt/automanip-requirements.txt \
   /workspace/automanip/evidence/
nvidia-smi -q > /workspace/automanip/evidence/nvidia-smi.txt
```

Keep the Pod ID, image digest, data center, quoted GPU/storage rates and actual
elapsed rental time with that evidence. The image excludes `.git`, so the current
trainer reports Git commit/diff as unavailable inside it; the revision label and
file hashes provide supplemental source identity. This is a local MVP workflow,
not a trainer/evaluator security boundary or an autonomous experiment runner.

## Check the simulator, then train

Check eight CUDA environments for one 50-step horizon without optimization:

```sh
cd /opt/automanip
/opt/conda/bin/python train.py --config configs/gpu-smoke.json \
  --backend physx_cuda --check-env
```

Expected output includes `wrapper_check: passed`, `num_envs: 8`, and
`observation_dim: 42`. The smoke configuration disables success-triggered resets
so every environment reaches the same horizon. CUDA arithmetic alone does not
validate PhysX. SAPIEN can
download its PhysX GPU library on first use; the download is not hash-locked in
this recipe. Preserve that library's hash and download version before reference
training. Panda/task assets may also be fetched on first use. `MS_ASSET_DIR`
persists task assets at `/workspace/mani_skill_assets`; the default PhysX cache
under `/root/.sapien` is on the container disk.

```sh
find /root/.sapien/physx -type f -exec sha256sum {} + \
  > /workspace/automanip/evidence/physx-sha256.txt
```

Run the tiny training check (800 transitions, up to 120 soft-budget seconds):

```sh
/opt/conda/bin/python train.py --config configs/gpu-smoke.json --backend physx_cuda \
  --seed 1 --output /workspace/automanip/runs/smoke-seed-1
/opt/conda/bin/python -c 'import json; from pathlib import Path; p=Path("/workspace/automanip/runs/smoke-seed-1"); r=json.loads((p/"run.json").read_text()); assert r["trained"] and r["learner_updates"] > 0 and not r["errors"]; assert (p/"policy/manifest.json").is_file(); print(r["status"], r["environment_transitions"], r["learner_updates"])'
/opt/conda/bin/python scripts/evaluate.py \
  /workspace/automanip/runs/smoke-seed-1/policy \
  --episodes 8 --num-envs 8 --backend physx_cuda \
  > /workspace/automanip/evidence/smoke-evaluation.log
```

Output directories must be new: on a repeat, choose another run name. The tiny
policy is a plumbing check, not a trained reference or a success-rate target.
Inspect the evaluation report for failed episodes before proceeding.

After smoke checks pass, preview or explicitly execute the three-seed baseline:

```sh
/opt/conda/bin/python scripts/run_baseline.py \
  --output-root /workspace/automanip/runs/reference
# Starts sequential training, seeds 1, 2, 3:
/opt/conda/bin/python scripts/run_baseline.py \
  --output-root /workspace/automanip/runs/reference --execute
```

Each baseline seed has a 600-second cooperative training budget and a 9,984,000
transition cap. There is no external process watchdog. Setup, serialization,
evaluation and idle Pod time can extend the billed rental. Evaluate each exported
policy separately with 128 paired development episodes before saving a reference.
Stock PickCube success does not require release. Independent confirmation seeds,
multiple training seeds and a private final holdout are required for later
promotion claims; the current CLI provides development evaluation only.

## Storage, cost and shutdown

Only an attached **network volume** makes `/workspace` independent of the Pod's
lifecycle. Save all run outputs there and retrieve them before deleting storage.
Container disk is disposable; a Pod volume disk survives stop/start but is lost on
Pod termination. A network volume survives Pod termination until separately
deleted, and continues billing. Never store the runtime under `/workspace`.

Finishing training or exiting SSH does **not** stop the Pod or its GPU charge.
After collecting results, stop the exact Pod ID in Runpod; verify its state and
remaining storage charges. Terminating a Pod deletes its local disks. There is no
automatic cloud shutdown in these scripts. The dated reference report records
the actual Pod/volume and their final state from the first executed batch.

If CUDA works but environment creation fails, first inspect the exception and
`vulkaninfo --summary`. Verify the host NVIDIA driver libraries and graphics
capability are mounted; do not install a host GPU driver inside the image or
silently change pinned dependencies. Record a failed GPU check as failed.

## Maintainer verification

`requirements/linux-cu124.txt` is a 98-package, hashed dependency resolution for
CPython 3.11/Linux x86_64/glibc 2.35. It installed successfully on the L4 Pod and
passed dependency checks, tests and training/evaluation. The SAPIEN beta pin is the pinned ManiSkill release's minimum
supported Linux version, and is distinct from the macOS development wheel.
Python wheels are pinned/hashed; Ubuntu apt packages and runtime-downloaded
PhysX/assets remain outside that lock. Record their versions/hashes on acceptance.

To regenerate intentionally with `uv==0.9.4` (review lock changes):

```sh
uv pip compile requirements/linux-cu124.in --python-version 3.11 \
  --python-platform x86_64-manylinux_2_35 --generate-hashes --only-binary :all: \
  --exclude-newer 2026-10-24 -o requirements/linux-cu124.txt
```

| Stage | Status |
| --- | --- |
| Recipe, lock, startup and smoke config authored | Passed |
| Shell syntax, config previews, 31 CPU tests | Passed locally |
| Linux dependency metadata resolution/install | Passed; 98 packages, L4 Pod |
| Custom image build | Unverified; no local Docker |
| Official image SSH and network-volume access | Passed; custom startup/restart remains unverified |
| Actual GPU simulation/training/evaluation | Passed on L4 |
| Runpod end-to-end test | Passed using official image plus bootstrap |
