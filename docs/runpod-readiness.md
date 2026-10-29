# Runpod preparation and account snapshot

Checked 2026-10-24 UTC. These are point-in-time reads, not a
reservation or a promise of future availability. No cloud resources were created,
changed or deleted during this task.

## Hardware

`get-gpu-type` reads used Secure Cloud, one GPU, product `POD`, and a minimum host
CUDA version of 12.8:

| GPU ID | VRAM | Catalog GPU price/hour | Stock by data center |
| --- | --- | --- | --- |
| `NVIDIA RTX A5000` | 24 GB | $0.27 | EU-SE-1: LOW |
| `NVIDIA GeForce RTX 3090` | 24 GB | $0.50 | EU-CZ-1: LOW |
| `NVIDIA A40` | 48 GB | $0.59 | EU-SE-1: HIGH; CA-MTL-1: LOW |

All three reads reported available host CUDA 13.2 capacity. The version list is
GPU-wide; it does not establish a specific host CUDA version in each data center.

Recommended first smoke-test candidate: A5000. Its 24 GB is a provisional choice
for the eight-environment smoke test and the 512-environment baseline; neither
memory use nor throughput has been measured. A40 supplies more memory if needed.
The three 600-second training budgets alone equal 0.5 GPU-hours: at the read A5000
rate, that is $0.135 of GPU time. Setup, evaluation, idle time, overrun and storage
are additional. No storage rate was quoted or storage charge created here.

## Account and billing

Fresh `list-pods`, `list-endpoints`, `list-network-volumes`, `list-clusters` and
`list-global-volumes` all returned empty lists without further pages. There are
no existing resource IDs to stop, delete or reconfigure. Current resource burn is
zero, and no idle-spend recommendation is needed.

`list-billing(lastN=1, bucketSize="month")` returned no records and the following
zero totals for its resolved window, 2026-10-01 through 2026-11-01 UTC (the current
month bucket, still in progress):

| Billing category | Spend |
| --- | --- |
| Pods: GPU, CPU and disk | $0.00 |
| Serverless: GPU, CPU, disk and fees | $0.00 |
| Public Endpoints | $0.00 |
| Network storage: standard and high performance | $0.00 |
| Clusters: GPU, disk and networking | $0.00 |
| **Total** | **$0.00** |

Nothing has billed in this window, so there is no costliest resource or category.
This does not make a statement about billing outside October or account credit.

## Deliverables and validation

See [the deployment runbook](../deploy/runpod/README.md) for the container build,
registry handoff, exact Pod settings, SSH setup, smoke test, baseline commands,
storage lifecycle and shutdown instructions.

Authored: digest-pinned Dockerfile, SSH startup, build-context allowlist, 98-package
hashed Linux dependency resolution, eight-environment GPU smoke configuration and
runbook. Existing trainer, task interfaces and evaluator logic were preserved.

Passed locally: all 31 CPU tests; shell syntax; smoke and three-seed command
previews; lock resolution for Python 3.11/Linux x86_64/glibc 2.35; every package has
hashes; whitespace checks. The dependency resolver was installed under ignored
`.runtime/runpod-tools` and did not replace the current simulator environment.

Unverified: Docker image build, Linux dependency installation, SSH startup/restart,
volume persistence, GPU physics, GPU training/evaluation and Runpod end-to-end
execution. Docker is not installed on this Mac. No CUDA tests or reference-policy
performance results are claimed. OS packages, runtime-downloaded PhysX and assets
are not part of the Python lock.

The PyTorch base digest was obtained from
[Docker Hub's published tag metadata](https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/2.6.0-cuda12.4-cudnn9-runtime).
The container libraries and GPU setup were checked against the
[pinned upstream Dockerfile](https://github.com/haosulab/ManiSkill/blob/v3.0.0b21/docker/Dockerfile)
and [official installation guidance](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html).
The official PPO recipe was re-read at tag `v3.0.0b21`; this preparation retains
the existing documented algorithm/controller differences.
