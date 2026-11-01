# Modal precision batch operations

Scientific scope and the outcome-dependent replication rule are frozen in
`docs/stackcube-precision-plan-20261027.md`, SHA256
`9daf8f11c8ed3ac0b9e8f28a5c5eb2478834bfb8d8baa3cb6eed7686e6e5f714`.
Root reviewed the complete adapter, source-generation guarantees and contracts,
independently passing five adapter tests and three precision tests before launch.
No new GPU smoke was needed for these isolated scalar/audit changes.

The exact three seed-1 arms were admitted under the shared $7 cap. App:
`ap-A5ZFpqYCeaYRkC1l3Up2WK`; coordinator:
`fc-01M5ZZ71ZWJ3VRXAE9AEAP21WX`. The admitted plan SHA256 is
`50416509abfb92b18421ae851175ede286ed4155c103c112760f2c6e426be298`.
Launch and copied admission records are under ignored
`.runtime/modal-precision-screen-*` paths. Do not repeat the entrypoint to inspect
progress: the campaign and individual jobs are protected by persistent once-keys.

| Arm | Trainer call | Admission UTC, 2026-10-28 | Original lease expiry UTC |
|---|---|---|---|
| precision-constant | `fc-01M5ZZ7B1FM8FRR5J2WHP9D5VD` | 00:40:43.490592 | 01:25:35.766682 |
| precision-entropy-anneal | `fc-01M5ZZ7EEKX6X3EFMWGVB6B0J7` | 00:40:50.084282 | 01:25:39.206534 |
| precision-lr-anneal | `fc-01M5ZZ7H8DT5Q7PZX734AHGREM` | 00:41:52.262733 | 01:25:42.860460 |

Each restricted L40S trainer has its own Volume named
`automanip-modal-{arm}-seed1-20261027`. Files live under its `/trial` subpath,
which the trainer sees as `/trainer-output`; the coordinator sees
`/{arm}-output/trial`. Each trainer also sees the validated read-only PhysX cache. The coordinator mounts
trainer Volumes read-only. Candidate networking and Modal RPC remain blocked;
no evaluator source, results mount or long-lived credentials reach the trainer.
Only reviewed self-contained source bytes may start one subprocess per durable
attempt marker. Preemption or queueing never refreshes the 2,700-second lease.

The trusted coordinator reuses the existing restricted weights-only converter
and unchanged frozen evaluator. At most nine development evaluations are selected
at 25.6M, 38.4M and final actual transitions. Coincident steps and identical policy
hashes are deduplicated. Raw native checkpoints/events/logs, safe actors and full
frozen development reports are retained. Measured cumulative optimizer steps,
all eight learned log standard deviations and actual entropy/LR traces must align
with every completed rollout; missing, nonfinite or inconsistent audit data fails
validation. Built-in evaluation experience is counted separately.

Conservative resource bound: $6.1615476, including 9,720 GPU-container seconds,
conversion and coordinator ceilings; reserve $0.8384524. Actual hardware names,
provider meter and lag, full archive verification and exact-app cleanup are checked
after terminal state. No confirmation or private holdout queries are authorized.

`deploy/modal/precision-replication-plan.json` contains six unlaunched recipe/seed
slots for seeds 2/3. The decision rule selects either constant alone (two jobs) or
a qualifying refinement with constant control (four jobs). `precision_replication.py` provides a separately reviewable adapter with default-denied
admission. No recipe/jobs, paid authorization or budget are selected. Root must
first review initial outcomes and bind exactly one branch. Proposed caps are
$4.50/$9 for two/four jobs, with resource bounds $4.1142924/$8.2088028. Four dedicated
trainers may run concurrently in the four-job branch. The initial seed-1 contract
remains unchanged. No replication resource has been created.

## Terminal results and archive

All three trainers completed exactly 49,971,200 transitions on actual NVIDIA
L40S hardware. Measured optimizer steps were 110,572 / 84,080 / 156,882 for
constant / entropy / LR respectively; trainer subprocess seconds were
2,295.408 / 2,172.979 / 2,466.694. Every required audit passed.

| Arm | 25.6M | 38.4M | Final 49,971,200 |
|---|---:|---:|---:|
| Constant | 117/128 | 126/128 | 127/128 |
| Entropy anneal | 117/128 | 116/128 | 121/128 |
| LR anneal | 117/128 | 123/128 | 127/128 |

Nine selected points required only seven distinct frozen development evaluations:
the three 25.6M actor hashes are identical, so their report was correctly reused.
These are single-seed development outcomes; no confirmation was queried.

Full outputs are under `runs/stackcube-precision-screen-20261027/`: 176 distinct
files, 192,969,983 bytes, verified after download. The first-pass immutable archive
manifest contains 178 references because the shared 25.6M report appears under
three curve references; it is stored physically once. The distinct-file proof is
`.runtime/modal-precision-screen-distinct-archive-verification.json`. Summary SHA256:
`21e300cf4467ee9dbeca1b91d35a3b5f8d6358f532bdf443f0ae939be95d2d2f`.
No files were overwritten and no native checkpoint was deserialized locally.

The exact app automatically stopped at 01:26:24 UTC, 2026-10-28; lifecycle read-back
confirmed stopped with zero tasks. No cleanup mutation was necessary. Reported
provider spend at close-out was $4.48262164, subject to meter lag. The shared
`.runtime/modal-tonight-budget-ledger.json` accounts for the whole session under
the user's $100 Modal cap, rather than treating per-campaign caps as new allowances.
