# Cumulative-track preflight: the source already covers the target

The user authorized starting the prepared pilot on 2026-10-30. The registered zero-shot stopping rule fired: the unchanged source actor already exceeds the development success threshold in both domains. **No scratch or warm-start training was launched.** The new final-assessment seed set remains untouched.

| Same unchanged actor | Successes | Rate | Execution failures |
|---|---:|---:|---:|
| Original StackCube distribution A | 124/128 | 96.875% | 0 |
| Registered north-strip placement variation B | 122/128 | 95.3125% | 0 |

These are fresh development results on the proposed seed namespace, not independent confirmation or historical benchmark scores. No controller selection, parameter update, reward modification or extra policy was used. Both domains used weights SHA256 `ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`. The registered joint development threshold is at least 116/128 in each domain, with original-task loss no greater than six episodes against the paired unchanged reference. At zero updates the actor is its own reference.

## What actually ran

Two bounded Modal L40S preflight attempts ran. Both used the pinned image/dependencies, read-only PhysX runtime cache, blocked outbound network, restricted Modal access, fresh containers, no automatic retries and a 300-second function timeout. No Runpod resource was used.

The physical checks verified 48-dimensional finite observations, transformed cube poses matching the observed poses after GPU synchronization, the registered destination support, initial cube separation, finite 50-step physics and a destination remaining on the table. Effective clocks were 20 Hz control and 100 Hz simulation. Dependency versions were ManiSkill `3.0.0b21`, SAPIEN `3.0.0b1`, Torch `2.6.0+cu124`, NumPy `1.26.4`, Gymnasium `0.29.1`; actual hardware was NVIDIA L40S. The PhysX runtime SHA matched the pin.

The existing analytic URDF model matched the initial simulator TCP positions. Position-only inverse kinematics reached all eight diagnostic positions above the destination within 0.01 m. This calculation was a feasibility check, not collision-free manipulation proof. Its joint solutions were never executed or supplied to a learner. The subsequent actual source-policy stacks demonstrate practical solvability on the tested development episodes.

The completed corrected worker took **59.800 seconds**, including approximately **45.008 seconds for both development assessments**; its GPU wrapper took **61.183 seconds**. These are evaluation/preflight times, not training times. The first worker/wrapper took 19.093/20.752 seconds. Image build, submission and startup were additional overhead; provider app lifetimes were approximately 70.520 and 99.229 seconds and include such overhead/queueing rather than pure GPU compute.

## Batching fault and narrow repair

The first attempt exposed an integration fault in the new `assess_controller` wrapper: it passed all 128 seeds into an eight-lane adapter. The unchanged evaluator conservatively returned 128 execution failures per domain before any policy step. The initial preflight worker also failed to reject those error reports. Its `state=complete` and reserved `development_transitions=12800` field must **not** be read as a valid score or as completed experience. Both old reports record zero completed evaluation transitions.

The repair changes only the new cumulative-track wrapper to split seeds into complete vector batches, and changes the new preflight to distinguish reservations from completed transitions and reject any assessment execution failure. The historical task, evaluator, validator, success rules, source actor and development seeds were unchanged. A regression test checks multiple fixed-size batches, exact episode identities, no padding and rejection of incomplete batches. **78 selected CPU tests passed in 1.241 seconds** after the repair.

The old reports, logs and admission are preserved separately. Reconstructed copies of the two old source files were verified byte-for-byte against their admitted SHA256 digests and saved with explicit reconstruction provenance. A separate recovery app/volume then performed the first valid assessments; it did not repeat a previously completed policy evaluation.

## Experience and cost ledger

- First attempt: 400 mechanical transitions; zero completed development transitions; 12,800 development interactions reserved/charged conservatively because the original report was erroneous. No updates.
- Corrected attempt: 400 mechanical transitions plus 12,800 completed development transitions. No updates.
- **Total completed new environment transitions: 13,600.** Training transitions: zero. Independent final episodes: zero. Analytic FK evaluations: 6,836 across both attempts, separately counted; these are not simulator transitions.
- The conservative charged interaction total, including the first attempt's unreleased reservation, is 26,400. Completed transitions and reservations are reported separately rather than presented as the same quantity.
- Reported Modal cost for both apps: **$0.05353416** at the last read. Metering may lag. The incremental internal pilot ceiling was $5, with $0.25 reserved per preflight attempt; no training reservations were consumed. This is not the source policy's lifetime cost.
- The reused source retains its separately recorded 49,971,200 acquisition transitions / 2,295.408 trainer seconds, with source-only dollars unknown. Existing demonstrations were not used.

Both exact apps were read back as `APP_STATE_STOPPED`, with no live tasks/containers. No unrelated resource was stopped or deleted. Their data volumes and once-only local submission records remain preserved.

## Evidence and scope

The [scientific design](cumulative-transfer-plan-20261029.md) and original [plan JSON](../cumulative/experiments/stackcube-north-pilot-v1.json) remain preserved. The [repaired preparation](../cumulative/experiments/stackcube-north-pilot-v2.json) changes source pins for the batching repair; it does not change the task, thresholds, methods, seeds or budget. Its preparation remains non-executing: separate attempt admissions carry the user's later authorization.

Local evidence:

- `artifacts/cumulative/preflight-20261030-archive/`: original erroneous reports and logs.
- `artifacts/cumulative/preflight-recovery-20261030-archive/`: valid per-domain reports, physical diagnostics, logs and checksummed archive manifest.
- `.runtime/cumulative-launch-20261030/`: source-bound admissions, source reconstructions, separate launch records, billing snapshots and stopped-app evidence.
- [Preflight worker](../research/cumulative_preflight.py), [initial Modal adapter](../deploy/modal/cumulative_preflight.py), [recovery adapter](../deploy/modal/cumulative_preflight_recovery.py), [batched assessment wrapper](../src/automanip/cumulative_env.py), [regression tests](../tests/test_cumulative.py).

This supports **existing zero-shot coverage of a modest placement shift**. It establishes neither learning acceleration nor a new cumulative mechanism, and does not execute the new PPO adaptation path on real robot simulation. The source is a single historically trained seed, and this is one variation. The historical training-run count remains 72; these are evaluations/diagnostics, not new scientific training runs.

The north-strip variation is therefore too easy for the intended learning comparison. The next design should register a qualitatively meaningful reuse-and-correction requirement before querying new evaluation outcomes—for example, recovery after a specified physical disturbance—then first verify an actual source-policy deficit. Do not search placement coordinates against these development episodes until the source fails, or claim zero-shot success as a tenfold from-scratch improvement. No further task, adaptation run, replication or holdout query was automatically launched.
