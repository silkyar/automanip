# Modal integration and validation

User authorized using the Modal credit grant and delegated this setup on
2026-10-27. Existing Runpod experiments were left untouched. Initial validation
and the small reviewed CAPG smoke are bounded separately; full experiments are
not launched by this integration yet.

## Verified

Modal SDK **1.6.2** is installed in `.venv-modal`, separate from the simulator
environment. Credentials live in ignored `.runtime/modal-config.toml`, mode 600,
under profile `automanip`. The authenticated workspace is `silkyarora-2188`.
The browser-issued token has a 24-hour lifetime. No credential file or secret is
included in a GPU image.

GPU image base is pinned to
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`.
The original hash-locked Linux and StackCube requirements are installed
unchanged. The lean base additionally needs `libglib2.0-0` for OpenCV; omitting it
caused the first preflight failure before simulation. That operational attempt
and its logs remain preserved.

Actual Modal **NVIDIA L4** validation passed:

- CUDA arithmetic under torch 2.6.0+cu124.
- ManiSkill 3.0.0b21, SAPIEN 3.0.0b1, NumPy 1.26.4, Gymnasium 0.29.1.
- Frozen scorer/space checks and reproducible resets in GPU physics.
- Real Vulkan RGB rendering: eight 512×512 RGB frames with nonzero variance.
- Full 128-episode development replay of the existing seed-1 reference: **114/128
  = 89.0625%**, with **zero outcome or diagnostic mismatches** against Runpod.
- Restricted trainer GPU scene creation and stepping with outbound network
  blocked and no evaluator source, benchmark directory or results mount.
- Restricted Modal RPC denial and absence of explicit long-lived credentials.
- Read-only versioned PhysX runtime cache, with an actual failed write probe.

Proof: `.runtime/modal-validation-final.json`, run
`validation7a8afa9616f44be987f4`. The final trusted GPU validation took about
70 seconds. Its CPU cloud coordinator continued independently after the local
launch command exited. No confirmation episodes were queried.

The PhysX cache contains
`physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so`, SHA256
`4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706`.
This runtime-only subdirectory is mounted read-only at `/root/.sapien` in
restricted workers, which cannot download it with networking disabled.

## Isolation and transport

`deploy/modal/bundle.py` uploads explicit source allowlists. It binds validation
to the frozen protocol SHA256 and verifies every frozen source digest before
upload. The trainer image gets the official trainer only; it does not get the
trusted evaluator package, benchmark seeds, Runpod controller or credentials.
Automatic project-source inclusion is disabled.

Trainer functions use `restrict_modal_access=True`, `block_network=True`, fresh
single-use containers and CBOR-only restricted outputs. The concrete boundary
probe exercised all three protections; it did not depend on Landlock being
available under gVisor. Separate converter and evaluator containers retain
restricted tensor deserialization and independent state scoring. These are
tested platform/container boundaries, not a claim that a privileged account
owner or compromised platform cannot bypass them.

Large results need special handling: Modal's SDK tries to upload outputs above
its inline transport threshold via `BlobCreate`. Restricted functions correctly
receive an authentication error for that RPC. Safe actor JSON is approximately
3 MB, but gzip is approximately 1.3 MB. The adapter therefore uses bounded gzip
transport, rejects oversized, truncated and concatenated streams, and keeps
the frozen policy validator unchanged. It does not relax RPC restrictions.

## Reviewed CAPG smoke

`deploy/modal/smoke_capg.py` admits only the reviewed CAPG payload SHA256
`8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971`.
The restricted trainer completed **400 transitions in 22.777 seconds**, producing
checkpoint SHA256
`09a5fead336c9f5b0fdc8e3752471a052a14c8de6ed56ef243864aea28c3b043`.
This tiny run is a pipeline check, not evidence of better task performance.

The first conversion result hit the restricted large-output RPC issue above.
Its native checkpoint and training log were saved before conversion. Recovery
reuses that same checkpoint: training is not repeated. The compressed converter and the final frozen eight-development-episode
evaluation both passed, with zero execution failures and zero successes (as
expected for only400 training transitions). The authoritative recovery report
is `.runtime/modal-capg-smoke-final.json`; `training_repeated` is false. The
original failure report remains separate. All validation Apps were read back
stopped with zero active containers after completion.

## Commands

From the repository root:

```sh
export MODAL_CONFIG_PATH="$PWD/.runtime/modal-config.toml"
.venv-modal/bin/modal run --detach deploy/modal/app.py
```

This submits a cloud coordinator, which writes all validation stages to Volume
`automanip-modal-validation`. The launch record under `.runtime/modal-*-launch.json`
contains its FunctionCall ID and output path. It continues after laptop
disconnect; ordinary local-entrypoint chaining alone would not submit later
stages after a disconnect.

After passing validation, the explicitly reviewed smoke can be launched with:

```sh
.venv-modal/bin/modal run --detach deploy/modal/smoke_capg.py::capg_smoke \
  --validation-id validation7a8afa9616f44be987f4
```

Do not repeat a completed smoke just to recover transport. The conversion-only
recovery command is:

```sh
.venv-modal/bin/modal run --detach deploy/modal/smoke_capg.py::finish_smoke \
  --run-id capgsmoke3f75232af38247c1aa51
```

## Full-experiment admission and budgets

The concrete six-job plan can be generated without launching compute:

```sh
.venv-modal/bin/python deploy/modal/plan_trials.py
```

It pairs control/CAPG across three seeds, L4, at most 1,320 trainer seconds and a
25M requested transition ceiling, with development-only scoring and a maximum
of three parallel trainers. Its `launch_enabled` field is deliberately false
until the main agent reviews the complete smoke and cumulative budget handling.

A trusted cloud coordinator must claim each trial once and fix one absolute
epoch deadline before submitting the restricted trainer. Every platform attempt
inherits that original deadline and only receives its remaining time. Queue
time can conservatively consume the lease and must be logged separately from
actual trainer process time. No retry may receive a fresh 22-minute allowance.
Interruptions, invalid artifacts and budget overruns are recorded explicitly.
The smoke implements the same approach with a 120-second lease.

`deploy/modal/evaluate.py` prepares a separate promotion adapter. CPU admission
and GPU execution both claim the protocol/weights digest/split before scoring.
The fixed trusted GPU evaluator needs its own claim RPC to reject automatic
preemption replays; the trainer remains unable to call it. The confirmation
allowlist was initially empty. Main review subsequently admitted exactly the
three immutable full-horizon policy hashes in the promotion preregistration. A CPU-only exact-image preflight passed imports, protocol verification and a
synthetic claim-RPC check while networking was blocked. It created no benchmark
claims and evaluated no episodes. The scoring adapter was subsequently invoked once per admitted full-horizon
policy on the 512-episode confirmation split; see the promotion record below.

Sources: [images](https://modal.com/docs/guide/images),
[restricted functions](https://modal.com/docs/guide/restricted-access),
[batch processing](https://modal.com/docs/guide/batch-processing),
[preemption](https://modal.com/docs/guide/preemption).


## Full-horizon promotion confirmation

Main review authorized exactly three immutable full-horizon policies for one
confirmation query each. Before submission,
`deploy/modal/full-horizon-promotion-20261027.json` preregistered eligibility as
mean confirmation success at least 90%, with every training seed at least 85%.
The preregistration SHA256 is
`32a63fcd93d2cef359742e374c51353d09b0902a9d7ec3d735ba057053367b0f`.
This is promotion against the existing independent confirmation distribution,
not a private final holdout or a 10x sample-efficiency discovery. A failure does
not authorize tuning or another query.

`confirm_full_horizon` submitted three parallel CPU coordinators, each invoking
one L4 frozen evaluator. Each coordinator and GPU attempt claims the
protocol/weights/split once; automatic retries are disabled. The maximum is
three concurrent evaluators, 1,200 seconds per GPU function, and $2 additional
credits. No trainer was submitted. The app is
`ap-8A3U3Ee2io7IMxOwzgmDTK`; call IDs and exact policy hashes are preserved at
`.runtime/modal-full-horizon-promotion/launch.json`. Exact episode outputs stay
in ignored private confirmation results and are excluded from candidate-agent
inputs.


The first promotion submission was capacity limited. Seed 3 eventually started
after approximately 19.2 minutes of GPU queue time and completed its unique
512-episode confirmation in 145.437 seconds: 484 successes (94.53125%), zero
execution failures, Wilson 95% interval [92.21%, 96.19%]. Seeds 1 and 2 never
acquired GPU attempt claims or executed episodes. With their CPU result writers
near expiry, the app was stopped only after seed 3's report was persisted and
live GPU state showed zero running inputs. The three-seed eligibility decision
is pending; a single completed seed does not establish the mean or lower tail
across training seeds. No policy was queried twice.

The SDK returned NotFound when cancelling an unstarted child call ID that its
call graph still listed; this discrepancy was preserved. The completed seed's
exact report is under ignored `runs/private-confirmation/`, while operational
claims, meter data and an episode-free summary are in
`.runtime/modal-full-horizon-promotion/`. Recovery for the two never-started
seeds requires main review and must preserve the existing GPU once-only claims.


## L40S compatibility and audited recovery

Root review authorized a development-only compatibility probe on L40S using the
already-calibrated old reference policy. Requested and actual hardware were
L40S / NVIDIA L40S, with the same pinned package versions and PhysX blob hash.
The probe passed CUDA, physics and nonblank 512px Vulkan rendering. All 128
development outcomes and stage diagnostics exactly matched the old reference
(114/128 successes, no mismatches). It took 52.982 seconds and queried zero
confirmation episodes. Evidence: `.runtime/modal-l40s-compatibility-result.json`.
Its reported cost was $0.03656949.

After that gate, root authorized recovery of only the two policies that never
started on L4. `deploy/modal/full-horizon-recovery-20261027.json` records the
exact admitted hashes, prior stopped-app/absent-GPU-claim evidence and exclusion
of completed seed 3. Its admitted content SHA256 is
`48e431896964ecfb8780a41585a5f8793c8624df6b51f9a4d0bc62162e2acde8`.
The recovery coordinator preserves original CPU admission claims, adds separate
audited recovery claims, rejects any existing GPU attempt, and submits only
seeds 1 and 2. The GPU still atomically claims the same
protocol/weights/split key, independent of hardware or app, so a late old job
and a recovery cannot both score the policy. No scientific protocol changed.

The trusted GPU evaluator now commits its own started, complete or failure
record before returning, under
`frozen-evaluation-attempts/<protocol>/<weights>/<split>/result.json`. It validates
report weights and protocol before marking complete. CPU coordination has a
5,400-second timeout versus the unchanged 1,200-second GPU timeout, and records
queue latency separately. Hard interruption preserves the started global claim
rather than allowing a fresh attempt. The result Volume is mounted once on the
trusted evaluator; its pinned PhysX cache is hash checked and copied locally.
Restricted trainers retain only their read-only cache submount.

Recovery app `ap-iRYUW23YEpPqQ6AHra4XOq` submitted both L40S workers. Their GPU
claims started after 6.38 and 4.44 seconds of queue time. No seed 3 invocation or
new training was submitted.

All three immutable policies now have one completed 512-episode confirmation:

| Training seed | Successes | Rate | Per-policy Wilson 95% interval | Evaluation seconds | Evaluation hardware |
|---|---:|---:|---:|---:|---|
| 1 | 478/512 | 93.3594% | 90.86–95.21% | 98.338 | Requested L40S; observed NVIDIA L40S |
| 2 | 483/512 | 94.3359% | 91.98–96.03% | 99.996 | Requested L40S; observed NVIDIA L40S |
| 3 | 484/512 | 94.5313% | 92.21–96.19% | 145.437 | Requested L4; actual name not captured by old adapter |

The equal-seed mean is 94.0755% (1,445/1,536 successful episodes), the observed
minimum training-seed rate is 93.3594%, and the sample standard deviation across
three training seeds is 0.6278 percentage points. This passes the preregistered
point-estimate rule (mean at least 90%, every seed at least 85%). The same 512
confirmation episodes are shared across policies, so a pooled independent-
episode confidence interval is not claimed. Three training seeds provide
limited evidence about the wider training-seed distribution. This establishes
reference eligibility, not a private final holdout result or a 10x discovery.

Every report has the original admitted weights/protocol hashes, exactly 512
episodes, and zero policy execution failures. No training or policy repeat was
performed. The source promotion manifest and preregistration hashes remained
unchanged. Exact reports and their SHA256 digests are preserved read-only under
ignored `runs/private-confirmation/stackcube-full-horizon-20261027/`; the complete
episode-free handoff is `.runtime/modal-full-horizon-confirmation-summary.json`.

Both L40S GPU attempts completed and committed reports before SDK Blob/DNS
errors prevented their full result dictionaries from reaching the coordinator.
The original failed coordinator audits remain unchanged. A separate trusted
CPU reconciliation function, with no GPU handle, revalidated both durable
reports and wrote `reconciled-summary.json`; it executed zero new episodes.
Future GPU returns contain only small metadata and the durable result path,
and CPU coordinators read and validate the report from the Volume. This keeps
large output away from the blocked-network transport path.

All involved apps are stopped, with zero active containers. The reported
promotion, L40S compatibility, recovery and reconciliation cost is $0.23203069,
well within the shared $2 cap; metering may lag the final CPU-only operation.
Hardware labels above are retained separately, and these evaluator runtimes
are not training speed comparisons. Modal documents the explicit L40S GPU
selector in its [GPU guide](https://modal.com/docs/guide/gpu).


## Archived CAPG and SAC screens

The reviewed seed-1 CAPG pair and adapted SAC screen completed with no fresh
training attempts, extra development queries, or confirmation queries during
recovery. Full durable outputs are archived under
`runs/stackcube-capg-screen-20261027/` (95 files, 85,510,800 bytes) and
`runs/stackcube-sac-screen-seed1-20261027/` (34 files, 21,302,106 bytes).
Their `archive-manifest.json` files record every path, length, SHA256 and source
Volume. Local read-back also checked reviewed trainer hashes, safe-policy hashes,
frozen protocol identity and curve/report agreement. Native checkpoints were
archived as bytes and never deserialized locally. Existing analysis files were
retained without overwrites.

The SAC trainer stopped at its original deadline after 262,432 transitions,
1,292.195 subprocess seconds, and 129,216 critic, actor and temperature optimizer
rounds. All seven native actor exports, traces, reports and logs are preserved.
The four preregistered frozen development evaluations completed. Scientific
interpretation belongs to the separate campaign reports.

Both exact Apps were read back `APP_STATE_STOPPED` with zero live containers;
they had automatically stopped, so no cleanup mutation was needed. Their output
Volumes and once-only claims remain intact. Final reported billing read on
2026-10-28 was **$1.68908538** for CAPG app
`ap-ptx3nCf4npDjFOHmCCmtvS` and **$0.92470181** for SAC app
`ap-IbYvtzte9proBhw5qZDELK`, below their separate $3/$1.50 admissions.
The provider meter may lag. Exact operational proofs use ignored
`.runtime/modal-{capg,sac}-screen-*` paths. No other resources were changed.
