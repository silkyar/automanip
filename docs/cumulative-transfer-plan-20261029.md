# Unlaunched cumulative-track pilot: StackCube north-strip adaptation

Registered design date: 2026-10-29. **Design/local validation only; no cloud, GPU, simulator collection, benchmark scoring or spending is authorized.** Historical admitted flags confer no authority. The user requested a thin locally testable reuse path; a future execution requires separate review of exact source/protocol/assets, feasibility and budgets.

## Question and prior information

Does ordinary actor warm-start reduce new experience needed on a held-out **StackCube placement variation**, while the same adapted controller retains source behavior? This is not unseen-task acquisition: the source actor already learned stacking. It tests policy reuse against scratch context, not superiority over ordinary fine-tuning (which is the transfer method here). No correction/model/selector mechanism is added before observing B's actual failure mode.

Source actor: `runs/stackcube-bc-exploration-pilot-20261028/teacher-input/`, weights SHA256 `ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`, hidden sizes `[256,256,256]`. Source acquisition is 49,971,200 transitions/2,295.408300588 trainer seconds; attributable dollars unknown. This actor's historical independent confirmation is 497/512 (97.07%); 95.90% describes the three-seed recipe. Those historical results/seeds are not reused as this pilot's retention assessment.

The existing 512-episode/25,600-transition public-training dataset is available and hashed in the [reconciliation](cumulative-review-reconciliation-20261029.md). **Neither arm trains on it** in this pilot. No BC, old replay, diagnostic actions, controller assistance or protected evaluation trajectory enters learning.

## Family and distribution split

Preserve source A: the pinned official `StackCube-v1` initial sampler, reward and success. Define B, `stackcube-north-strip-v1`, by applying the following deterministic map to **the official sampled initial poses**, once during reset:

| Position coordinate | Cube A | Cube B (stacking destination) |
|---|---|---|
| x | `0.4 * original_x` | `0.25 * original_x` |
| y | `0.12 + 0.1 * original_y` | `0.34 + original_y / 15` |
| z and quaternion | Unchanged | Unchanged |

The official sampler draws a shared xy shift in `[-.1,.1]` plus placements with y in `[-.2,.2]`, giving original y support `[-.3,.3]`. B's cube-A y is `[.09,.15]`; its destination y is `[.32,.36]`, outside A's destination support. Original x support `[-.2,.2]` becomes A x `[-.08,.08]`, B x `[-.05,.05]`. The y gap is at least .17 m, so the transform does not introduce initial cube overlap geometrically. **Robot reachability, contacts, valid observations and practical solvability are unverified.** Pure coordinate arithmetic is not a simulator feasibility test. If invalid/unreachable, stop and register a revised variation before training; do not silently tune this transform after seeing outcomes.

No extra RNG draws, physics steps, demonstrations or target-task search are used by the transform. Apply it inside the new track's reset path, refresh observations after pose changes, and initialize diagnostic source height/distance from the transformed poses. Do not alter the official sampler file, task reward, evaluator kernel or historical fingerprints.

Both domains retain `panda_wristcam`, state observations 48, normalized actions 8, `pd_joint_delta_pos`, 20 Hz control/100 Hz physics, horizon 50 and original success semantics. Arm actions are current-joint deltas up to ±.1 rad; gripper action is its existing normalized mimic position target (approximately `0.015 + .025*a` m), not a new force command. Observation layout is qpos 0:9, qvel 9:18, TCP pose 18:25, cube-A pose 25:32, cube-B pose 32:39 and existing relative vectors 39:48; poses use scene-local positions/quaternion wxyz. No new goal feature is appended: cube B's observed pose encodes the goal.

Pins: ManiSkill `3.0.0b21`, upstream `6226567ce326625fa6eacf9aeb3f4008e4339f7e`, StackCube source SHA `ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440`; SAPIEN `3.0.0b1`, production Torch `2.6.0+cu124`, NumPy `1.26.4`, Gymnasium `0.29.1`; PhysX blob `4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706`. A future run must verify effective clocks/pins rather than assume them from a Mac CPU check.

Keep artifact/interface compatibility separate from evaluation-distribution identity. Retain the current safe MLP format/validator and compatible actor schema. Give A/B in this cumulative track separately hashed goal/reset/seed protocol identities; **never store a B score under the historical StackCube protocol or reuse its scoring claims**. Trusted evaluation may reuse the immutable rollout/metric kernel through a separately reviewed reset-only adapter. No candidate Python is imported by scoring.

## Methods and initialization

| Method | Prior policy | Adaptation |
|---|---|---|
| Unchanged source reference | Exact source actor | Zero updates; zero-shot B and fresh A retention reference. |
| Scratch PPO context | Fresh random actor | Same architecture, PPO settings, seeds and incremental budget as warm-start; no source information. |
| Ordinary actor-warm-start PPO | Exact source actor mean network | Ordinary PPO fine-tuning. This is the essential simple transfer control for any later mechanism. |

For both trainable arms, critic, Adam moments/step counters and attempt RNG are fresh; Gaussian log-std is **−0.5**, trainable thereafter. No source log-std is implied by the deterministic safe actor. No normalizer or replay is restored; there is no cross-task reward relabeling. Use unchanged existing PPO math, sampled Gaussian-action log probabilities and executed-action clipping. Preserve actual pre-autoreset successors, termination versus timeout and GAE boundaries. Train full 50-step episodes, without early reset on success.

Freeze scalars: gamma `.9`, GAE lambda `.9`, LR `.0003` constant, entropy `.003`, clip `.2`, value coefficient `.5`, max grad norm `.5`, target KL `.1`, eight epochs, 32 minibatches, 64 environments × 50 steps. Batch is 3,200/minibatch 100. Maximum 2,048,000 training transitions gives 640 rollouts and at most 163,840 Adam calls per arm; KL/time limits may reduce actual work. Record it rather than infer equal optimizer counts from equal transitions. Source acquisition used 1,024 environments: the new batch size/throughput is a declared adaptation configuration, not a reproduction of that source run.

## Seeds and partitions

These are fresh public proposed namespaces, not historical private holdouts. They must be rejected if a preparation inventory discovers a prior use/collision; choose a new complete registration before any execution, not after outcomes.

| Purpose | Exact allocation |
|---|---|
| Pilot learner/exploration initialization | 41 in both trainable arms; source actor remains fixed. |
| Pilot training vector resets | Scalar public seed `93,440,000 + b`, batch b = 0…639. Each batch has lanes 0…63; episode identity is `(batch_seed, lane)`, not a claim of 64 replicated states. |
| Development A and B | 128 base episode seeds `93,450,000…93,450,127`, reused across declared checkpoints/methods; B applies only its frozen pose transform. |
| Independent final A and B | 512 base episode seeds `93,550,000…93,550,511`, inaccessible to training/checkpoint selection; no historical confirmation reuse. |
| Future replication slots only | Learner seeds 42/43; vector-reset seeds `93,441,000+b` / `93,442,000+b`, b = 0…639. No execution is admitted. Final replication protocol/assessment access must be reviewed before those runs. |

Declare learner and reset RNG handling in the implementation: fresh attempt state, no source RNG restoration, reproducible reset schedule and recorded exploration sampling. Shared seed labels support common-scenario comparisons, **not a guarantee of identical physical trajectories across runs**. Do not use the failed paired-state physical probe as proof of such identity.

Training episodes/data remain public-training assets; development data/feedback are separately tagged and never silently inserted into PPO rollouts; final evaluation outputs never become training assets. Persist lane, episode/reset identity, executed and sampled/proposal actions when different, actual successor, terminated/truncated, task/goal, behavior digest and complete/incomplete status. Preserve all successes, failures and partial attempts. Extend the existing collector/recorder conventions rather than introducing a second collection system.

## Budgets, cadence and clocks

Per trainable arm: **at most 2,048,000 new B training transitions and a 600-second total adaptation window**. Count source lookup/validation, initialization, resets, collection, PPO updates, export and scheduled A/B development feedback inside that window. A future driver must enforce it; a trainer-only 600-second clock does not establish a ten-minute adaptation result. Provider queue/image/startup/upload time must be separately logged and included in any claimed end-to-end wall time. Final assessment below is separately budgeted after freezing controllers.

Development checkpoints: **0, 128,000, 512,000, 1,024,000, 2,048,000 training transitions**, after the corresponding learner update block. With 3,200-step rollouts these are iterations 0/40/160/320/640. Score A and B on the same checkpoint, at most 128 episodes/domain; deduplicate identical controller digests without losing checkpoint aliases. Maximum two arms × five checkpoints × two domains × 128 × 50 = **128,000 development transitions**, charged separately from the **4,096,000 training-transition ceiling**. Zero-shot source assessment is warm-start step0, not an extra uncounted policy. The same source-A reference can be reused across paired development retention comparisons.

These ceilings assume no measured new-track throughput. Reaching 2.048M within 600 seconds requires at least 3,413 training transitions/s before evaluation/other overhead; 64 environments and small minibatches can be much slower than the historical source recipe. This is a **ceiling, not a prediction or speed claim**. Evaluation may consume the window; log missed points/unfinished batches and update-censoring. Do not extend, resume, replace a missed exact-step primary with another checkpoint, or interpolate success. An unscheduled deadline-terminal export is archived but not an additional feedback/scoring endpoint. All attempt, reset, simulator, optimization, inference, validation, export and development work is counted.

Use one hardware class for the future paired run and record actual devices/parallelism. Two simultaneous learners each receive 600 seconds; report up to 1,200 aggregate adaptation-process seconds, not 600 total compute. Simulated parallel experience is not physical-robot time. No dollar allowance is activated here; any future monetary envelope must include startup, both attempts, development, final assessment, setup/failures and metering reserve within a newly reviewed shared budget.

## Selection, retention and independent assessment

Success is the existing **end-of-50-step stock StackCube metric**. Grasp/lift/transport/contact-free/stability are diagnostics, not substitute objectives. Target threshold is ≥90%; retention requires **the same adapted controller** to score ≥90% on A **and** lose at most five percentage points versus the unchanged source controller evaluated on the same A seeds.

Development integer gates (n=128): B ≥116 successes; adapted A ≥`max(116, source_A_successes − 6)`. The six-count allowance is ≤5 pp; seven counts would exceed it. Select the **earliest declared checkpoint satisfying both**; if none qualifies, select the last fully evaluated declared checkpoint for failure description only. Do not select different A/B controllers or use a task router. If the unchanged source already meets both gates at step0, report zero-shot coverage/no demonstrated adaptation need; do not launch unnecessary adaptation under this plan without a revised scientific reason.

Freeze source reference and one checkpoint per trainable arm before independent assessment. Proposed final ceiling: those three controller digests × A/B × 512 episodes × 50 = **153,600 final transitions**, separate from adaptation feedback. Integer final gates: B ≥461/512; adapted A ≥`max(461, paired unchanged_A_successes − 25)`. Final outputs are assessed once and do not select replacement checkpoints or authorize retries. If source itself fails its fresh A applicability check, report that limitation rather than replacing the source secretly.

Report first **observed** threshold checkpoints with prior observed brackets in transitions and total adaptation time; no monotonicity or interpolation assumption. Missing/unreached thresholds are censored, never invented speedups. Zero-shot has zero new training interactions but nonzero assessment/lookup cost. A one-seed pilot supports feasibility and directional evidence only; the final assessment is independent of development but does not remove training-seed uncertainty. No automatic promotion, replication or new mechanism follows.

Future replication at seeds 42/43 would add up to 8,192,000 training transitions, 256,000 development transitions and 2,400 aggregate adaptation-window seconds; six controller-domain final evaluations per paired seed add at most 307,200 final transitions. These are separate **unlaunched** ceilings, not an approval or throughput estimate. Freeze any new final-assessment allocation before replication and avoid repeated adaptive use of the pilot's final set.

## Lifetime accounting and falsification

Record the source training once in an immutable parent-asset lifetime ledger; link it from both reference and warm-start attempts rather than repeatedly adding 49.971M as new experience. Still display that prior-information advantage against scratch. The unused demo dataset retains its separate 25,600-transition/29.370-second collector-process acquisition entry (26.572 seconds inside the collection loop); no demonstration/BC cost is hidden in this policy-only pilot. Original broader research/search costs remain historical; do not equate source-only cost with the entire project or invent a per-source dollar allocation.

New pilot maxima, if every endpoint is reached: 4,096,000 training + 128,000 feedback + 153,600 final = **4,377,600 simulator transitions**. Report partial/failing attempts, evaluation/reset work, optimizer calls, all CPU/GPU time, actual dollars or explicit unknowns and inference overhead. Current local unit tests consume no new simulator interactions and prove neither reachability nor adaptation.

- If zero-shot already works, no learning acceleration is established.
- If ordinary warm-start reaches the joint target/retention gate earlier than scratch, that supports **prior-policy transfer on B**, not a novel cumulative mechanism or unseen-task claim.
- If warm-start is no better, harms A, or reaches B only by sacrificing A, report the corresponding negative. If scratch also fails, assess feasibility/configuration before adding complexity; no automatic extra budget.
- Censored curves establish no measured speed ratio. The first pilot does not promise 2× or 10× acceleration.

For the later A→B→held-out C study, freeze the accumulated library and all C procedures **before C feedback**. With versus without a B asset tests additional prior information; a proposed mechanism versus ordinary fine-tuning/pooled learning with the **same B experience** tests how that information is used. Do not credit a representation/selector for extra B data alone. C and a correction/world-model/selector are deliberately not implemented or admitted now.

## Readiness boundary

Local assets are present; safe warm-start, compatibility rejection, episode persistence, fresh-state semantics, exact budget accounting and target/retention assessment can be CPU-tested now. Production goal-reset/controller/observation validity, north-strip reachability, learning efficacy, timing and behavioral retention remain unverified. Root's local patch/test report supplies implementation evidence separately. No current historical benchmark or evaluator file must change, and this document authorizes no execution.
