# Cumulative-learning review reconciled with the local repository

Date: 2026-10-29. This note supports the user's **local-only first cumulative-learning milestone**. It grants no cloud, GPU, simulator-collection or benchmark-scoring authorization. Historical admission flags and the previous research mandate do not override that restriction.

## Evidence and revision boundary

Read the downloaded `AutoManip_Source_Review.txt` (SHA256 `0a3d47038867570220840ff537c643cd267aad3fcfc0ee35a5f95c48f410f0d1`) and `AutoManip_Codex_First_Cumulative_Learning_Milestone.md` (`730709c3c6e32f0d2f291e8427ea717fa7f8df33f7c924530acd5ede9e79523a`), plus [REVIEWING.md](../REVIEWING.md), [PROJECT_SPEC.md](../PROJECT_SPEC.md), the [research mandate](../research/AUTOMANIP_RESEARCH_MANDATE.md), and the implementation paths below.

The static review names snapshot `bc0ffc20ae66796552caece65c611e62cc856f92`. Local HEAD is `97215850fbcc192c4ca7be5878b48e9fb066322e`, with substantial pre-existing tracked modifications and untracked source, deployment, tests and reports. The reviewed commit is not available as an object in this local checkout, so this is a source/evidence reconciliation, **not a claimed Git diff or ancestry comparison**. HEAD alone does not identify the working implementation. New attempt manifests must bind actual file hashes. Unrelated dirty files and historical records remain untouched.

## Findings checked

| Review finding | Current local evidence and implication |
|---|---|
| Historical task interfaces differ | [task_spec.py](../src/automanip/task_spec.py) fixes PickCube to `panda`, `pd_ee_delta_pose`, 42/7; [stackcube.py](../src/automanip/stackcube.py) fixes StackCube to `panda_wristcam`, `pd_joint_delta_pos`, 48/8. Direct weight transfer or vector padding is unsupported. Preserve both contracts. |
| Safe runtime is a stateless MLP | [policy_io.py](../src/automanip/policy_io.py) validates bounded JSON tensors, task identity, dimensions and architecture; [inference.py](../src/automanip/inference.py) applies tanh hidden layers and clips the linear output. This is not an arbitrary candidate-code runtime. Keep it unchanged for actor warm-start. |
| Useful assets exist but are not a tested cumulative loop | [amp_collect_v1.py](../research/discovery/amp_collect_v1.py) preserves complete teacher episodes, executed actions, 51 observations per 50 actions, rewards, successes/failures and provenance. [replay_robot_full_curve.py](../research/discovery/replay_robot_full_curve.py) preserves diagnostic model/RNG snapshots and recent behavior windows. Neither reviewed path proves cross-task adaptation or full learner/simulator resume. |
| Existing PPO initializes fresh state | [train.py](../train.py) constructs an actor, critic, log-standard-deviation and Adam optimizer, collects transient rollouts, then exports a safe actor. Its GAE, successor handling and PPO update are reusable; its historical task global is PickCube, so a separate compatible StackCube adaptation path is required rather than changing that global. |
| Replay snapshots are not complete resumes | Recent snapshots include actor, online/target critics, alpha and Torch RNG, but no optimizer/replay/environment/controller-state resume contract. The new path deliberately restores none of those states. |
| Residual exploration is not correction control | [bc_residual_exploration_v2.py](../research/discovery/bc_residual_exploration_v2.py) calibrates Gaussian noise from demonstration residuals. It does not implement a reusable learned residual controller around a frozen base. |
| Research evidence and autonomy already exist | The mandate and `research/discovery/decisions.jsonl` already describe mechanisms, predictions and failures. A new agent hierarchy, generic memory database or selector is not needed for this milestone. Campaign dispatch is not itself an adaptation policy. |
| Retention requires behavior measurement | Preserving source weights only preserves a file/mapping. Assess the **same adapted controller** on source A; do not give retention credit for routing back to the archived source policy. |

The safe actor is a deterministic evaluation export. PPO's sampled Gaussian exploration is learning state, not part of that export. Actor-only warm-start therefore must explicitly select a fresh exploration initialization; it cannot claim restoration of the source learner's exploration state.

## Material updates after the review snapshot

The review's main architectural conclusions remain accurate in the paths checked. Two dated findings supersede its unrun-diagnostic descriptions:

- The [history-motion probe](history-motion-results-20261029.md) ran once and failed its registered material-effect gate in both fixed seed corpora. History error was 9.662 versus 11.642 mm for the wider current-state control in seed1 (17.0%, below 20%); seed2 history error 9.219 exceeded 8.975 mm. This admits neither a recurrent learner nor a claim that history is generally useless. The corpora are not a matched causal seed comparison.
- The [soft-value envelope audit](soft-value-envelope-results-20261029.md) subsequently completed without fitting or simulator queries. Late ordinary SAC had upper-bound breaches against a conditional frozen-policy/constant-alpha reference, including the twin minimum on 57/3,328 stored behavior state/actions. Historical thousand-scale minibatch targets were not reconstructed. This does not establish true historical Q error, a training bug or the cause of regression, and does not justify a new intervention.

The [full-task SAC curves](full-stacking-curve-results-20261029.md) remain negative for final stacking at every declared checkpoint. Intermediate lifting/transport and subsequent regression are context; they are not cumulative learning or a tenfold efficiency result. The dated aggregate reports supersede stale pending-confirmation statements in older plans/mandates; no historical document is rewritten here.

## Local assets: upload omission is not local absence

The following files were found locally and their SHA256 digests checked without executing a policy or simulator:

| Asset | Exact identity and evidence |
|---|---|
| Source actor | `runs/stackcube-bc-exploration-pilot-20261028/teacher-input/{manifest.json,weights.json}`. Weights `ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`; safe format `automanip-json-f32-v1`, hidden sizes `[256,256,256]`, interface 48/8, StackCube fingerprint `b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f`. |
| Complete public-training dataset | `runs/stackcube-bc-exploration-pilot-20261028/collect/native/dataset/trajectories.npz`: `495a392364926211397222bfa77e79ff4a6c2c87564f226dc1212f333ebec782`. Metadata `dataset.json`: `dbb3c1c4b21adf24c2d600de576807c458487b3eeeae07aded85cc517f661c1b`. 512 completed episodes, 497 successful/15 unsuccessful, 25,600 transitions, no incomplete rows, public vector-reset seeds 90,281,000–007. Collection-loop wall 26.571935388 s; enclosing collector-process wall 29.36977253 s is charged by the asset registry. Attributable dollars unknown. |
| Source acquisition | Precision-constant training seed1: 49,971,200 transitions, 2,295.408300588 trainer seconds. Exact source-only dollar attribution is unknown; do not divide campaign costs or call it free. |
| Applicability evidence | This exact seed1 actor scored 127/128 historical development and **497/512 = 97.07%** independent confirmation. **95.90% is the three-training-seed recipe mean**, not this actor's score. The dataset's separate 497 successful training episodes are not the confirmation episodes. See [confirmation aggregates](stackcube-precision-confirmation-results-20261028.md). |

Bulk weights, training NPZ files and native checkpoints were intentionally excluded from the uploaded snapshot. Their absence there is not evidence that collection or asset validation failed. No protected confirmation episode output was inspected for this reconciliation. Historical aggregate confirmation evidence is provenance only, never new adaptation feedback.

## Narrow implementation decision

Root owns the local implementation: contracts/assets and assessment in `cumulative.py`, existing PPO math reused by `cumulative_ppo.py`, a small episode recorder, and a reset-only goal adapter where needed. This note does not certify their in-progress implementation or test outcomes.

The supported operation is **actor-only ordinary PPO adaptation**: import compatible source actor tensors, initialize a fresh critic and Adam optimizer, use fresh attempt RNG and explicitly reset Gaussian log-std to −0.5. No replay, old critic, optimizer moments, normalizer state or simulator state is restored. Training restarts at episode boundaries. This is warm-start adaptation, not exact continuation. The current pilot consumes the policy asset only; the existing demonstration dataset remains a provenance/round-trip fixture and contributes no BC updates or online labels.

Reuse the current PPO GAE, clipping, successor/timeout semantics and safe export boundary. Existing [PPO tests](../tests/test_ppo.py) and [rollout tests](../tests/test_rollout.py) cover useful algebra and boundaries; reading those tests is not a new test run. Root's narrow patch must independently demonstrate compatible initial-action parity, incompatible-asset rejection, fresh-state semantics, partition/boundary preservation, budget/censoring handling and behavioral-retention assessment. Report exact commands and outcomes with that patch.

There is **no missing local actor/dataset blocker**. Real goal-reset feasibility, reachability, timing, simulator parity, learning efficacy and source retention remain unvalidated. The [unlaunched transfer plan](cumulative-transfer-plan-20261029.md) defines that future test. Full resume, corrective controllers, consequence models, automatic selection and the later A→B→C claim are out of scope now.
