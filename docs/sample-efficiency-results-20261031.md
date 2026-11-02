# Sample-efficiency screen results (October 31, 2026)

Protocol and decision rule: [plan](sample-efficiency-plan-20261031.md). Every run is listed. Scores are successes out of 128 public-development episodes (seeds 97,100,000–97,100,127) at each 1.024M-transition export. No protected development or confirmation seed was queried. All runs reached 15.36M transitions within the 2,700 s trainer ceiling; there were no trainer or scorer failures in batches 1–2.

Source: `research/sample-efficiency-20261031/ppo.py` (batch 1 sha256 `7fce49b3…`; batch 2 adds Student-t, default path unchanged). Launcher: `deploy/modal/sample_efficiency.py`. Analysis: `scripts/analyze_sample_efficiency.py`.

## Batch 1: update cadence and observation normalization

T80/T90 are the first sustained crossings of 103/128 and 116/128 (M transitions); "—" means censored at 15.36M.

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | seed 13 T80 / T90 | median T80 | trainer s (15.36M) |
|---|---|---|---|---|---|
| control (1024 envs) | 14.34 / — | 11.26 / 14.34 | — / — | 14.34 | 583–743 |
| env512 | 7.17 / 9.22 | 8.19 / 10.24 | 12.29 / 14.34 | **8.19 (0.57×)** | 991–1090 |
| env256 | 9.22 / 14.34 | 8.19 / 14.34 | 7.17 / 10.24 | **8.19 (0.57×)** | 1969–1984 |
| obsnorm (1024) | 8.19 / 10.24 | 10.24 / 13.31 | 11.26 / 12.29 | 10.24 (0.71×) | 599–609 |
| obsnorm-env256 | 11.26 / 14.34 | 11.26 / 13.31 | 7.17 / 12.29 | 11.26 (0.79×) | 1713–1958 |

Final-checkpoint successes (15.36M): control 115, 120, 82; env512 109, 114, 119; env256 117, 116, 115; obsnorm 120, 114, 121; obsnorm-env256 118, 117, 121.

Under the preregistered rule, `env512` and `env256` are promising (all three seeds reach T80, median ≤0.67× control, no seed worse). `obsnorm` and `obsnorm-env256` are not. Fewer environments buys transitions, not wall-clock: env256 takes about 2.7× the trainer seconds of control per transition, though its time to T80 (~1,050 s median) is similar to control's (~690 s for the two seeds that got there; one did not).

**Stage diagnostics.** In every control seed, placement is reached in 115–128/128 episodes by 4.1M but release stays at 0–2/128 until 8.2M (seed 12), 10.2M (seed 11) and 14.3M (seed 13). env256 places by 2.0M and releases in 65–85/128 by 4.1M. Most of the cadence speedup is earlier escape from the release plateau.

## Batch 2: Student-t (df 2) exploration — negative

| arm | seed 11 | seed 12 | seed 13 |
|---|---|---|---|
| t-gripper (1024) | 0 at all checkpoints | 0 | max 1/128 |
| t-all (1024) | 0 | 0 | 0 |
| obsnorm-env256-t-gripper | T80 11.26, T90 14.34 | T80 10.24, T90 13.31 | max 1/128 |
| obsnorm-env256-t-all | max 5/128 | 0 | 0 |

Nine of twelve runs never exceed 5/128. The 1024-env runs place in 119–127/128 episodes and release in 0–1/128 through 15.36M. The registered hypothesis (heavier tails cross the flat gripper dead zone) is refuted.

**Why.** Exported actors evaluated on 9,180 placed states drawn from old control traces: Gaussian control actors keep the pre-clip gripper mean near −0.95 (median, at 6.1M); Student-t actors sit at −1.06…−2.04. The environment clips commands below −1, so moving the location beyond the clip is free; heavy tails cause drops while carrying, and the policy defends the grasp by moving its location deeper into the flat region, widening the dead zone it was meant to cross.

## Batch 3: Bernoulli gripper (negative) and env128

| arm | seed 11 | seed 12 | seed 13 | trainer s |
|---|---|---|---|---|
| bin (1024) | max 1/128 | 0 | 0 | 615–753 |
| env256-bin | max 1/128 | 0 | max 1/128 | 2039–2057 |
| env128 | T80 8.19, T90 9.22 | T80 11.26, T90 — | T80 8.19, T90 — | 2671 (ceiling) at 10.9M / 12.4M / 11.0M |

The binary arms learn the early stages fastest of anything so far (env256-bin places in 81/128 episodes at 1.02M and 126/128 at 2.05M), then never release (0–2/128 through 15.36M). Exported actors on old-trace placed states have gripper logits −19…−28 (p_open 1e-9…1e-13). Unbounded certainty, not the dead zone alone, is the mechanism: carry states punish opening without limit, placed states share the representation, and the Bernoulli score gradient of a rare open is proportional to p, so learning stops with exploration. env128 is no better than env256 in transitions and is ceiling-limited in wall-clock.

A 2.048M smoke (seed 11, separate volume, not part of the screen) preceded batch 3 and ran cleanly.

## Batch 4: bounded gripper exploration mass — positive

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | seed 13 T80 / T90 | median T80 | trainer s |
|---|---|---|---|---|---|
| gfloor (σ≥0.5 + mean barrier, 1024) | 11.26 / 11.26 | 9.22 / 12.29 | 10.24 / 12.29 | 10.24 (0.71×) | 748–794 |
| env256-gfloor | 8.19 / 8.19 | 10.24 / 11.26 | 7.17 / 9.22 | 8.19 (0.57×) | 2014–2158 |
| **bineps** (ε 0.02 + logit L2, 1024) | 8.19 / 10.24 | 8.19 / 9.22 | 9.22 / 10.24 | **8.19 (0.57×)** | 904–940 |
| **env256-bineps** | 7.17 / 8.19 | 9.22 / 12.29 | 7.17 / 9.22 | **7.17 (0.50×)** | 1928–2228 |

`bineps` and `env256-bineps` meet the promising rule against `control`. `bineps` matches the best batch-1 cadence result in transitions at control's wall-clock, with T90 reached in all three seeds (control: one of three). Final successes at 15.36M: bineps 124, 123, 126; env256-bineps 119, 118, 116. Release still starts about 4M transitions after placement (bineps first releases at 7.2M; it places by about 3M).

## Batch 5: exploration floor size and persistence — null

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | seed 13 T80 / T90 | median T80 | final (15.36M) |
|---|---|---|---|---|---|
| bineps05 (ε 0.05) | 8.19 / 8.19 | 12.29 / 12.29 | 9.22 / 9.22 | 9.22 | 124, 124, 127 |
| bineps-sticky50 | 7.17 / 9.22 | 8.19 / 8.19 | 9.22 / 9.22 | 8.19 | 119, 122, 121 |
| bineps-sticky80 | 8.19 / — | 9.22 / — | 11.26 / — | 9.22 | 101, 102, 99 |

None improves on `bineps` (8.19, 8.19, 9.22). ε 0.05 delays placement (seed 12 places only at 6–8M), so more exploration mass costs carrying. Persistence ρ 0.5 is within noise of `bineps`; ρ 0.8 reaches T80 but plateaus at 99–116/128 and never sustains T90 (persistent opens likely disturb the final stable stack). The release switch-on near 7M in the binary arms is not a schedule artifact: the entropy/LR schedule is constant at these budgets.

## Batch 6: exact base-yaw symmetry augmentation — null for T80

Simulator probe (public reset seed 97,200,000; 64 states): static observation error ≤5e-6 per dimension; reward differences ≤2.3e-3 over 10 steps under identical actions.

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | seed 13 T80 / T90 | median T80 | final | trainer s |
|---|---|---|---|---|---|---|
| sym (θ 0.4, control base) | 12.29 / 14.34 | 11.26 / 14.34 | 9.22 / 11.26 | 11.26 (0.79×) | 116, 119, 119 | 688–849 |
| bineps-sym (θ 0.4) | 8.19 / 9.22 | 9.22 / 9.22 | 10.24 / 11.26 | 9.22 | 126, 126, 120 | 1329–1421 |
| bineps-sym08 (θ 0.8) | 8.19 / 9.22 | 10.24 / 11.26 | 9.22 / 10.24 | 9.22 | 124, 121, 124 | 998–1334 |

Symmetry speeds placement on the control base (placed in 42–107/128 at 3.1M) but release is unchanged in timing, so T80 does not improve on `bineps`.

**Logged exploration std** (audit/logstd, three runs read): release starts once arm dims 0–3 reach σ≈0.2/0.13. Control: by 5M, release 8–10M. bineps: about 6M, release 6–7M. env256-bineps: 2–3M, release about 5M. Batch 7 tests this precision clock.

## Batch 7: initial exploration std and entropy — negative / null

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | seed 13 T80 / T90 | median T80 | trainer s |
|---|---|---|---|---|---|
| bineps-std37 (init σ 0.37) | 12.29 / 14.34 | 12.29 / — | 14.34 / — | 12.29 | 956–979 |
| bineps-std22 (init σ 0.22) | — (never places) | — | — | censored | 780–946 |
| bineps-ent0 | 8.19 / 14.34 | 8.19 / 9.22 | 8.19 / 10.24 | 8.19 | 679–784 |

The precision-clock hypothesis is refuted in its simple form: smaller initial exploration slows placement learning (σ 0.22 runs place in ≤1/128 through 15.36M). Entropy 0 is within noise of bineps. Operational: the seed-12 bineps-ent0 scorer call failed with `FileNotFoundError` after training completed; re-scored with `score_only` on unchanged artifacts.

## Replication r1 (fresh seeds 21–25) — fails the preregistered bar

| arm | T80 per seed 21–25 (M) | median T80 | vs fresh control | T90 per seed |
|---|---|---|---|---|
| control | —, 10.24, 10.24, 9.22, 11.26 | 10.24 | 1× | —, 12.29, 14.34, 10.24, 12.29 |
| bineps | 9.22, 9.22, 8.19, 7.17, 8.19 | 8.19 | 0.80× | 9.22, 9.22, 9.22, 8.19, 9.22 |
| env256-bineps | 9.22, 7.17, 7.17, 7.17, 7.17 | 7.17 | 0.70× | 13.31, 11.26, 10.24, 9.22, 12.29 |

Neither arm meets the 0.67× rule; both beat the paired control on all five seeds. The fresh control is faster than batch 1's (median 10.24M vs 14.34M), so part of the batch-1 margin was control-seed noise. Pooled over eight seeds (11–13, 21–25): control 11.26M, bineps 8.19M (0.73×), env256-bineps 7.17M (0.64×). `bineps` reaches T90 in 5/5 fresh seeds by 9.22M; control in 4/5 by 10.24–14.34M.

## Batch 8: env512-bineps (exploratory)

T80 6.14, 6.14, 5.12M; T90 8.19, 7.17, 6.14M (seeds 11–13). Trainer seconds to 6.14M: 533–543 (1,124–1,152 s for 15.36M; control 583–743 s for 15.36M). Replication r2 on seeds 21–25 registered before outcome.

## Replication r2: env512-bineps on fresh seeds 21–25 — fails the preregistered bar

T80 7.17, 8.19, 5.12, 7.17, 7.17M (median 7.17M = 0.70× of the r1 fresh control's 10.24M; bar ≤6.86M). T90 9.22, 8.19, 7.17, 7.17, 11.26M. Trainer seconds for 15.36M: 1,152–1,447.

## Summary (status at handover pause, $82.35 Modal spend on Oct 31)

- **Runs:** 98 screened training runs (batches 1–8, r1, r2), plus two smoke batches (4 runs) and one simulator probe. Every run trained to 15.36M or the 2,700 s ceiling (env128 only); one scorer call failed transiently and was re-scored.
- **Mechanism:** the PPO StackCube plateau is mostly the release step. Policies place by about 3–5M and then fail to release, because nothing bounds the gripper's certainty: carry states punish opening, placed states share the representation, and for a Bernoulli parameterization the score gradient of a rare open also vanishes. Student-t noise and an unfloored binary gripper make this worse (measured pre-clip means −1.1…−2.0 and logits −20…−28 on placed states).
- **What helps:** a training-time Bernoulli gripper with an exploration floor (ε 0.02) and logit L2 (`bineps`), and smaller rollouts (512/256 environments). Exported artifacts are the unchanged restricted MLP.
- **Best arm:** `env512-bineps`. Pooled over 8 seeds: median T80 6.66M vs control 11.26M (0.59×); better than the paired control in 8/8 seeds; T80 reached in about 9–10 trainer minutes vs about 10–11 for control. On fresh seeds alone it is 0.70×, failing the preregistered 0.67× bar, as do `bineps` (0.80×) and `env256-bineps` (0.70×).
- **Not established:** a large (≥3×) sample-efficiency gain, or any protected-confirmation result. The historical 95.9% confirmation is unchanged; no protected seed was queried.
- **Null/negative:** observation normalization (alone or with env256), Student-t exploration, unfloored binary gripper, Gaussian σ floor + barrier (weaker than bineps), larger ε, sticky gripper, symmetry augmentation under PPO, lower initial σ, entropy 0, env128.

## Off-policy pilot o1 (TD3) — preregistered gate fails; strongest learner so far

Seeds 11–12, 5.12M transitions, exports every 512k. Successes/128 at 3.07 / 3.58 / 4.10 / 4.61 / 5.12M:

| arm | seed 11 | seed 12 | sustained T80 (512k grid) | trainer s |
|---|---|---|---|---|
| td3 | 2 / 3 / 97 / 122 / 123 | 2 / 44 / 120 / 125 / 124 | 4.61M, 4.10M | 407, 378 |
| td3-burst (p 0.005) | 0 / 0 / 93 / 115 / 122 | 0 / 0 / 1 / 102 / 122 | 4.61M, — | 422, 420 |
| td3-burst-sym | 0 / 0 / 0 / 0 / 53 | 1 / 5 / 103 / 119 / 116 | —, 4.10M | 477, 454 |

The go gate (both seeds ≥103 by 3.07M) fails. Plain TD3 places in 126–127/128 by 2.56M and releases within about 1–1.5M more, without exploration bursts: the actor follows the critic's action gradient, which does not vanish with the policy's own probability of opening. Bursts do not help; symmetry slowed one seed (no lift until 3.6M). Fresh-seed replication r3 registered before outcomes.

## Replication r3: TD3 on fresh seeds 21–25 — fails the preregistered 0.5× bar (0.55×)

| seed | T80 (512k grid) | T90 | T80 on 1.024M grid | trainer s at T80 |
|---|---|---|---|---|
| 21 | 5.12M | 5.63M | 5.12M | 417 |
| 22 | 6.14M | 6.66M | 6.14M | 480 |
| 23 | 6.66M | 7.17M | censored (7.68M run) | 524 |
| 24 | 4.10M | 4.61M | 4.10M | 334 |
| 25 | 5.63M | 5.63M | 6.14M | 464 |

Median T80 5.63M vs the fresh PPO control's 10.24M on the same seeds (0.55×; 0.60× on the PPO grid) and vs `env512-bineps` 7.17M (0.79×). Median T90 5.63M vs control 12.29M (0.46×). TD3 reaches T80 in 6–9 trainer minutes vs about 10–11 for PPO control. Final successes at 7.68M: 118, 118, 116, 124, 123.

## Screen o2: TD3 update-to-data ratio

| arm | seed 11 T80 / T90 | seed 12 T80 / T90 | final (5.12M) | trainer s |
|---|---|---|---|---|
| td3 (4 updates/step, o1) | 4.61 / — | 4.10 / 4.61 | 123, 124 | 407, 378 |
| td3-utd8 | 3.07 / 3.58 | 4.10 / 4.61 | 124, 117 | 609, 605 |
| td3-utd16 | 3.58 / 3.58 | 2.05 / 3.07 | 122, 124 | 1027, 1009 |

`td3-utd16` meets the screen rule (earlier on both seeds); replication r4 on fresh seeds registered before outcome. Higher UTD costs wall-clock: about 2.6× trainer seconds per transition.

## Replication r4: td3-utd16 on fresh seeds 21–25 — **passes** the preregistered bar

| seed | T80 (512k grid) | T90 | T80 (1.024M grid) | final (7.68M) | trainer s (7.68M) |
|---|---|---|---|---|---|
| 21 | 4.61M | 5.12M | 5.12M | 124 | 1572 |
| 22 | 3.58M | 3.58M | 4.10M | 127 | 1581 |
| 23 | 3.58M | 4.10M | 4.10M | 122 | 1573 |
| 24 | 3.07M | 3.58M | 3.07M | 121 | 1570 |
| 25 | 3.58M | 5.12M | 4.10M | 124 | 1569 |

Median T80 3.58M vs the fresh PPO control's 10.24M on the same seeds (**0.35×**; 0.40× on the PPO grid), median T90 4.10M vs 12.29M (0.33×), final success 121–127/128 (94.5–99.2%) on every seed. Trainer time to T80 ≈ 12 minutes (similar to the PPO control's ≈10–11 minutes to its later T80). This is the first preregistered fresh-seed pass in this work: about 3× fewer environment interactions to 80% and 90% with final success in the mid-to-high 90s.

## Transfer t1/t2 (research-side evaluation; frozen evaluator covers only StackCube)

ManiSkill `success` at episode end on 128 fixed reset seeds 300,000,000–300,000,127, deterministic action, identical across arms.

**PlaceSphere-v1 (release required)**

| arm | T80 per seed | median T80 | peak | final | trainer s |
|---|---|---|---|---|---|
| ppo-control | —, 10.24, 8.19 | 10.24 | 85, 116, 125 | 83, 112, 101 | 793–894 |
| ppo-env512-bineps | 4.10, 5.12, 4.10 | **4.10 (0.40×)** | 123, 119, 120 | 98, 103, 108 | 1509–1706 |
| td3 | 5.12, —, 5.63 | 5.63 | 118, 113, 107 | 118, 109, 109 | 625–667 |

**PickCube-v1 (hold at goal)**

| arm | T80 per seed | median T80 | final |
|---|---|---|---|
| ppo-control | 4.10, 4.10, 5.12 | 4.10 | 128, 120, 126 |
| ppo-env512-bineps | 4.10, 3.07, 4.10 | 4.10 (1.0×) | 128, 128, 128 |
| td3 | 3.07, 3.07, 3.07 | 3.07 (0.75×) | 128, 128, 128 |

The release fix transfers to the other release task (0.40× to T80) and does not hurt holding. On PlaceSphere every arm's success decays late in training, so none meets the ≥116 final requirement; this is not specific to the new methods.

## t3/t4: td3-utd16 transfer; o3: UTD 32

| task | T80 per seed 11–13 | median T80 | final | vs PPO control (t1/t2) |
|---|---|---|---|---|
| PickCube-v1 | 2.05, 3.07, 2.05M | 2.05M | 128, 127, 128 | 0.50× (control 4.10M): transfers |
| PlaceSphere-v1 | —, 4.61M, — | censored | 104, 113, 99 | does not transfer (2/3 seeds first succeed at 6.7–7.7M) |

StackCube `td3-utd32` (seeds 11–12): T80 2.56, 3.07M; T90 4.10, 4.10M; final 123, 117; trainer 1,874–1,889 s for 5.12M. Not earlier than utd16 on seed 12 (2.05M): not promising; UTD 16 kept.

Candidate digests for a possible protected confirmation of the r4 final policies (not run; requires root-reviewed admission): [td3-utd16-confirmation-candidates-20261031.json](td3-utd16-confirmation-candidates-20261031.json).

## t5: PlaceSphere with linear LR decay; t6: LiftPegUpright transfer

**PlaceSphere-v1, LR annealed to 0 over 15.36M**

| arm | T80 per seed 11–13 | median T80 | final | trainer s |
|---|---|---|---|---|
| ppo-control-anneal | 8.19, 9.22, 14.34 | 9.22 | 110, 102, 125 | 759–812 |
| ppo-env512-bineps-anneal | 4.10, 4.10, 4.10 | **4.10 (0.44×)** | **121, 124, 125** | 1566–1674 |

Annealing removes the late decay for the release-fix arm (finals 94.5–97.7%) but not for the control (median 110).

**LiftPegUpright-v1 (no release)**

| arm | T80 per seed 11–13 | median T80 | final | trainer s |
|---|---|---|---|---|
| ppo-control | 5.12, 5.12, 5.12 | 5.12 | 127, 128, 127 | 633–642 |
| ppo-env512-bineps | 3.07, 3.07, 3.07 | 3.07 (0.60×) | 128, 128, 128 | 1074–1306 |
| td3-utd16 | 3.07, 3.58, 2.56 | 3.07 (0.60×) | 126, 126, 128 | 1547–1585 |

On this task the bineps arm's gain is presumably from the 512-env cadence (no release step); the two components are not separated here.

## Replications r5 (PlaceSphere) and r6 (PickCube) on fresh seeds 21–25 — both **pass**

Same research-side evaluation as t1–t6. Preregistered rule ([plan](sample-efficiency-plan-20261031.md)): treatment reaches T80 in at least as many seeds as control, median T80 ≤0.67× control, median final ≥116 and not more than 5 below control's.

**r5 — PlaceSphere-v1, LR annealed over 15.36M**

| arm | T80 per seed 21–25 | median T80 | T90 per seed | final | trainer s |
|---|---|---|---|---|---|
| ppo-control-anneal | 7.17, 8.19, 10.24, 8.19, 8.19 | 8.19 | —, —, —, 8.19, 11.26 | 109, 106, 106, 115, 116 (median 109) | 800–972 |
| ppo-env512-bineps-anneal | 6.14, 4.10, 4.10, 4.10, 4.10 | **4.10 (0.50×)** | 6.14, 10.24, 7.17, 8.19, 4.10 | 126, 124, 121, 123, 119 (median **123**) | 1560–1726 |

Every control seed is at 0/128 through 6.1M and first succeeds at 7.2–9.2M; every treatment seed is at 105–122/128 by 4.1M. The control's median final (85%) fails the 116 bar; the treatment's (96%) passes it. Combined with t5 (seeds 11–13), the median T80 over 8 seeds is 4.10M vs 8.19M, and the treatment is earlier on 8/8 seeds.

**r6 — PickCube-v1**

| arm | T80 per seed 21–25 | median T80 | final | trainer s |
|---|---|---|---|---|
| ppo-control (1.024M grid) | 4.10, 4.10, 3.07, 4.10, 3.07 | 4.10 | 124, 128, 127, 128, 127 | 520–571 |
| td3-utd16 (512k grid) | 2.05, 2.05, 1.54, 1.54, 2.05 | **2.05 (0.50×)** | 128, 128, 127, 128, 128 | 1525–1591 |

On the PPO 1.024M grid, td3-utd16 reaches T80 at 2.05M on all five seeds, so the ratio is also 0.50×. Its trainer seconds for 7.68M are about 3× the control's for 15.36M, so this is a sample gain and not a wall-clock gain.

## Final summary (Nov 1; $137.80 Modal spend for this whole sample-efficiency effort, cap $200)

- **Runs:** 179 screened from-scratch training runs: 98 PPO (batches 1–8, r1, r2), 22 StackCube TD3 (o1–o3, r3, r4), and 59 transfer runs (t1–t6, r5, r6). Four smoke-batch runs and one simulator probe are additional. No trainer failures. One transient scorer failure was re-scored. No protected seed was queried.
- **StackCube headline (frozen scorer, public-dev episodes):** `td3-utd16` (TD3, 1024 envs, 16 updates/step, n-step 3, LayerNorm twin critics, per-env noise and gripper mean barrier, exporting the unchanged restricted MLP) passes the preregistered fresh-seed bar. Median T80 is 3.58M vs 10.24M for the PPO control (**0.35×**) and median T90 is 4.10M vs 12.29M (0.33×). Finals are 121–127/128 (94.5–99.2%) at 7.68M, i.e. about **2.9–3× fewer environment interactions**, with wall-clock time to T80 similar (≈12 min).
- **Mechanism (transferable):** with PPO, release plateaus come from unbounded gripper certainty. A training-time ε-floored Bernoulli gripper (`bineps`) plus 512-env cadence fixes it on StackCube (pooled 0.59×, fresh 0.70×, fails the bar), and on PlaceSphere (fresh-seed **pass**, 0.50×, finals 96% with LR annealing). On no-release tasks it is neutral (PickCube) or helps through cadence (LiftPegUpright, 0.60×, unseparated).
- **Off-policy transfer:** td3-utd16 passes on PickCube (fresh seeds, 0.50×, finals 99–100%) and is 0.60× on LiftPegUpright (seeds 11–13). It **does not** transfer to PlaceSphere (2/3 seeds censored).
- **Not done:** protected confirmation of the r4 StackCube policies (candidates prepared, needs root-reviewed admission). No ≥10× gain.

## Nov 1 follow-ups

### d1: why td3-utd16 fails on PlaceSphere — placement precision, not release

Rerun of seeds 11–13 with per-stage counts (research-side evaluation, 128 episodes per 512k checkpoint). T80 is censored on all three seeds (it was censored on 2/3 in t3); finals are 104, 104, 99.

| seed | grasped by 0.5M | first checkpoint with on-bin ≥20/128 | median closest object–bin xy at 2.0 / 4.1 / 5.1M | first success ≥63/128 | then |
|---|---|---|---|---|---|
| 11 | 128/128 | 5.12M (53) | 20.5 / 12.9 / 4.4 mm | 6.66M (113) | on-bin 128, grasped-at-end 0 |
| 12 | 128/128 | 6.14M (25) | 23.5 / 13.7 / 6.6 mm | 6.66M (63) | 104 at 7.68M |
| 13 | 128/128 | 6.66M (41) | 22.8 / 12.2 / 11.9 mm | 7.68M (99) | — |

The policy grasps and carries by 0.5M, then spends 4–6M transitions within about 1–2 cm of the goal without entering the 5 mm on-bin tolerance. When it does enter, release and success follow within one checkpoint (grasped-at-end drops from about 128 to 0). The dense reward changes by about 0.025 (out of 13) between 10 mm and 5 mm, so the decisive on-bin transitions are a rare frontier event for the critic. On StackCube the stage counts show the same pattern at placement (r4 seed 21: placement 3→118/128 between 1.0 and 2.6M). This motivates o4/d2 (prioritized replay).

### a1: separating the two components of `env512-bineps` on LiftPegUpright (no release)

| arm | T80 per seed 11–13 | median | final | vs t6 |
|---|---|---|---|---|
| ppo-env512 | 3.07, 3.07, 3.07M | **3.07M (0.60×)** | 128 ×3 | same as env512-bineps |
| ppo-bineps (1024 envs) | 5.12, 7.17, 5.12M | 5.12M (1.0×) | 128 ×3 | no gain (control 5.12 ×3) |

On a no-release task the whole gain comes from the 512-env update cadence; the ε-floored gripper is neutral at best (one seed later). This matches the mechanism: the floor targets release, so its benefit is specific to release tasks (StackCube, PlaceSphere), whereas cadence helps generally.

### o4: prioritized replay for td3-utd16 on StackCube — negative

Seeds 11–13, 5.12M, frozen public-development scorer at 512k exports. One scorer call (control, seed 13) failed with a transient `FileNotFoundError` after training and was re-scored with `score_only` on the unchanged artifacts.

| arm | T80 per seed 11–13 | median T80 | final | trainer s |
|---|---|---|---|---|
| td3-utd16 (paired control) | 2.56, 2.56, 3.58M | 2.56M | 123, 125, 120 | 1032–1081 |
| td3-utd16-per (α .6) | 3.58, 3.58, 4.10M | 3.58M (1.40×) | 116, 125, 121 | 1072–1104 |

PER is later than its paired control on every seed (by one to two checkpoints) and does not meet the 0.8× bar. Success under PER stays at 0–8/128 until 3.07M, while two control seeds pass 100/128 by 2.56M. Reweighting by TD error does not speed up the frontier stages here. A plausible reason (not tested) is that early large-TD transitions are noisy contact/grasp transitions, and the importance weights shrink the effective batch. Not replicated; PER is dropped for StackCube.

### d2: prioritized replay on PlaceSphere — negative (worse than uniform)

`td3-utd16-per` (α .6), seeds 11–13, 7.68M, stage diagnostics: **0/128 success at every checkpoint on all three seeds** (uniform-replay d1: finals 104, 104, 99). On-bin is reached in at most 25/128 episodes at any checkpoint. The median closest object–bin distance never settles below about 4 mm and swings back up to 13–55 mm late in training. The preregistered prediction (PER fixes the precision frontier) is refuted. With o4, PER is dropped on both tasks: reweighting the few on-target transitions that exist does not substitute for producing more of them. This leaves demo-seeded resets (reverse curriculum) as the untested candidate for precision frontiers, as the literature review recommends.

### smoke-x1 (PokeCube, 1.024M, execution check after the predictions were registered)

All 12 trainer calls completed. At 1.024M, research-side successes: PPO control 0, 0, 0; env512 22, 6, 0; env512-bineps 0, 0, 0; td3-utd16 40, 128, 125 (seeds 11, 12, 13). The predictions in [lesson-transfer-predictions](lesson-transfer-predictions-20261101.md) were written before this smoke and are not changed. The early TD3 numbers already point against P3; the full x1 run scores the predictions.

## Protected confirmation of td3-utd16 (once-only, root-approved) — **target met**

Approved one-time by the repository owner on Nov 1 ([plan](td3-utd16-confirmation-plan-20261101.md)). 512 frozen confirmation episodes per policy, the unchanged frozen evaluator, the final 7.68M export of every r4 seed (fixed before any query). The first launch omitted `--detach`: the app stopped when the local entrypoint returned, and the coordinator ended before claiming the campaign. It was verified that no claim, summary or evaluation attempt existed for any digest, so no protected episode was queried. That is recorded in `.runtime/modal-td3-confirmation-setup-failure-1.json`, and the campaign was then launched once with `--detach`.

| seed | success at end | success ever | Wilson 95% |
|---|---|---|---|
| 21 | 483/512 (94.34%) | 98.63% | 92.0–96.0% |
| 22 | 490/512 (95.70%) | 99.22% | 93.6–97.1% |
| 23 | 489/512 (95.51%) | 98.83% | 93.3–97.0% |
| 24 | 488/512 (95.31%) | 99.02% | 93.1–96.8% |
| 25 | 492/512 (96.09%) | 98.63% | 94.0–97.5% |
| **mean** | **95.39%** | | |

Preregistered target: mean ≥95% and every seed ≥90%. **Met** (mean 95.39%, minimum 94.34%). For comparison, the historical 50M PPO recipe confirmed at 95.90% on the same confirmation distribution after 49.97M transitions (about 37 trainer minutes per seed on an L40S). The TD3 recipe confirms at 95.39% after 7.68M transitions (about 26 trainer minutes per seed): about 6.5× fewer environment interactions and about 0.7× the trainer time for a similar confirmed success. To 80–90% on public development episodes, TD3 is about 40% slower in wall-clock than the PPO control (median 12.4 vs about 8.5 minutes to 80%; one of five control seeds never reached 80%).

## x1: lesson transfer to an unseen task (PokeCube-v1; predictions registered first in [lesson-transfer-predictions](lesson-transfer-predictions-20261101.md))

Research-side evaluation, 128 fixed episodes, T80 = 103/128, seeds 11/12/13:

| Arm | T80 | Finals |
|---|---|---|
| ppo-control | 10.24, 7.17M, censored at 15.36M | 111, 124, 0 |
| ppo-env512 | 9.22, 3.07, 3.07M (median 0.30×) | 121, 128, 126 |
| ppo-env512-bineps | 4.10, 5.12, 3.07M | 128, 126, 128 |
| td3-utd16 | 1.02M ×3 (about 0.10×) | 127, 128, 128 |

Predictions: P1 (cadence helps) correct; P2 (the gripper floor adds nothing without a release step) correct; P3 (TD3 stalls on precise alignment, as on PlaceSphere) **wrong**; P4 (any stall is at alignment) wrong. The "precision frontier" explanation was a property of PlaceSphere's reward, not a task category. Execution notes: the first td3-utd16 seed-13 attempt crashed at env creation (SAPIEN CUDA error 700) before training. Its log and receipt are kept in `.runtime/sample-efficiency-x1-20261031/failed-trial-11-setup-cuda700/`, and the trial was rerun once. The rerun trained to completion, with every checkpoint evaluation present and a final of 128/128, but the process exited with code 1 at teardown with no traceback, so its receipt state is `failed`. It is counted as a completed training run, with this caveat.

## Overnight Nov 1–2 (owner-authorized exploration, extra $100)

### c1: reverse curriculum on PlaceSphere td3-utd16 — fails its bar; it moves the frontier, not the ceiling
See the [plan](sample-efficiency-plan-20261031.md) for the registration and table. Median T80: control 6.14M; demo curriculum 5.12M (0.83×); self curriculum 4.61M (0.75×). Bar ≤0.67×. Both curricula bring the jump to about 75% roughly 2× earlier (median first ≥64/128 at 2.05M and 2.56M vs 5.12M), and the demo-free self curriculum matches the demo one. All arms then plateau at 75–85%. Final stage counts: in every run 128/128 episodes end on the bin, released; all failures are the sphere still moving at the last step (`is_obj_static`). About 27 trainer minutes per run.

### h1: hard-task headroom probe (2 seeds; PPO 30.72M, td3-utd16 10.24M)
PegInsertionSide: PPO 0 and 2/128; td3-utd16 82 and 75, still rising. PushT: PPO 7 and 22 (unstable); td3-utd16 127 and 128. PullCubeTool: PPO 79 and 114; td3-utd16 127 and 128. PlugCharger: 0 for all. Trainer minutes: PPO 18–36, td3-utd16 34–42. Without any retuning, the StackCube lesson (high-update-ratio TD3) transfers to three harder tasks and is the only learner here that solves PushT and PullCubeTool reliably within the budget. PegInsertionSide is chosen as the next target.

### c2: reverse curriculum on StackCube — negative
Frozen scorer, public development episodes. Median T80: control 3.07M; rc-self 3.07M (1.0×); rc-demo 3.58M (1.17×). Finals 114–125 for all arms. The curriculum only helps where success is a rare frontier for a long stretch. It is not a category-level method on this evidence.

### p1: PegInsertionSide — the discount, not the curriculum
Research-side evaluation, 15.36M, seeds 11–13. Under td3-utd16 at γ 0.9, all three seeds are censored (finals 89, 87, 60). Per-env self curriculum: T80 in 1/3 seeds, so it fails. **γ 0.95: T80 9.73, 6.66, 7.17M; finals 121, 124, 126. Passes.** Every arm grasps and pre-aligns the peg. Under γ 0.9 the failures are completed alignments that never insert. Peg episodes last 100 steps, twice StackCube's 50, so an effective horizon of about 10 steps is too short to reach the insertion stage.

### r7: g95 replication on fresh seeds 21–25 — passes
T80 6.66, 4.61, 6.66, 7.17, 7.17M (median 6.66M); finals 125, 128, 127, 121, 118 (median 125/128 = 97.7%). About 60 trainer minutes per run on an L40S. For reference, PPO at γ 0.9 reached 0 and 2/128 at 30.72M (h1), and PPO at γ 0.95 reached 0/128 on all three seeds (s1). ManiSkill's own PegInsertionSide PPO example budgets 250M steps; that example is not run here.

### s1: attribution
- γ 0.97: median T80 7.17M, finals 125–127, the same as γ 0.95.
- PPO at γ 0.95: 0/128 throughout 30.72M on seeds 11–13. The prediction held: the gain needs the off-policy learner *and* a discount matched to the horizon.
- PlugCharger at γ 0.97: 0/128 on seeds 11–13. The prediction held. There is no success signal for the discount to propagate.

**Transferable lesson (candidate, one task so far):** scale γ with the episode horizon. StackCube (50 steps) is best at 0.9 and worse at 0.95 under PPO. PegInsertionSide (100 steps) needs at least 0.95. This should be tested as a registered prediction on another long-horizon task before it is called a rule.
