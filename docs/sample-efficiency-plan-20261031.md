# Sample-efficiency screen, batch 1: update cadence and observation normalization

October 31, 2026. Written before any batch-1 outcome was observed. Owner: Claude (autonomous research handover; user cap $100 Modal total for this push).

## Why these levers

The [handover diagnosis](experiment-ledger.md) found that post-Oct-28 work screened mechanism changes at budgets where the strong PPO control itself scores 0/128, so screens could not discriminate. The simplest PPO sample-efficiency levers were never run: the [env-count plan](stackcube-env-count-plan-20261028.md) was prepared and shelved, and the trainer has no observation normalization.

- **Update cadence.** 1024 envs × 50 steps gives 51,200 transitions per PPO iteration, each reused for 8 epochs × 32 minibatches. 512/256 envs keep epochs and minibatch count, giving 2×/4× more optimizer steps per transition and fresher data.
- **Observation normalization.** Running mean/variance over all collected observations, no clipping, frozen within each rollout and updated after it. At export the affine map is folded exactly into the first actor layer, so the frozen evaluator still receives the unchanged restricted 48→256→256→256→8 tanh MLP. [Tests](../tests/test_sample_efficiency_obs_norm.py) verify merged statistics, the disabled identity path, and that the exported artifact run by the trusted NumPy evaluator matches the training policy on raw observations (≤2e-5).

## Fixed protocol

- Source: [`research/sample-efficiency-20261031/ppo.py`](../research/sample-efficiency-20261031/ppo.py), a small diff from the strong control used in scratch-credit (`ppo-gae.py`): optional obs-norm, 1.024M checkpoint interval, per-iteration 64-lane trace dumps removed (I/O only). With obs-norm off the learning computation is unchanged.
- Recipe otherwise fixed: StackCube-v1, state obs, `pd_joint_delta_pos`, normalized dense reward, 50-step full horizon, no partial reset, gamma .9, GAE λ .9, entropy .003, LR 3e-4, 8 epochs, 32 minibatches, target KL .1, clip .2. Random initialization; no teacher, demos, or prior data.
- Arms: `control` (1024 envs), `env512`, `env256`, `obsnorm` (1024), `obsnorm-env256`. Training seeds 11, 12, 13 for every arm (fresh seeds, not used before).
- Ceilings: 15,360,000 transitions or 2,700 trainer seconds, whichever first. One L40S per trial.
- Scoring: separate image, read-only candidate mount, no network. Every 1.024M export scored on the same 128 public-development episodes, seeds 97,100,000–97,100,127 (disjoint from protected development 0–127 and confirmation 1,000,000+). No protected query in this batch.

## Metrics and decision rule

- Primary: per-seed **T80** = first checkpoint with ≥103/128 successes that is also followed by a checkpoint ≥103/128 (sustained crossing); censored if not observed. Same for T90 (≥116/128).
- Secondary: success AUC over the common observed grid; trainer seconds to T80 (wall-clock cost is reported, not traded away).
- An arm is **promising** if, versus `control`, it reaches T80 in at least as many seeds and its median T80 is ≤ 0.67× control's (censored counted as the ceiling), with no seed worse by >2 checkpoints. Promising arms get combined/stacked in batch 2 with fresh seeds. Final claims require fresh-seed replication and one protected confirmation of the final policy only.
- Every run is reported, including failures and censoring.

## Batch 2 (registered while batch 1 runs, before any batch-1 outcome)

**Diagnosis from existing traces.** Saved 64-lane rollouts from the three scratch-credit GAE controls (seeds 1–3, 10.24M) show every seed placing and holding cube A on B by about 3.7M transitions (1,000–2,000 of 3,200 traced states per checkpoint). In those placed-and-held states the gripper command stays at μ≈−1 with σ≈0.33 and P(a>0.2)≈0.000 for millions of transitions. Seed 1 drifts out from about 6M and releases about 8.6M; seeds 2–3 remain stuck at 10.24M. The gripper is an absolute position target −0.01+0.025(a+1) m, so for every a∈[−1,0.2] the fingers still press on the 4 cm cube (measured finger width .037 m), and the dense reward is exactly flat. The Gaussian policy-gradient signal for release then scales like the tail mass exp(−Δ²/2σ²) with Δ=1.2: about 1.6e-4 per step.

**Intervention.** Student-t(ν=2) exploration noise with the same location and learned scale. The deterministic exported action is unchanged and the PPO log-density is exact. Tail mass beyond 3.6 scale units rises to about 3.5e-2, decaying polynomially rather than exponentially, a generic remedy for flat dead zones rather than a release-specific rule. Arms: `t-gripper` (gripper dim only) and `t-all` (all 8 dims), each on the 1024-env control base and on the obsnorm-env256 base. Seeds 11–13, paired with the batch-1 arms at identical initialization. Same ceilings, scorer, episodes and decision rule as batch 1.

## Batch 3 (registered after batch 1–2 outcomes, before any batch-3 outcome)

**What batches 1–2 showed.** Batch 1: `env512` and `env256` meet the promising rule (median T80 8.19M vs control 14.34M, 0.57×); `obsnorm` (0.71×) and `obsnorm-env256` (0.79×) do not. Stage diagnostics on the 128 scored episodes confirm the release bottleneck in the fresh controls: every control seed places in 115–128/128 episodes by 4.1M yet releases in 0 until 8–14M. Batch 2: Student-t exploration is clearly harmful (9/12 runs never exceed 5/128; t-gripper/t-all on the control base are 0/128 at 15.36M in all six runs, placed but never released). Exported actors evaluated on placed states from old traces show why: Gaussian controls keep the pre-clip gripper mean at about −0.95, Student-t policies push it to −1.1…−2.0 (beyond the clip, at no cost), widening the dead zone. Heavy tails also cause mid-carry drops, so the policy defends the grasp by moving its location further into the flat region.

**Intervention.** The dead zone exists because the continuous gripper command has a continuum of physically identical values. A training-time Bernoulli open/close gripper (logit = the actor's gripper output; arm dims unchanged Gaussian) removes the continuum: there are two actions, and opening on a placed cube changes reward immediately. The exported restricted MLP is the same architecture with the gripper output row multiplied by 20, so the evaluator's clip returns approximately sign(logit), the Bernoulli mode (tested: [binary gripper tests](../tests/test_sample_efficiency_obs_norm.py)). Arms: `bin` (1024 envs), `env256-bin`, plus `env128` (Gaussian) to test whether finer cadence keeps helping. Seeds 11–13, paired with batch 1. `env128` may hit the 2,700 s trainer ceiling before 15.36M; censoring is reported. Same scorer, episodes, ceilings and decision rule; `env256-bin` is additionally compared against `env256`.

## Batch 4 (registered after batch 3 outcomes, before any batch-4 outcome)

**Batch 3 outcome.** Bernoulli gripper is negative: all six runs place (116–128/128) and essentially never release (max 1/128). `env128` reaches T80 at 8.19, 11.26, 8.19M (no better than env256) and hits the trainer ceiling at 10.9–12.4M. Exported binary actors on old-trace placed states have gripper logits of −19…−28 (p_open 1e-9…1e-13). The dead-zone account was incomplete: the general mechanism is unbounded certainty. Carry states punish opening and drive the gripper parameter without limit; placed states share the representation. For a Bernoulli the score gradient of a rare "open" is also proportional to p, so learning dies with exploration. A Gaussian's location gradient (a−μ)/σ² does not vanish, which is why Gaussian controls eventually escape.

**Intervention: bounded exploration mass with a live gradient.** (a) `gfloor`: Gaussian gripper σ = max(exp(logstd), 0.5) plus loss 1.0·relu(|μ_grip|−1)², so μ cannot drift beyond the clip for free; P(a>0.2 | μ=−1) ≈ 0.8% per step. (b) `bineps`: Bernoulli open probability 0.02 + 0.96·σ(z) plus 1e-3·z² so z stays where its gradient is alive. Exported mode unchanged in both. Arms on the control (1024) and env256 bases; seeds 11–13 paired; same rule, compared with `control` and `env256`.

## Batch 5 (registered after batch 4 outcomes, before any batch-5 outcome)

**Batch 4 outcome.** Median T80 (M): `gfloor` 10.24 (0.71×), `env256-gfloor` 8.19 (0.57×), `bineps` 8.19 (0.57×, all seeds 8.19–9.22, T90 9.22–10.24 in every seed, 1024-env wall-clock), `env256-bineps` 7.17 (0.50×; 0.875× of env256). `bineps` and `env256-bineps` meet the promising rule. Release still begins about 4M transitions after placement.

**Hypothesis.** A one-step random open re-closes before the cube is free; release needs several consecutive open commands, so per-step independent exploration pays ε^k for a k-step commitment. Make the training gripper policy persistent: with probability ρ repeat the previous gripper command, otherwise sample the ε-floored Bernoulli; the mixture log-prob is exact given the stored previous command, and the exported mode is unchanged. Arms on the 1024 base (cheapest wall-clock): `bineps05` (ε 0.05, separating "more mass" from persistence), `bineps-sticky50` (ε 0.02, ρ 0.5), `bineps-sticky80` (ε 0.02, ρ 0.8). Compared with `bineps` and `control`, same seeds and rule.

## Batch 6 (registered after batch 5 outcomes, before any batch-6 outcome)

**Batch 5 outcome.** No arm beats `bineps`: ε 0.05 delays placement; stickiness ρ 0.5 is within noise; ρ 0.8 plateaus below T90.

**Exact symmetry augmentation.** Rotating the scene by θ about the robot base's vertical axis equals adding θ to joint 1; joint-delta actions and the dense reward are invariant. [`symmetry.py`](../research/sample-efficiency-20261031/symmetry.py) maps observations exactly; a GPU-simulator probe ([`symmetry_probe.py`](../deploy/modal/symmetry_probe.py), public reset seed 97,200,000) set rotated copies of 64 random-walk states and found static observation error ≤5e-6 in every dimension, and reward differences ≤2.3e-3 over 10 subsequent steps under identical actions (observation differences grow to O(0.1–1) in some steps, consistent with contact-chaos amplification of float differences). Each PPO minibatch adds a rotated copy (θ ~ U(−θmax, θmax) per sample) with importance ratio π_new(a|Rs)/π_old(a|s) (the rotated real trajectory's behavior policy), PPO clipping plus dual-clip 3 for negative advantages, and critic regression of V(Rs) on the same return. Arms (1024 envs, raw observations): `sym` (θmax 0.4, control base), `bineps-sym` (0.4), `bineps-sym08` (0.8). Seeds 11–13, same rule, compared with `control` and `bineps`.

## Batch 7 (registered after batch 6 outcomes, before any batch-7 outcome)

**Batch 6 outcome.** Symmetry augmentation speeds placement (control base places 42–107/128 at 3.1M vs 0–125 at 4.1M for control) and improves the control-base median T80 to 11.26M (0.79×, not promising); on the bineps base it gives median 9.22M (vs bineps 8.19M), with higher final success (120–126). Null for the T80 criterion.

**Precision clock.** Logged exploration std shows release beginning once arm dims 0–3 reach about σ≈0.2/0.13, in every arm checked: control reaches that σ by 5M but releases at 8–10M (gripper exploration), bineps reaches it about 6M and releases at 6–7M, env256-bineps reaches it by 2–3M and releases about 5M. With state-independent σ the stochastic placement is not accurate enough for release to pay until σ has decayed. Arms on the bineps base: `bineps-std37` (initial logstd −1.0, σ 0.37), `bineps-std22` (−1.5, σ 0.22), `bineps-ent0` (entropy coefficient 0). Note: the trainer's schedule previously overwrote ent_coef with the constant .003; it now keeps the CLI value (launcher default .003, so earlier arms are unchanged).

## Batch 7 outcome, batch 8 and replication r1 (registered before their outcomes)

**Batch 7 outcome.** Lower initial σ is harmful (σ 0.37: median T80 12.29M; σ 0.22: never places through 15.36M); entropy 0: 8.19, 8.19, 8.19 (same as bineps). Arm σ decay is a consequence of learning to place, not an independent clock. (One bineps-ent0 scorer call failed with FileNotFoundError after training completed; it was re-scored with `score_only`, unchanged artifacts.)

**Batch 8 (exploratory).** `env512-bineps`, seeds 11–13.

**Replication r1.** Fresh training seeds 21–25 (never used), arms `control`, `bineps`, `env256-bineps`, otherwise identical protocol, scorer and public-development episodes. Replication succeeds for an arm if, on seeds 21–25 alone, it reaches T80 in at least as many seeds as the fresh control and its median T80 (censored = 15.36M ceiling) is ≤ 0.67× the fresh control's. Pooled seeds 11–13 + 21–25 are reported too. No protected development or confirmation seed is used; a protected confirmation of a fixed final policy requires root-reviewed admission per the existing confirmation protocol and is not run by the research agent.

## Replication r2 (registered after r1 and batch 8 outcomes, before any r2 outcome)

**r1 outcome.** Fresh control (seeds 21–25) median T80 10.24M (one censored). `bineps` 8.19M (0.80×) and `env256-bineps` 7.17M (0.70×): both fail the preregistered 0.67× replication bar, although each beats the paired control on all five seeds. **Batch 8:** `env512-bineps` (seeds 11–13) T80 6.14, 6.14, 5.12M.

**r2.** `env512-bineps` on seeds 21–25, compared with the already-observed r1 fresh control under the same rule: success if it reaches T80 in ≥4 seeds (control reached it in 4) and its median T80 ≤ 0.67 × 10.24M = 6.86M (in practice ≤6.14M on the checkpoint grid).

## Off-policy pilot o1 (registered after r2, before any o1 outcome; user approved a larger budget, working cap about $150)

**r2 outcome.** `env512-bineps` on fresh seeds: T80 7.17, 8.19, 5.12, 7.17, 7.17M, median 0.70× of the fresh control; fails the 0.67× bar.

**Why off-policy.** The earlier adapted SAC run placed in 55/128 episodes by 256k transitions (about 10× fewer than PPO) but never released, consistent with the unbounded-certainty mechanism, and ran at about 200 transitions/s. PPO discards each sample after one rollout. [`td3.py`](../research/sample-efficiency-20261031/td3.py): TD3 on 1,024 GPU-parallel envs, 3-step returns (truncated at the fixed 50-step episode end, always bootstrapped since terminations are ignored as in the PPO recipe), twin LayerNorm critics (3×512), 4 updates of 16,384 replay rows per vector step, per-env exploration std U(0.05, 0.4) redrawn each episode, actor = the exported restricted MLP with linear output and a relu(|μ|−1)² barrier. Exploration bursts (behavior only; TD3 needs no importance correction) hold the gripper fully open for 3–10 steps, starting with probability 0.005 per step. Optional exact symmetry rotates each replay pair (s, s′) together. Smoke (1.024M, seed 11, td3-burst-sym at burst 0.02): 104 s, grasp 127/128, lift 13/128 at 1.02M; a first smoke attempt failed in the launcher (missing config keys) before training and its output was replaced.

**Arms and gate.** `td3`, `td3-burst`, `td3-burst-sym`; seeds 11, 12; 5.12M transitions, export every 512k, same scorer and public-development episodes. **Go** if any arm has both seeds at ≥103/128 by 3.07M (vs PPO best arm ≥5.12M); otherwise the off-policy track stops after reporting.

## o1 outcome and replication r3 (registered before any r3 outcome)

**o1 outcome: the preregistered go gate fails** (no arm has both seeds ≥103/128 by 3.07M). Sustained T80: `td3` 4.61M (seed 11), 4.10M (seed 12); `td3-burst` 4.61M, censored at 5.12M; `td3-burst-sym` censored, 4.10M. Plain TD3 releases without bursts. The gate's 3.07M threshold was an arbitrary choice made for the spending decision; it is not changed after the fact.

**r3 (new confirmatory test, fresh data).** Plain `td3` (identical code and hyperparameters as o1) on fresh seeds 21–25, 7.68M transitions, exports every 512k. Compared with the r1 fresh PPO control on the same seeds (median T80 10.24M; reached in 4/5). Success if `td3` reaches sustained T80 in ≥4/5 seeds and its median T80 ≤ 0.5 × 10.24M = 5.12M. Also reported against `env512-bineps` (r2 median 7.17M). Sustained crossing uses the 512k grid (next checkpoint also ≥103), which is finer than PPO's 1.024M grid; both grids are reported.

## r3 outcome and screen o2 (registered before any o2 outcome)

**r3 outcome: fails the 0.5× bar.** `td3` on seeds 21–25: sustained T80 (512k grid) 5.12, 6.14, 6.66, 4.10, 5.63M, median 5.63M = 0.55× of the fresh PPO control (1.024M grid: median 6.14M, 0.60×; one seed censored at 7.68M). T90 median 5.63M vs control 12.29M (0.46×).

**o2.** Update-to-data ratio: `td3-utd8` and `td3-utd16` (8 / 16 updates of 16,384 rows per vector step vs 4), seeds 11–12, 5.12M, paired with o1 `td3` (T80 4.61, 4.10M). Screen rule: promising if both seeds reach T80 at least 1 checkpoint (512k) earlier than the paired o1 seed; promising arms go to a fresh-seed replication against r3 `td3`.

## Transfer tests t1 (PlaceSphere-v1) and t2 (PickCube-v1) (registered before any outcome)

User instruction (Oct 31 night): total cap $200; test whether the learnings transfer to other tasks; keep final success in the 90s.

**Question.** Does the unbounded-certainty fix (`bineps`, with 512 envs) and/or off-policy TD3 transfer? PlaceSphere-v1 requires release (success = sphere in bin, static, not grasped), the same mechanism as StackCube; PickCube-v1 requires holding at a goal (success = cube at goal, robot static), where an exploration floor on opening could hurt.

**Protocol.** Same recipes as StackCube without retuning: `ppo-control` (1,024 envs, γ .9, entropy .003, full horizon), `ppo-env512-bineps`, `td3`; seeds 11–13; PPO 15.36M (evaluation every 1.024M), TD3 7.68M (every 512k). The frozen evaluator covers only StackCube, so these use **research-side** deterministic evaluation inside the trainer: ManiSkill's `success` at the end of the 50-step episode on 128 fixed reset seeds 300,000,000–300,000,127, identical across arms, with the deterministic action (mean; binary gripper mode). Metrics: sustained T80/T90 as before, and final success. Transfer claim for an arm: on a task, median T80 ≤ 0.67× control's with median final success ≥ 90% (≥116/128) and not below control's final by more than 5/128.

## o2 outcome and replication r4 (registered before any r4 outcome)

**o2 outcome.** `td3-utd8` T80 3.07, 4.10M (earlier than o1 on seed 11 only: not promising). `td3-utd16` T80 3.58, 2.05M; T90 3.58, 3.07M; final 122, 124 (earlier on both seeds: promising). Trainer ≈1,010 s for 5.12M vs ≈390 s at 4 updates.

**r4.** `td3-utd16` on fresh seeds 21–25, 7.68M, exports every 512k. Primary (the original off-policy bar): T80 reached in ≥4/5 seeds, median T80 ≤ 0.5 × 10.24M = 5.12M versus the r1 fresh PPO control on the same seeds, and median final success ≥116/128. Also reported against r3 `td3` and on the 1.024M grid.

## t1/t2 and r4 outcomes; t3/t4 and o3 (registered before their outcomes)

**r4 passes the preregistered bar.** `td3-utd16`, seeds 21–25: T80 4.61, 3.58, 3.58, 3.07, 3.58M (median 3.58M = 0.35× of the fresh PPO control; 1.024M grid 4.10M = 0.40×), T90 median 4.10M (control 12.29M), final 121–127/128, 5/5 seeds.

**Transfer outcomes (research-side evaluation).** PlaceSphere: `ppo-env512-bineps` median T80 4.10M vs control 10.24M (0.40×), but final success decays late in every arm (control 83/112/101, bineps 98/103/108, td3 118/109/109); no arm meets the ≥116 final requirement. PickCube: control 4.10M, bineps 4.10M (no harm; finals 128×3), td3 3.07M (finals 128×3).

**t3/t4.** `td3-utd16` on PlaceSphere-v1 and PickCube-v1, seeds 11–13, 7.68M, same research-side evaluation; compared with each task's t1/t2 arms under the transfer rule (median T80 ≤ 0.67× control, median final ≥116, not below control final by >5).

**o3.** `td3-utd32`, seeds 11–12, 5.12M (StackCube, frozen scorer); promising if both seeds reach T80 at least one 512k checkpoint earlier than the paired o2 `td3-utd16` seed (3.58, 2.05M).

## t3/t4/o3 outcomes; t5 (registered before any t5 outcome)

**Outcomes.** PickCube `td3-utd16`: T80 2.05, 3.07, 2.05M (median 0.50× of PPO control 4.10M), finals 128, 127, 128 — meets the transfer rule. PlaceSphere `td3-utd16`: T80 censored, 4.61M, censored; finals 104, 113, 99 — does not transfer. StackCube `td3-utd32`: T80 2.56, 3.07M vs utd16 3.58, 2.05M — not promising.

**t5.** On PlaceSphere every PPO arm peaks at 116–125/128 and then decays to 83–112 by 15.36M. Hypothesis: constant learning rate. Arms `ppo-control-anneal` and `ppo-env512-bineps-anneal` (linear LR decay to 0 over 15.36M; otherwise identical to t1), seeds 11–13. Compared with t1 arms: the decay is fixed if median final ≥116; the release-fix transfer claim is then re-evaluated with the transfer rule on the annealed pair.

**t6 (registered before outcome).** Third transfer task LiftPegUpright-v1 (reorient a peg upright; no release): `ppo-control`, `ppo-env512-bineps`, `td3-utd16`, seeds 11–13, same research-side evaluation and transfer rule.

## t5/t6 outcomes; replications r5 (PlaceSphere) and r6 (PickCube) (registered before their outcomes)

**t5.** PlaceSphere with linear LR decay: `ppo-control-anneal` T80 8.19, 9.22, 14.34M, finals 110, 102, 125; `ppo-env512-bineps-anneal` T80 4.10 ×3 (0.44×), finals 121, 124, 125 — meets the transfer rule. **t6.** LiftPegUpright: control T80 5.12 ×3 (finals 127, 128, 127); `ppo-env512-bineps` 3.07 ×3 (0.60×; finals 128 ×3); `td3-utd16` 3.07, 3.58, 2.56 (0.60×; finals 126, 126, 128). On this no-release task the bineps gain is presumably the 512-env cadence, not the gripper floor (not separated).

**r5.** PlaceSphere `ppo-control-anneal` vs `ppo-env512-bineps-anneal`, fresh seeds 21–25. **r6.** PickCube `ppo-control` vs `td3-utd16`, fresh seeds 21–25. Success for the treatment in each: T80 in ≥ as many seeds as its control, median T80 ≤ 0.67× control (censored at the run length), median final ≥116/128 and not below control's median final by >5.

## Nov 1 follow-up (registered before outcomes)

**d1 — why td3-utd16 fails on PlaceSphere.** Rerun seeds 11–13 for 7.68M with stage counts at every 512k checkpoint: episodes ever grasped, ever on the bin (5 mm xy and z tolerance), ever static, ever successful, plus 10/50/90% quantiles of the closest object-to-bin xy distance. Reading: if on-bin is frequent (≥64/128) while success stays near 0, the bottleneck is release/settling. If on-bin stays rare and the median closest distance sits above 5 mm, it is placement precision. If grasp itself is rare, it is earlier. The fix tried next depends on this reading; there is no pass/fail bar for d1.

**a1 — separate the bineps gripper from the 512-env cadence on LiftPegUpright.** Arms `ppo-env512` and `ppo-bineps` (1024 envs), seeds 11–13, 15.36M, compared against t6 (`ppo-control` 5.12M ×3 and `ppo-env512-bineps` 3.07M ×3). Attribution: a component explains the gain if its median T80 ≤4.10M (at least one 1.024M checkpoint earlier than control) with every seed ≤ control's.

**o4 — prioritized replay for td3-utd16 (StackCube, seeds 11–13, 5.12M, 512k exports).** Rationale: in r4 the stage counts show two discovery waits (transport→placement takes about 1.5M, placement→release about 1M). At each frontier the newly reached transitions are about 1% of a 2.1M-transition buffer, so uniform replay spends most critic updates on stages already learned. Proportional PER (α 0.6, β 0.4→1, new transitions at max priority) moves updates to high-Bellman-error transitions. Arms: `td3-utd16` (paired control, current code) and `td3-utd16-per`. Promising if `per` median T80 ≤0.8× the paired control, no seed later than its paired control by more than one checkpoint, and median final ≥116. Promising → fresh-seed replication on 21–25 against r4 (bar: median T80 ≤2.56M, i.e. ≤0.25× the PPO control's 10.24M, with finals ≥116).

**d1 reading (recorded before d2).** The bottleneck is placement precision, not release: for 4–6M transitions the median closest object–bin xy distance is 8–24 mm and on-bin (5 mm tolerance) is ever reached in 0–18/128 episodes; release follows within one checkpoint once the median closest distance falls below about 2 mm. Shaping between 10 mm and 5 mm changes the dense reward by about 0.025/13, so on-bin transitions form a rare frontier event.

**d2 — prioritized replay on PlaceSphere.** `td3-utd16-per` (α 0.6), seeds 11–13, 7.68M, stage diagnostics. Prediction: PER fixes the precision frontier. Pass: T80 in ≥2/3 seeds, median T80 ≤6.14M (≤0.67× the anneal control's 9.22M in t5), median final ≥116.
