# AutoManip experiment ledger

Updated 2026-10-29. **72 completed scientific training runs across 40 experiment families.** This comprises 71 RL runs and 1 behavior-cloning fits; collection, scoring and smokes are separate.
The current fixed 50M StackCube recipe scores **98.70% development and 95.90% independent confirmation** across three training seeds. The research objective is now at least **10× fewer interactions to 80–90% success** against strong matched baselines, with total compute and assistance accounted for. The user authorized continuous investigation until asked to stop and explicitly resumed bounded Modal work on Oct29, with cheap mechanism gates before bounded robot work. The completed [first-update mechanism probe](first-update-probe-results-20261028.md), [scientific diagnosis](research-strategy-diagnosis-20261028.md) and [37-paper literature map](literature/survey-20261028.md) remain the research context.

Training time below is the measured trainer process per seed, including startup and built-in diagnostics. Independent scoring, capacity queues, image setup and orchestration are separate. A mean across three seeds is not a claim that one run took three times that long. All tasks use state observations; no vision training has run.

## Completed training experiments

Development uses 128 fixed episodes per policy; confirmation uses 512 independent episodes. A dash means no confirmation was queried. Each name links to its recorded results. Rows with three seeds show mean trainer time and mean success; exact seed records follow below.

| Task | Experiment and change | Seeds | GPU | Mean minutes per seed | New steps per seed | Development | Confirmation |
|---|---|---:|---|---:|---|---:|---:|
| PickCube | [PickCube adapted PPO](../runs/runpod-20261025/summary.json) — State PPO, 600-second budget; stock pick metric | 3 | L4 | 10.00 | 5.622–5.854M | 98.70% | — |
| StackCube | [StackCube official PPO](../runs/stackcube-20261026/summary.json) — Unmodified official PPO; discount .8, entropy 0 | 3 | L4 | 23.64 | 24.986M | 31.51% | 32.23% |
| StackCube | [gamma090](../runs/stackcube-research-20261027/summary.json) — Discount .8 → .9 | 3 | L4 | 22.23 | 24.986M | 71.09% | — |
| StackCube | [gamma095](../runs/stackcube-research-20261027/summary.json) — Discount .8 → .95 | 3 | L4 | 20.96 | 24.986M | 34.90% | — |
| StackCube | [tune-gamma0875-control](../runs/stackcube-research-20261027/overnight/tune/summary.json) — Discount .875; entropy 0 | 3 | L4 | 21.79 | 24.986M | 70.57% | — |
| StackCube | [tune-gamma0925-control](../runs/stackcube-research-20261027/overnight/tune/summary.json) — Discount .925; entropy 0 | 3 | L4 | 21.32 | 24.986M | 81.25% | — |
| StackCube | [tune-entropy001-control](../runs/stackcube-research-20261027/overnight/tune/summary.json) — Discount .9; entropy .001 | 3 | L4 | 21.88 | 24.986M | 65.10% | — |
| StackCube | [tune-entropy003-control](../runs/stackcube-research-20261027/overnight/tune/summary.json) — Discount .9; entropy .003 | 3 | L4 | 22.44 | 24.986M | 83.59% | 82.10% |
| StackCube | [short-entropy003-control](../runs/stackcube-research-20261027/overnight/short/summary.json) — Selected recipe, half the transition budget | 3 | L4 | 11.51 | 12.493M | 12.24% | — |
| StackCube | [short-entropy003-lr2](../runs/stackcube-research-20261027/overnight/short/summary.json) — Half budget; double learning rate to .0006 | 3 | L4 | 10.68 | 12.493M | 4.69% | — |
| StackCube | [short-entropy003-epochs4](../runs/stackcube-research-20261027/overnight/short/summary.json) — Half budget; 4 PPO epochs instead of 8 | 3 | L4 | 10.34 | 12.493M | 2.86% | — |
| StackCube | [settle-timed-control](../.runtime/settling-20261027-results.tar.gz) — Fresh .9/.003 control under 22-minute ceiling | 3 | L4 | 21.46 | 24.986M | 83.59% | — |
| StackCube | [settle-full-horizon](../.runtime/settling-20261027-results.tar.gz) — Keep training episodes to horizon after success | 3 | L4 | 17.82 | 24.986M | 94.01% | 94.08% |
| StackCube | [settle-entropy-decay](../.runtime/settling-20261027-results.tar.gz) — Entropy .003 → 0 after 15M; original success resets | 3 | L4 | 21.38 | 24.986M | 71.35% | — |
| StackCube | [settle-gamma0925-entropy003](../.runtime/settling-20261027-results.tar.gz) — Discount .925 plus entropy .003 | 3 | L4 | 20.60 | 24.986M | 61.20% | — |
| StackCube | [full-horizon-control](../runs/stackcube-capg-screen-20261027/summary.json) — Matched L40S full-horizon PPO control | 1 | L40S | 19.20 | 24.986M | 94.53% | — |
| StackCube | [full-horizon-capg](../runs/stackcube-capg-screen-20261027/summary.json) — Boundary-probability likelihood for clipped actions | 1 | L40S | 19.29 | 24.986M | 92.19% | — |
| StackCube | [precision-constant](../runs/stackcube-precision-screen-20261027/summary.json) — Full-horizon PPO, twice the transition budget | 3 | L40S | 36.85 | 49.971M | 98.70% | 95.90% |
| StackCube | [precision-entropy-anneal](../runs/stackcube-precision-screen-20261027/summary.json) — 50M; reduce entropy to zero after 25.6M | 1 | L40S | 36.22 | 49.971M | 94.53% | — |
| StackCube | [precision-lr-anneal](../runs/stackcube-precision-screen-20261027/summary.json) — 50M; reduce LR to .00003 after 25.6M | 1 | L40S | 41.11 | 49.971M | 99.22% | — |
| StackCube | [risk-ppo-control](../runs/stackcube-risk-continuation-20261028/summary.json) — Continue own 25M policy using ordinary PPO | 1 | L40S | 1.43 | 2.048M new + 24.986M prefix | 96.09% | — |
| StackCube | [risk-episode-mean](../runs/stackcube-risk-continuation-20261028/summary.json) — Same continuation; episode-mean score gradient | 1 | L40S | 2.44 | 2.048M new + 24.986M prefix | 81.25% | — |
| StackCube | [risk-episode-cvar10](../runs/stackcube-risk-continuation-20261028/summary.json) — Same continuation; lower-tail CVaR10 score gradient | 1 | L40S | 1.60 | 2.048M new + 24.986M prefix | 71.09% | — |
| StackCube | [SAC adapted](../runs/stackcube-sac-screen-seed1-20261027/summary.json) — Replay-based SAC with safe-schema-compatible actor | 1 | L40S | 21.54 | 0.262M | 0.00% | — |
| StackCube | [BC-only random student](../runs/stackcube-bc-exploration-pilot-20261028/summary.json) — Imitate own teacher demonstrations; no student RL | 1 | L40S | 0.18 | 0 RL; 25.6k demos; assisted* | 38.28% | — |
| StackCube | [BC plus standard PPO](../runs/stackcube-bc-exploration-pilot-20261028/summary.json) — Shared clone; ordinary PPO exploration | 1 | L40S | 5.13 | 5.120M RL; assisted* | 0.00% | — |
| StackCube | [BC plus residual exploration PPO](../runs/stackcube-bc-exploration-pilot-20261028/summary.json) — Same clone; initial noise from training-demo errors | 1 | L40S | 3.49 | 5.120M RL; assisted* | 0.00% | — |
| StackCube | [scratch-kl-stop-control](../runs/stackcube-kl-value-screen-20261028/reconciled-summary.json) — Fresh strong PPO control; original joint KL stop | 1 | L40S | 9.54 | 10.240M | 76.56% | — |
| StackCube | [scratch-kl-stop-critic](../runs/stackcube-kl-value-screen-20261028/reconciled-summary.json) — Continue critic only after actor KL stop | 1 | L40S | 9.39 | 10.240M | 22.66% | — |
| StackCube | [replay-target-sac](../runs/stackcube-replay-moments-screen-20261028/trusted-summary.json) — Adapted target-network replay comparator; matched actor and update cadence | 1 | L40S | 9.77 | 0.875M | 0.00% | — |
| StackCube | [replay-joint-moments](../runs/stackcube-replay-moments-screen-20261028/trusted-summary.json) — Target-free critic; joint current/next normalization moments | 1 | L40S | 9.77 | 0.597M | 0.00% | — |
| StackCube | [replay-split-moments](../runs/stackcube-replay-moments-screen-20261028/trusted-summary.json) — Same target-free critic; split current/next moments, pooled EMA clock | 1 | L40S | 9.78 | 0.410M | 0.00% | — |
| StackCube | [replay-gamma090-sufficiency](../runs/gamma-sufficiency-20261028/trusted-summary.json) — Fast target-SAC gamma .99 → .9 only; smaller early-stage diagnostic budget | 1 | L40S | 2.52 | 0.256M | 0.00% | — |
| StackCube | [replay-initial-alpha005-sufficiency](../runs/temperature-sufficiency-20261028/trusted-summary.json) — Fast gamma .99 target-SAC initial alpha1 → .005 only | 1 | L40S | 3.26 | 0.256M | 0.00% | — |
| StackCube | [entropy-route-high-actor-low-backup](../runs/entropy-route-20261029/reconciled-summary.json) — Original high actor/temperature alpha; scale only soft backup entropy by .005 | 1 | L40S | 1.25 | 0.102M | 0.00% | — |
| StackCube | [entropy-route-low-actor-high-backup](../runs/entropy-route-20261029/reconciled-summary.json) — Original low actor/temperature alpha; scale only soft backup entropy by200 | 1 | L40S | 0.98 | 0.102M | 0.00% | — |
| StackCube | [behavior-joint-uniform](../runs/joint-exploration-20261029/reconciled-summary.json) — Original low-alpha SAC learner;10% joint-vector uniform behavior mixture | 1 | L40S | 1.41 | 0.102M | 0.00% | — |
| StackCube | [behavior-coordinate-uniform](../runs/joint-exploration-20261029/reconciled-summary.json) — Same learner and conditional coordinate marginals;10% independent-coordinate uniform mixture | 1 | L40S | 1.49 | 0.102M | 0.00% | — |
| StackCube | [full-task-baseline-SAC](../runs/full-stacking-curve-20261029/reconciled-summary.json) — Fresh seed2 ordinary low-alpha SAC; full-task curve,600s/1.024M dual ceilings | 1 | L40S | 9.78 | 0.784M | 0.00% | — |
| StackCube | [full-task-coordinate-SAC](../runs/full-stacking-curve-20261029/reconciled-summary.json) — Same fresh seed2 learner;10% coordinate mixture; full-task curve, same dual ceilings | 1 | L40S | 9.78 | 0.801M | 0.00% | — |

## What changed performance

The main measured recipe gains followed changes to discounting, entropy, training through the full horizon after success, and finally more training. These interventions do not by themselves identify the causal learning mechanism. The full-horizon 25M recipe confirmed at **94.08%** in roughly **17.8 minutes per seed** on L4. Doubling transitions to 50M confirmed at **95.90%**, with roughly **36.8 minutes per seed** on L40S. Hardware and budgets differ; this is not a controlled wall-time speedup comparison.

The 99.22% headline was one development seed. The three-seed development mean is 98.70%; confirmation is 95.90%. The 50M recipe missed its preregistered 99% confirmation target. Existing confirmation sets are independent of development but are historical fixed sets, not a private final holdout.

The half-budget recipes lost substantial performance. CAPG failed its learning-curve gate despite a mixed late-checkpoint signal. The tested SAC recipe learned grasp/transport but scored zero final stacks within its cap; this does not refute SAC generally. Entropy and LR refinement did not beat the simpler 50M constant recipe. Episode-risk continuation lost to ordinary PPO and was stopped.

The three risk rows are **continuations**, not fresh short runs: each starts from our own 24,985,600-step, 1,088.77-second teacher and restarts its optimizer. Times and transitions in those rows show only additional work; their shared prefix is disclosed and is already counted once in the project training history.

PickCube uses the stock pick metric, which does not require release, and a different controller from StackCube. Its 10-minute result is not an earliest-time-to-success measurement. StackCube uses the official frozen final-step stacking metric; ten-step stability is a separate diagnostic.

Assisted* rows share the **49,971,200-transition / 2,295.41-second teacher**, plus **25,600 demo transitions**, **29.37 seconds of generation** and **5,000 BC updates / 10.87 trainer seconds**. Both PPO variants then used 5.12M new transitions and fell from BC-only49/128 to0/128. Residual-noise initialization failed its gate and is stopped. Shared assistance is disclosed rather than counted as free or as a from-scratch gain.

## Current research review and prepared work

The separately approved [first-update mechanism probe](first-update-probe-results-20261028.md) is complete and stopped: 201,600 new transitions, 46.66 seconds on one L40S, six explicitly counted one-step optimizer calls. Large mean steps damaged the assisted policy; small-step improvement direction remains unproven, so the full hypothesis gate did not pass. This is one controlled probe, not additional independently trained policies or a benchmark comparison. The later user-authorized [replay-moment study](replay-moment-study-plan-20261028.md) completed three fresh runs:1,881,024 total transitions,14 once-only development scores, all0/128, no confirmation query or replication gate passed. Different throughput censored final step counts, so shared51.2k/102.4k/256k checkpoints are the equal-step comparison. All three apps for its smoke, screen and separate public video were verified stopped with zero tasks. The two earlier critic/control records were reconciled without rerunning training. Prepared payloads below remain disabled.

| Work | Change and budget | Status |
|---|---|---|
| [BC exploration pilot](stackcube-bc-exploration-plan-20261028.md) | Fresh GPU teacher demonstrations, one random-student BC fit, then matched standard BC+PPO versus residual-calibrated initial action noise; at most 5.12M new RL steps / 600 seconds each | Complete: BC49/128, both PPO variants0/128; failed gate, stopped |
| AMP motion prior | Add adversarial transition-style reward to BC+PPO | Deferred after local temporal/OOD diagnostics; no GPU AMP training run |
| [From-scratch critic completion](stackcube-kl-value-plan-20261028.md) | Stop policy at KL limit, continue value fitting on the same data; 10.24M-step / 600s ceilings | Complete: control98/128 versus critic29/128 at10.24M; negative gate, stopped |
| From-scratch update cadence | Use 512 or 256 environments instead of 1024, giving more policy updates per collected transition | Prepared but disabled under research strategy change; not launched |
| Assisted mean learning rate | Scale policy-mean LR with the calibrated noise scale; critic/noise LR unchanged | Frozen proposal only; research strategy review required before any launch |

The BC experiment is an assisted track: it reuses our own 49,971,200-step / 2,295.41-second teacher. Teacher training, fresh demo generation, BC updates and student RL must be reported separately. It cannot establish a from-scratch sample-efficiency gain by hiding the teacher prefix.

## Mechanism probes and execution checks

These are excluded from the scientific GPU training count above. Toy success and action MSE are not StackCube benchmark success.

| Probe | Measured duration | Outcome |
|---|---:|---|
| [Assisted lift-control probe](lift-control-results-20261029.md) |11.47 s GPU child /15.75 s wrapper |512 public transitions; paired physical-state gate failed before any intervention. No grip/command result, fitting or protected score. Archived and stopped. |
| [Frozen actor/critic gradient snapshot](actor-gradient-snapshot-results-20261029.md) |.669 s CPU |1,904 existing grasp states,16 fixed draws each. Q favors further positive saturated-joint action; entropy opposes it. Zero new interactions/fitting; no critic-truth or efficiency claim. |
| [Coordinate entropy gate](coordinate-entropy-gate-results-20261029.md) |.061 s CPU |All eight hidden-coordinate gates fail; scalar conditional pressure mostly raises temperature. Analytic equal-entropy remedy costs5.05x in the declared quadratic negative control. No fitting/robot queries/method launch. |
| [Randomized assisted lift probe](randomized-lift-results-20261029.md) |13.17 s GPU diagnostic /17.37 s wrapper |6,400 public transitions. Servo+closure lifts20/21eligible versus4/19servo+policygripper; allassigned1/32,1/32,4/32,20/32. No training/protectedscore or learned-efficiency claim. Archived and stopped. |
| [Assisted first-update mechanism probe](first-update-probe-results-20261028.md) | 46.66 s GPU process | 201,600 total new transitions; six one-step optimizer calls /9,600 fitting rows. No-update75/192, full step38/192, bounded72/192; useful forward direction not established. No benchmark query or automatic follow-up. |
| [Replay moment operator gate](research-window-replay-review-20261028.md) | .0792 s CPU | Shifted-half aliasing and gradient coupling; finite-epsilon BRN caveat. Zero robot transitions or optimizer calls. |
| [Actual-critic scalar control diagnostic](research-window-replay-value-20261028.md) |17.3927 s CPU including measurement-only repeat |18,000 total optimizer calls on16,384 synthetic training rows; split normalization gives gross Q error despite small self-TD residual. Joint does not beat competent target control; not a robot result. |
| Replay execution smoke |40.28 s GPU wrapper |15,360 total transitions across three tiny execution trials; six safe artifacts validated, paired initialization, finite counters. No benchmark score; excluded from scientific run count. |
| Recursive credit / replay prototypes | See [discovery journal](research-discovery-20261027.md) and [neural gate](research-discovery-neural-gate-20261027.md) | CPU falsification; no robot deployment |
| Learned contraction | 10.50 s | Failed smooth-system control despite hybrid-toy gain; stopped |
| State-conditioned variance head | 58.91 s | Toy 88.67% versus global-variance 100%; stopped |
| Quantile lower-tail objective | 10.35 s | Toy gate passed; lower reward tradeoff; motivated the later failed robot risk pilot |
| Synthetic BC / discriminator component checks | 1.64 s | Dataflow gate passed across six seeds; no robot-performance claim |
| Local CPU teacher collection | 14.01 s | 256 complete episodes, 12,800 transitions, 247 successful / 9 failed, all retained |
| Local CPU BC | 11.77 s | 5,000 updates, held-out demo action MSE .01017 |
| Fresh local CPU rollout probe | 8.57 s across three policies | Random 0/64, BC 21/64, teacher 61/64; Mac SAPIEN differs from frozen GPU stack |
| Real-transition discriminator | .90 s | Weak temporal sensitivity; high style on unsuccessful OOD random policies; AMP deferred |
| Residual-noise calibration | .34 s | Held-out expected clipped-action MSE .26744 → .01613; no RL gain established; low noise can suppress corrections |

Simulator smokes, policy-converter parity, provider compatibility and isolation checks are execution validation, not additional successful research experiments. Preserved failures include a SAC replay dtype error and an SDK image-option error before confirmation submission. No failed run is converted into a benchmark success. Current task/evaluator/seeds/physics/interfaces are unchanged.

## Every completed training run

| Experiment | Seed | Trainer seconds | New transitions | Development | Confirmation |
|---|---:|---:|---:|---:|---:|
| PickCube adapted PPO | 1 | 600.043 | 5,853,696 | 125/128 | — |
| PickCube adapted PPO | 2 | 600.020 | 5,657,600 | 127/128 | — |
| PickCube adapted PPO | 3 | 600.043 | 5,621,760 | 127/128 | — |
| StackCube official PPO | 1 | 1319.652 | 24,985,600 | 30/128 | 115/512 |
| StackCube official PPO | 2 | 1455.519 | 24,985,600 | 61/128 | 244/512 |
| StackCube official PPO | 3 | 1480.039 | 24,985,600 | 30/128 | 136/512 |
| gamma090 | 1 | 1349.071 | 24,985,600 | 72/128 | — |
| gamma090 | 2 | 1299.144 | 24,985,600 | 100/128 | — |
| gamma090 | 3 | 1354.009 | 24,985,600 | 101/128 | — |
| gamma095 | 1 | 1238.621 | 24,985,600 | 10/128 | — |
| gamma095 | 2 | 1269.805 | 24,985,600 | 92/128 | — |
| gamma095 | 3 | 1264.059 | 24,985,600 | 32/128 | — |
| tune-gamma0875-control | 1 | 1299.804 | 24,985,600 | 81/128 | — |
| tune-gamma0875-control | 2 | 1273.879 | 24,985,600 | 94/128 | — |
| tune-gamma0875-control | 3 | 1349.285 | 24,985,600 | 96/128 | — |
| tune-gamma0925-control | 1 | 1304.534 | 24,985,600 | 114/128 | — |
| tune-gamma0925-control | 2 | 1264.008 | 24,985,600 | 105/128 | — |
| tune-gamma0925-control | 3 | 1269.062 | 24,985,600 | 93/128 | — |
| tune-entropy001-control | 1 | 1344.802 | 24,985,600 | 55/128 | — |
| tune-entropy001-control | 2 | 1289.206 | 24,985,600 | 87/128 | — |
| tune-entropy001-control | 3 | 1304.372 | 24,985,600 | 108/128 | — |
| tune-entropy003-control | 1 | 1369.621 | 24,985,600 | 114/128 | 465/512 |
| tune-entropy003-control | 2 | 1339.296 | 24,985,600 | 96/128 | 347/512 |
| tune-entropy003-control | 3 | 1329.912 | 24,985,600 | 111/128 | 449/512 |
| short-entropy003-control | 1 | 697.478 | 12,492,800 | 7/128 | — |
| short-entropy003-control | 2 | 692.635 | 12,492,800 | 2/128 | — |
| short-entropy003-control | 3 | 682.579 | 12,492,800 | 38/128 | — |
| short-entropy003-lr2 | 1 | 642.163 | 12,492,800 | 4/128 | — |
| short-entropy003-lr2 | 2 | 632.297 | 12,492,800 | 8/128 | — |
| short-entropy003-lr2 | 3 | 647.412 | 12,492,800 | 6/128 | — |
| short-entropy003-epochs4 | 1 | 617.643 | 12,492,800 | 0/128 | — |
| short-entropy003-epochs4 | 2 | 622.061 | 12,492,800 | 0/128 | — |
| short-entropy003-epochs4 | 3 | 622.373 | 12,492,800 | 11/128 | — |
| settle-timed-control | 1 | 1298.890 | 24,985,600 | 114/128 | — |
| settle-timed-control | 2 | 1279.896 | 24,985,600 | 96/128 | — |
| settle-timed-control | 3 | 1284.790 | 24,985,600 | 111/128 | — |
| settle-full-horizon | 1 | 1088.767 | 24,985,600 | 120/128 | 478/512 |
| settle-full-horizon | 2 | 1049.020 | 24,985,600 | 122/128 | 483/512 |
| settle-full-horizon | 3 | 1069.014 | 24,985,600 | 119/128 | 484/512 |
| settle-entropy-decay | 1 | 1304.092 | 24,985,600 | 104/128 | — |
| settle-entropy-decay | 2 | 1269.634 | 24,985,600 | 73/128 | — |
| settle-entropy-decay | 3 | 1274.339 | 24,985,600 | 97/128 | — |
| settle-gamma0925-entropy003 | 1 | 1218.897 | 24,985,600 | 10/128 | — |
| settle-gamma0925-entropy003 | 2 | 1235.393 | 24,985,600 | 103/128 | — |
| settle-gamma0925-entropy003 | 3 | 1254.240 | 24,985,600 | 122/128 | — |
| full-horizon-control | 1 | 1152.073 | 24,985,600 | 121/128 | — |
| full-horizon-capg | 1 | 1157.158 | 24,985,600 | 118/128 | — |
| precision-constant | 1 | 2295.408 | 49,971,200 | 127/128 | 497/512 |
| precision-constant | 2 | 2228.223 | 49,971,200 | 126/128 | 489/512 |
| precision-constant | 3 | 2108.697 | 49,971,200 | 126/128 | 487/512 |
| precision-entropy-anneal | 1 | 2172.979 | 49,971,200 | 121/128 | — |
| precision-lr-anneal | 1 | 2466.694 | 49,971,200 | 127/128 | — |
| risk-ppo-control | 1 | 85.566 | 2,048,000 | 123/128 | — |
| risk-episode-mean | 1 | 146.272 | 2,048,000 | 104/128 | — |
| risk-episode-cvar10 | 1 | 96.085 | 2,048,000 | 91/128 | — |
| SAC adapted | 1 | 1292.195 | 262,432 | 0/128 | — |
| BC-only random student | 1 | 10.868 | 0 | 49/128 | — |
| BC plus standard PPO | 1 | 307.649 | 5,120,000 | 0/128 | — |
| BC plus residual exploration PPO | 1 | 209.447 | 5,120,000 | 0/128 | — |
| scratch-kl-stop-control | 1 | 572.611 | 10,240,000 | 98/128 | — |
| scratch-kl-stop-critic | 1 | 563.380 | 10,240,000 | 29/128 | — |
| replay-target-sac | 1 | 586.449 | 874,880 | 0/128 | — |
| replay-joint-moments | 1 | 586.448 | 596,544 | 0/128 | — |
| replay-split-moments | 1 | 586.860 | 409,600 | 0/128 | — |
| replay-gamma090-sufficiency | 1 | 151.354 | 256,000 | 0/128 | — |
| replay-initial-alpha005-sufficiency | 1 | 195.434 | 256,000 | 0/128 | — |
| entropy-route-high-actor-low-backup | 1 | 75.286 | 102,400 | 0/128 | — |
| entropy-route-low-actor-high-backup | 1 | 58.754 | 102,400 | 0/128 | — |
| behavior-joint-uniform | 1 | 84.374 | 102,400 | 0/128 | — |
| behavior-coordinate-uniform | 1 | 89.241 | 102,400 | 0/128 | — |
| full-task-baseline-SAC | 2 | 586.806 | 783,936 | 0/128 | — |
| full-task-coordinate-SAC | 2 | 587.051 | 801,472 | 0/128 | — |

## Source records and updates

[Exact run data and source digests](experiment-runs.json). Rebuild with `python scripts/build_experiment_ledger.py` from this checkout and its retained local archives. The generator reads recorded aggregates only; it does not train, evaluate, deserialize native checkpoints or change cloud resources.

Detailed reports: [official StackCube baseline](stackcube-results-20261026.md), [early sweeps](stackcube-overnight-results-20261027.md), [full-horizon investigation](stackcube-settling-20261027.md), [CAPG](stackcube-capg-results-20261027.md), [SAC](stackcube-sac-results-20261027.md), [50M screen](stackcube-precision-results-20261027.md), [50M replication](stackcube-precision-replication-results-20261028.md), [local BC/prior checks](stackcube-amp-local-dataflow-20261027.md), and [residual exploration hypothesis](research-discovery-bc-residual-exploration-v2-20261028.md).
