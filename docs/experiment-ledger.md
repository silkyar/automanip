# AutoManip experiment ledger

Updated 2026-11-01. **266 completed scientific training runs across 92 experiment families.** This comprises 265 RL runs and 1 behavior-cloning fits; collection, scoring and smokes are separate.

The [Plan-to-Distill pilot](gpd-pilot-closeout-20261031.md) is now closed negative at 984,320 transitions. Final student-state aggregation scored 6/512, equal to original-state aggregation and below the unchanged student (12/512). Earlier, the [first readout test](readout-distillation-results-20261031.md) found successful training continuations for 89/128 states after a controller-coordinate correction, but the resulting fixed readout student scored only 34/1024 on fresh public episodes versus 1/1024 for the original weak actors and 14/1024 for an equal-physics unbranched-data student. The predeclared material screens failed; the historical 95.9% confirmation result is unchanged. Subsequent endpoint and behavioral-preservation probes are recorded separately below. CPU adaptation fits and all failed attempts are separate from the 84 historical GPU training runs. The [earlier waiting prototype](continuation-correction-results-20261030.md) also failed its strongest control. No from-scratch or 10x sample-efficiency improvement is established.

**Oct 31–Nov 1 sample-efficiency screen (Claude handover):** the `se-` rows below are from-scratch runs recorded in [sample-efficiency results](sample-efficiency-results-20261031.md) (plan: [preregistrations](sample-efficiency-plan-20261031.md); compact data: [runs](sample-efficiency-runs-20261031.json)). StackCube `se-` rows use fresh public-development seeds 97,100,000+ through the frozen scorer; non-StackCube rows use a research-side evaluation on seeds 300,000,000+ and are not frozen-evaluator results. Best StackCube arm `td3-utd16` passes the preregistered fresh-seed bar: median T80 3.58M vs PPO control 10.24M (0.35×), finals 121–127/128. The PPO release fix `env512-bineps` (with LR annealing) passes a fresh-seed PlaceSphere replication (0.50×), and `td3-utd16` passes a fresh-seed PickCube replication (0.50×). No protected seed was queried; a once-only [confirmation package](td3-utd16-confirmation-plan-20261101.md) is prepared but disabled pending root review.

The current fixed 50M StackCube recipe scores **98.70% development and 95.90% independent confirmation** across three training seeds. The research objective is now at least **10× fewer interactions to 80–90% success** against strong matched baselines, with total compute and assistance accounted for. The user authorized continuous investigation until asked to stop and explicitly resumed bounded Modal work on Oct29, with cheap mechanism gates before bounded robot work. The completed [first-update mechanism probe](first-update-probe-results-20261028.md), [scientific diagnosis](research-strategy-diagnosis-20261028.md) and [37-paper literature map](literature/survey-20261028.md) remain the research context.

Training time below is the measured trainer process per seed, including startup and built-in diagnostics. Independent scoring, capacity queues, image setup and orchestration are separate. A mean across three seeds is not a claim that one run took three times that long. All tasks use state observations; no vision training has run.

## Completed training experiments

Historical development uses128 fixed episodes per policy; confirmation uses512 independent episodes. The scratch-credit and feature-value rows explicitly use paired fresh public-development episodes and do not replace historical benchmark numbers. A dash means no confirmation was queried. Each name links to its records. Three-seed rows show mean trainer time and mean success; exact seed records follow below.

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
| StackCube | [scratch-credit-ppo-gae](../cumulative/results/scratch-credit-20261030.json) — Strong gamma.9/entropy.003/full-horizon PPO,600s/10.24M ceilings; fresh public-development curves | 3 | L40S | 9.39 | 10.240M | 30.21% | — |
| StackCube | [scratch-credit-ppo-finite-mc](../cumulative/results/scratch-credit-20261030.json) — Same PPO; replace complete GAE target by finite50-step MC; same ceilings, fresh public-development curves | 3 | L40S | 9.94 | 8.960–9.626M | 0.00% | — |
| StackCube | [feature-value-ppo-feature-value](../cumulative/results/feature-value-20261030.json) — FiniteMC PPO; cross-fitted linear value on current actor features; inactive neural critic;600s/10.24M ceilings | 3 | L40S | 9.59 | 9.677–10.240M | 0.00% | — |
| StackCube | [feature-value-ppo-clock-feature-value](../cumulative/results/feature-value-20261030.json) — Identical cross-fitted linear value with remaining-time feature scale; actor interface unchanged | 3 | L40S | 9.93 | 9.933–10.240M | 0.00% | — |
| StackCube | [gripper-prior-full](../cumulative/results/gripper-hold-20261031.json) — Original strong full256 PPO; fresh seed41,300s/4.096M ceilings | 1 | L40S | 4.17 | 4.096M | 0.00% | — |
| StackCube | [gripper-prior-masked](../cumulative/results/gripper-hold-20261031.json) — Same strong PPO; fixed isolated zero-output path and255-unit learned residual | 1 | L40S | 3.96 | 4.096M | 0.00% | — |
| StackCube | [gripper-prior-hold](../cumulative/results/gripper-hold-20261031.json) — Matched255-unit residual plus fixed analytic gripper hold mean; no pretrained weights or data | 1 | L40S | 3.55 | 4.096M | 0.00% | — |
| StackCube | [se-control](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 8 | L40S | 11.68 | 15.360M | 87.79% | — |
| StackCube | [se-env512](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 17.17 | 15.360M | 89.06% | — |
| StackCube | [se-env256](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 32.96 | 15.360M | 90.62% | — |
| StackCube | [se-obsnorm](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; observation normalization; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 10.13 | 15.360M | 92.45% | — |
| StackCube | [se-obsnorm-env256](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; observation normalization; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 31.29 | 15.360M | 92.71% | — |
| StackCube | [se-t-gripper](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; Student-t df 2 noise on gripper; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 12.00 | 15.360M | 0.00% | — |
| StackCube | [se-t-all](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; Student-t df 2 noise on all; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 12.53 | 15.360M | 0.00% | — |
| StackCube | [se-obsnorm-env256-t-gripper](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; Student-t df 2 noise on gripper; observation normalization; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 33.03 | 15.360M | 61.98% | — |
| StackCube | [se-obsnorm-env256-t-all](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; Student-t df 2 noise on all; observation normalization; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 35.55 | 15.360M | 1.04% | — |
| StackCube | [se-bin](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper, unfloored; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 11.56 | 15.360M | 0.00% | — |
| StackCube | [se-env256-bin](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; training-time Bernoulli gripper, unfloored; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 34.20 | 15.360M | 0.26% | — |
| StackCube | [se-env128](../docs/sample-efficiency-runs-20261031.json) — PPO, 128 envs; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 44.54 | 10.912–12.416M | 87.76% | — |
| StackCube | [se-gfloor](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; Gaussian gripper σ floor .5 and mean barrier; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 12.89 | 15.360M | 95.57% | — |
| StackCube | [se-env256-gfloor](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; Gaussian gripper σ floor .5 and mean barrier; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 34.70 | 15.360M | 91.15% | — |
| StackCube | [se-bineps](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 8 | L40S | 15.16 | 15.360M | 95.90% | — |
| StackCube | [se-env256-bineps](../docs/sample-efficiency-runs-20261031.json) — PPO, 256 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 8 | L40S | 33.94 | 15.360M | 91.80% | — |
| StackCube | [se-bineps05](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.05 floor and logit L2; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 13.66 | 15.360M | 97.66% | — |
| StackCube | [se-bineps-sticky50](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; sticky gripper 0.5; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 15.42 | 15.360M | 94.27% | — |
| StackCube | [se-bineps-sticky80](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; sticky gripper 0.8; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 14.77 | 15.360M | 78.65% | — |
| StackCube | [se-sym](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; base-yaw symmetry augmentation ±0.4; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 12.76 | 15.360M | 92.19% | — |
| StackCube | [se-bineps-sym](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; base-yaw symmetry augmentation ±0.4; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 22.85 | 15.360M | 96.88% | — |
| StackCube | [se-bineps-sym08](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; base-yaw symmetry augmentation ±0.8; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 20.35 | 15.360M | 96.09% | — |
| StackCube | [se-bineps-std37](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; initial log σ -1.0; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 16.16 | 15.360M | 89.58% | — |
| StackCube | [se-bineps-std22](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; initial log σ -1.5; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 14.34 | 15.360M | 0.00% | — |
| StackCube | [se-bineps-ent0](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; entropy 0.0; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 3 | L40S | 12.16 | 15.360M | 95.31% | — |
| StackCube | [se-env512-bineps](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 8 | L40S | 20.56 | 15.360M | 94.04% | — |
| StackCube | [se-td3](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 4 updates/step, n-step 3, LayerNorm twin critics; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 7 | L40S | 9.25 | 5.120–7.680M | 94.42% | — |
| StackCube | [se-td3-burst](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 4 updates/step, n-step 3, LayerNorm twin critics; gripper-open bursts p 0.005; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 2 | L40S | 7.12 | 5.120M | 95.31% | — |
| StackCube | [se-td3-burst-sym](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 4 updates/step, n-step 3, LayerNorm twin critics; gripper-open bursts p 0.005; base-yaw symmetry augmentation ±0.4; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 2 | L40S | 7.86 | 5.120M | 66.02% | — |
| StackCube | [se-td3-utd8](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 8 updates/step, n-step 3, LayerNorm twin critics; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 2 | L40S | 10.21 | 5.120M | 94.14% | — |
| StackCube | [se-td3-utd16](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 16 updates/step, n-step 3, LayerNorm twin critics; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 7 | L40S | 23.67 | 5.120–7.680M | 96.43% | — |
| StackCube | [se-td3-utd32](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 32 updates/step, n-step 3, LayerNorm twin critics; Oct 31 sample-efficiency screen, fresh public-development seeds 97,100,000+ | 2 | L40S | 31.45 | 5.120M | 93.75% | — |
| PlaceSphere | [se-ppo-control [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 14.18 | 15.360M | 77.08% | — |
| PlaceSphere | [se-ppo-env512-bineps [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 26.93 | 15.360M | 80.47% | — |
| PlaceSphere | [se-td3 [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 4 updates/step, n-step 3, LayerNorm twin critics; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 10.74 | 7.680M | 87.50% | — |
| PickCube | [se-ppo-control [PickCube]](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 8 | L40S | 9.16 | 15.360M | 98.44% | — |
| PickCube | [se-ppo-env512-bineps [PickCube]](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 19.28 | 15.360M | 100.00% | — |
| PickCube | [se-td3 [PickCube]](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 4 updates/step, n-step 3, LayerNorm twin critics; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 9.54 | 7.680M | 100.00% | — |
| PlaceSphere | [se-td3-utd16 [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 16 updates/step, n-step 3, LayerNorm twin critics; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 26.85 | 7.680M | 82.29% | — |
| PickCube | [se-td3-utd16 [PickCube]](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 16 updates/step, n-step 3, LayerNorm twin critics; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 8 | L40S | 26.09 | 7.680M | 99.80% | — |
| PlaceSphere | [se-ppo-control-anneal [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; linear LR decay; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 8 | L40S | 13.75 | 15.360M | 86.82% | — |
| PlaceSphere | [se-ppo-env512-bineps-anneal [PlaceSphere]](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; linear LR decay; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 8 | L40S | 26.94 | 15.360M | 96.00% | — |
| LiftPegUpright | [se-ppo-control [LiftPegUpright]](../docs/sample-efficiency-runs-20261031.json) — PPO, 1024 envs; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 10.65 | 15.360M | 99.48% | — |
| LiftPegUpright | [se-ppo-env512-bineps [LiftPegUpright]](../docs/sample-efficiency-runs-20261031.json) — PPO, 512 envs; training-time Bernoulli gripper with ε 0.02 floor and logit L2; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 19.42 | 15.360M | 100.00% | — |
| LiftPegUpright | [se-td3-utd16 [LiftPegUpright]](../docs/sample-efficiency-runs-20261031.json) — TD3, 1024 envs, 16 updates/step, n-step 3, LayerNorm twin critics; research-side transfer evaluation (128 seeds 300,000,000+, ManiSkill success), not the frozen evaluator | 3 | L40S | 26.09 | 7.680M | 98.96% | — |

## What changed performance

The main measured recipe gains followed changes to discounting, entropy, training through the full horizon after success, and finally more training. These interventions do not by themselves identify the causal learning mechanism. The full-horizon 25M recipe confirmed at **94.08%** in roughly **17.8 minutes per seed** on L4. Doubling transitions to 50M confirmed at **95.90%**, with roughly **36.8 minutes per seed** on L40S. Hardware and budgets differ; this is not a controlled wall-time speedup comparison.

The 99.22% headline was one development seed. The three-seed development mean is 98.70%; confirmation is 95.90%. The 50M recipe missed its preregistered 99% confirmation target. Existing confirmation sets are independent of development but are historical fixed sets, not a private final holdout.

The half-budget recipes lost substantial performance. CAPG failed its learning-curve gate despite a mixed late-checkpoint signal. The tested SAC recipe learned grasp/transport but scored zero final stacks within its cap; this does not refute SAC generally. Entropy and LR refinement did not beat the simpler 50M constant recipe. Episode-risk continuation lost to ordinary PPO and was stopped.

The three risk rows are **continuations**, not fresh short runs: each starts from our own 24,985,600-step, 1,088.77-second teacher and restarts its optimizer. Times and transitions in those rows show only additional work; their shared prefix is disclosed and is already counted once in the project training history.

PickCube uses the stock pick metric, which does not require release, and a different controller from StackCube. Its 10-minute result is not an earliest-time-to-success measurement. StackCube uses the official frozen final-step stacking metric; ten-step stability is a separate diagnostic.

Assisted* rows share the **49,971,200-transition / 2,295.41-second teacher**, plus **25,600 demo transitions**, **29.37 seconds of generation** and **5,000 BC updates / 10.87 trainer seconds**. Both PPO variants then used 5.12M new transitions and fell from BC-only49/128 to0/128. Residual-noise initialization failed its gate and is stopped. Shared assistance is disclosed rather than counted as free or as a from-scratch gain.

## Current research review and prepared work

The original [first-update mechanism probe](first-update-probe-results-20261028.md) is complete:201,600 new transitions,46.66s on one L40S, six counted one-step optimizer calls. Its Adam direction failed the forward/reverse criterion; large movement caused damage. The [October30 follow-ups](autoresearch-journal-20261030.md) subsequently found useful constrained MC directions, then failed to transfer that benefit to complete from-scratch PPO. These scoped findings coexist; the original record is unchanged. The earlier [replay-moment study](replay-moment-study-plan-20261028.md) completed three fresh runs:1,881,024 transitions,14 once-only development scores, all0/128. Shared51.2k/102.4k/256k checkpoints are its equal-step comparison. Historical prepared payloads below remain disabled; current autonomy does not silently activate them.

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
| [Short-history motion prediction](history-motion-results-20261029.md) |2.754 s CPU child |20 closed-form fits on existing data; primary history-effect gate failed in both seeds against current-state/capacity/neighbor/kinematic controls. No new robot interaction, GPU work or RL gain. |
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
| scratch-credit-ppo-gae | 1 | 591.079 | 10,240,000 | 116/128 | — |
| scratch-credit-ppo-gae | 2 | 553.243 | 10,240,000 | 0/128 | — |
| scratch-credit-ppo-gae | 3 | 546.757 | 10,240,000 | 0/128 | — |
| scratch-credit-ppo-finite-mc | 1 | 596.618 | 8,960,000 | 0/128 | — |
| scratch-credit-ppo-finite-mc | 2 | 596.730 | 9,625,600 | 0/128 | — |
| scratch-credit-ppo-finite-mc | 3 | 596.691 | 9,574,400 | 0/128 | — |
| feature-value-ppo-feature-value | 1 | 595.585 | 9,676,800 | 0/128 | — |
| feature-value-ppo-feature-value | 2 | 595.410 | 10,035,200 | 0/128 | — |
| feature-value-ppo-feature-value | 3 | 535.998 | 10,240,000 | 0/128 | — |
| feature-value-ppo-clock-feature-value | 1 | 596.479 | 9,932,800 | 0/128 | — |
| feature-value-ppo-clock-feature-value | 2 | 594.519 | 10,188,800 | 0/128 | — |
| feature-value-ppo-clock-feature-value | 3 | 596.718 | 10,240,000 | 0/128 | — |
| gripper-prior-full | 41 | 250.305 | 4,096,000 | 0/128 | — |
| gripper-prior-masked | 41 | 237.650 | 4,096,000 | 0/128 | — |
| gripper-prior-hold | 41 | 212.976 | 4,096,000 | 0/128 | — |
| se-control | 11 | 584.828 | 15,360,000 | 115/128 | — |
| se-control | 12 | 745.311 | 15,360,000 | 120/128 | — |
| se-control | 13 | 726.775 | 15,360,000 | 82/128 | — |
| se-control | 21 | 692.471 | 15,360,000 | 103/128 | — |
| se-control | 22 | 712.779 | 15,360,000 | 121/128 | — |
| se-control | 23 | 707.998 | 15,360,000 | 119/128 | — |
| se-control | 24 | 715.508 | 15,360,000 | 118/128 | — |
| se-control | 25 | 721.025 | 15,360,000 | 121/128 | — |
| se-env512 | 11 | 1005.960 | 15,360,000 | 109/128 | — |
| se-env512 | 12 | 1091.354 | 15,360,000 | 114/128 | — |
| se-env512 | 13 | 992.867 | 15,360,000 | 119/128 | — |
| se-env256 | 11 | 1976.181 | 15,360,000 | 117/128 | — |
| se-env256 | 12 | 1970.150 | 15,360,000 | 116/128 | — |
| se-env256 | 13 | 1985.724 | 15,360,000 | 115/128 | — |
| se-obsnorm | 11 | 610.292 | 15,360,000 | 120/128 | — |
| se-obsnorm | 12 | 601.320 | 15,360,000 | 114/128 | — |
| se-obsnorm | 13 | 612.219 | 15,360,000 | 121/128 | — |
| se-obsnorm-env256 | 11 | 1958.339 | 15,360,000 | 118/128 | — |
| se-obsnorm-env256 | 12 | 1960.052 | 15,360,000 | 117/128 | — |
| se-obsnorm-env256 | 13 | 1714.662 | 15,360,000 | 121/128 | — |
| se-t-gripper | 11 | 652.504 | 15,360,000 | 0/128 | — |
| se-t-gripper | 12 | 765.154 | 15,360,000 | 0/128 | — |
| se-t-gripper | 13 | 742.678 | 15,360,000 | 0/128 | — |
| se-t-all | 11 | 747.405 | 15,360,000 | 0/128 | — |
| se-t-all | 12 | 766.378 | 15,360,000 | 0/128 | — |
| se-t-all | 13 | 741.422 | 15,360,000 | 0/128 | — |
| se-obsnorm-env256-t-gripper | 11 | 2048.838 | 15,360,000 | 117/128 | — |
| se-obsnorm-env256-t-gripper | 12 | 2062.882 | 15,360,000 | 121/128 | — |
| se-obsnorm-env256-t-gripper | 13 | 1834.533 | 15,360,000 | 0/128 | — |
| se-obsnorm-env256-t-all | 11 | 2119.861 | 15,360,000 | 4/128 | — |
| se-obsnorm-env256-t-all | 12 | 2147.305 | 15,360,000 | 0/128 | — |
| se-obsnorm-env256-t-all | 13 | 2132.670 | 15,360,000 | 0/128 | — |
| se-bin | 11 | 616.882 | 15,360,000 | 0/128 | — |
| se-bin | 12 | 754.932 | 15,360,000 | 0/128 | — |
| se-bin | 13 | 709.143 | 15,360,000 | 0/128 | — |
| se-env256-bin | 11 | 2040.360 | 15,360,000 | 0/128 | — |
| se-env256-bin | 12 | 2058.428 | 15,360,000 | 0/128 | — |
| se-env256-bin | 13 | 2056.414 | 15,360,000 | 1/128 | — |
| se-env128 | 11 | 2671.944 | 10,912,000 | 121/128 | — |
| se-env128 | 12 | 2672.717 | 12,416,000 | 102/128 | — |
| se-env128 | 13 | 2672.649 | 10,963,200 | 114/128 | — |
| se-gfloor | 11 | 796.084 | 15,360,000 | 125/128 | — |
| se-gfloor | 12 | 750.089 | 15,360,000 | 118/128 | — |
| se-gfloor | 13 | 773.883 | 15,360,000 | 124/128 | — |
| se-env256-gfloor | 11 | 2160.342 | 15,360,000 | 108/128 | — |
| se-env256-gfloor | 12 | 2015.593 | 15,360,000 | 120/128 | — |
| se-env256-gfloor | 13 | 2069.807 | 15,360,000 | 122/128 | — |
| se-bineps | 11 | 941.666 | 15,360,000 | 124/128 | — |
| se-bineps | 12 | 931.358 | 15,360,000 | 123/128 | — |
| se-bineps | 13 | 906.108 | 15,360,000 | 126/128 | — |
| se-bineps | 21 | 872.507 | 15,360,000 | 122/128 | — |
| se-bineps | 22 | 935.796 | 15,360,000 | 115/128 | — |
| se-bineps | 23 | 957.171 | 15,360,000 | 125/128 | — |
| se-bineps | 24 | 819.877 | 15,360,000 | 125/128 | — |
| se-bineps | 25 | 911.085 | 15,360,000 | 122/128 | — |
| se-env256-bineps | 11 | 2229.993 | 15,360,000 | 119/128 | — |
| se-env256-bineps | 12 | 1929.359 | 15,360,000 | 118/128 | — |
| se-env256-bineps | 13 | 1932.576 | 15,360,000 | 116/128 | — |
| se-env256-bineps | 21 | 1967.025 | 15,360,000 | 117/128 | — |
| se-env256-bineps | 22 | 1907.360 | 15,360,000 | 115/128 | — |
| se-env256-bineps | 23 | 1904.996 | 15,360,000 | 119/128 | — |
| se-env256-bineps | 24 | 2207.453 | 15,360,000 | 120/128 | — |
| se-env256-bineps | 25 | 2213.023 | 15,360,000 | 116/128 | — |
| se-bineps05 | 11 | 850.425 | 15,360,000 | 124/128 | — |
| se-bineps05 | 12 | 803.239 | 15,360,000 | 124/128 | — |
| se-bineps05 | 13 | 804.590 | 15,360,000 | 127/128 | — |
| se-bineps-sticky50 | 11 | 983.939 | 15,360,000 | 119/128 | — |
| se-bineps-sticky50 | 12 | 985.200 | 15,360,000 | 122/128 | — |
| se-bineps-sticky50 | 13 | 805.758 | 15,360,000 | 121/128 | — |
| se-bineps-sticky80 | 11 | 822.474 | 15,360,000 | 101/128 | — |
| se-bineps-sticky80 | 12 | 814.782 | 15,360,000 | 102/128 | — |
| se-bineps-sticky80 | 13 | 1022.125 | 15,360,000 | 99/128 | — |
| se-sym | 11 | 851.761 | 15,360,000 | 116/128 | — |
| se-sym | 12 | 689.373 | 15,360,000 | 119/128 | — |
| se-sym | 13 | 755.197 | 15,360,000 | 119/128 | — |
| se-bineps-sym | 11 | 1358.741 | 15,360,000 | 126/128 | — |
| se-bineps-sym | 12 | 1423.444 | 15,360,000 | 126/128 | — |
| se-bineps-sym | 13 | 1330.502 | 15,360,000 | 120/128 | — |
| se-bineps-sym08 | 11 | 1336.101 | 15,360,000 | 124/128 | — |
| se-bineps-sym08 | 12 | 1327.706 | 15,360,000 | 121/128 | — |
| se-bineps-sym08 | 13 | 999.559 | 15,360,000 | 124/128 | — |
| se-bineps-std37 | 11 | 958.057 | 15,360,000 | 116/128 | — |
| se-bineps-std37 | 12 | 981.348 | 15,360,000 | 116/128 | — |
| se-bineps-std37 | 13 | 968.534 | 15,360,000 | 112/128 | — |
| se-bineps-std22 | 11 | 850.671 | 15,360,000 | 0/128 | — |
| se-bineps-std22 | 12 | 782.213 | 15,360,000 | 0/128 | — |
| se-bineps-std22 | 13 | 948.047 | 15,360,000 | 0/128 | — |
| se-bineps-ent0 | 11 | 681.114 | 15,360,000 | 121/128 | — |
| se-bineps-ent0 | 12 | 783.540 | 15,360,000 | 125/128 | — |
| se-bineps-ent0 | 13 | 723.687 | 15,360,000 | 120/128 | — |
| se-env512-bineps | 11 | 1125.539 | 15,360,000 | 120/128 | — |
| se-env512-bineps | 12 | 1153.349 | 15,360,000 | 121/128 | — |
| se-env512-bineps | 13 | 1129.707 | 15,360,000 | 124/128 | — |
| se-env512-bineps | 21 | 1229.578 | 15,360,000 | 124/128 | — |
| se-env512-bineps | 22 | 1153.362 | 15,360,000 | 115/128 | — |
| se-env512-bineps | 23 | 1275.358 | 15,360,000 | 124/128 | — |
| se-env512-bineps | 24 | 1350.757 | 15,360,000 | 119/128 | — |
| se-env512-bineps | 25 | 1449.173 | 15,360,000 | 116/128 | — |
| se-td3 | 11 | 413.026 | 5,120,000 | 123/128 | — |
| se-td3 | 12 | 382.772 | 5,120,000 | 124/128 | — |
| se-td3 | 21 | 626.833 | 7,680,000 | 118/128 | — |
| se-td3 | 22 | 599.399 | 7,680,000 | 118/128 | — |
| se-td3 | 23 | 607.921 | 7,680,000 | 116/128 | — |
| se-td3 | 24 | 618.249 | 7,680,000 | 124/128 | — |
| se-td3 | 25 | 637.376 | 7,680,000 | 123/128 | — |
| se-td3-burst | 11 | 427.924 | 5,120,000 | 122/128 | — |
| se-td3-burst | 12 | 426.261 | 5,120,000 | 122/128 | — |
| se-td3-burst-sym | 11 | 483.389 | 5,120,000 | 53/128 | — |
| se-td3-burst-sym | 12 | 459.579 | 5,120,000 | 116/128 | — |
| se-td3-utd8 | 11 | 614.665 | 5,120,000 | 124/128 | — |
| se-td3-utd8 | 12 | 610.581 | 5,120,000 | 117/128 | — |
| se-td3-utd16 | 11 | 1032.492 | 5,120,000 | 122/128 | — |
| se-td3-utd16 | 12 | 1014.743 | 5,120,000 | 124/128 | — |
| se-td3-utd16 | 21 | 1578.738 | 7,680,000 | 124/128 | — |
| se-td3-utd16 | 22 | 1586.700 | 7,680,000 | 127/128 | — |
| se-td3-utd16 | 23 | 1579.528 | 7,680,000 | 122/128 | — |
| se-td3-utd16 | 24 | 1576.111 | 7,680,000 | 121/128 | — |
| se-td3-utd16 | 25 | 1574.949 | 7,680,000 | 124/128 | — |
| se-td3-utd32 | 11 | 1879.193 | 5,120,000 | 123/128 | — |
| se-td3-utd32 | 12 | 1894.832 | 5,120,000 | 117/128 | — |
| se-ppo-control [PlaceSphere] | 11 | 792.845 | 15,360,000 | 83/128 | — |
| se-ppo-control [PlaceSphere] | 12 | 865.909 | 15,360,000 | 112/128 | — |
| se-ppo-control [PlaceSphere] | 13 | 893.993 | 15,360,000 | 101/128 | — |
| se-ppo-env512-bineps [PlaceSphere] | 11 | 1509.248 | 15,360,000 | 98/128 | — |
| se-ppo-env512-bineps [PlaceSphere] | 12 | 1705.976 | 15,360,000 | 108/128 | — |
| se-ppo-env512-bineps [PlaceSphere] | 13 | 1631.438 | 15,360,000 | 103/128 | — |
| se-td3 [PlaceSphere] | 11 | 667.291 | 7,680,000 | 109/128 | — |
| se-td3 [PlaceSphere] | 12 | 624.720 | 7,680,000 | 118/128 | — |
| se-td3 [PlaceSphere] | 13 | 642.051 | 7,680,000 | 109/128 | — |
| se-ppo-control [PickCube] | 11 | 517.003 | 15,360,000 | 128/128 | — |
| se-ppo-control [PickCube] | 12 | 534.493 | 15,360,000 | 120/128 | — |
| se-ppo-control [PickCube] | 13 | 620.429 | 15,360,000 | 126/128 | — |
| se-ppo-control [PickCube] | 21 | 520.050 | 15,360,000 | 124/128 | — |
| se-ppo-control [PickCube] | 22 | 541.105 | 15,360,000 | 128/128 | — |
| se-ppo-control [PickCube] | 23 | 547.841 | 15,360,000 | 127/128 | — |
| se-ppo-control [PickCube] | 24 | 570.576 | 15,360,000 | 128/128 | — |
| se-ppo-control [PickCube] | 25 | 544.141 | 15,360,000 | 127/128 | — |
| se-ppo-env512-bineps [PickCube] | 11 | 1122.613 | 15,360,000 | 128/128 | — |
| se-ppo-env512-bineps [PickCube] | 12 | 1111.868 | 15,360,000 | 128/128 | — |
| se-ppo-env512-bineps [PickCube] | 13 | 1236.733 | 15,360,000 | 128/128 | — |
| se-td3 [PickCube] | 11 | 564.664 | 7,680,000 | 128/128 | — |
| se-td3 [PickCube] | 12 | 591.127 | 7,680,000 | 128/128 | — |
| se-td3 [PickCube] | 13 | 561.267 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PlaceSphere] | 11 | 1567.623 | 7,680,000 | 104/128 | — |
| se-td3-utd16 [PlaceSphere] | 12 | 1611.344 | 7,680,000 | 113/128 | — |
| se-td3-utd16 [PlaceSphere] | 13 | 1653.995 | 7,680,000 | 99/128 | — |
| se-td3-utd16 [PickCube] | 11 | 1574.395 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PickCube] | 12 | 1582.409 | 7,680,000 | 127/128 | — |
| se-td3-utd16 [PickCube] | 13 | 1588.898 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PickCube] | 21 | 1553.188 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PickCube] | 22 | 1524.521 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PickCube] | 23 | 1552.344 | 7,680,000 | 127/128 | — |
| se-td3-utd16 [PickCube] | 24 | 1556.611 | 7,680,000 | 128/128 | — |
| se-td3-utd16 [PickCube] | 25 | 1590.817 | 7,680,000 | 128/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 11 | 758.939 | 15,360,000 | 102/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 12 | 811.578 | 15,360,000 | 125/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 13 | 764.144 | 15,360,000 | 110/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 21 | 854.718 | 15,360,000 | 109/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 22 | 819.702 | 15,360,000 | 106/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 23 | 972.292 | 15,360,000 | 106/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 24 | 800.000 | 15,360,000 | 115/128 | — |
| se-ppo-control-anneal [PlaceSphere] | 25 | 817.835 | 15,360,000 | 116/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 11 | 1599.755 | 15,360,000 | 124/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 12 | 1673.904 | 15,360,000 | 125/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 13 | 1565.726 | 15,360,000 | 121/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 21 | 1639.209 | 15,360,000 | 126/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 22 | 1560.484 | 15,360,000 | 124/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 23 | 1577.624 | 15,360,000 | 121/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 24 | 1589.483 | 15,360,000 | 123/128 | — |
| se-ppo-env512-bineps-anneal [PlaceSphere] | 25 | 1725.834 | 15,360,000 | 119/128 | — |
| se-ppo-control [LiftPegUpright] | 11 | 642.058 | 15,360,000 | 127/128 | — |
| se-ppo-control [LiftPegUpright] | 12 | 641.936 | 15,360,000 | 128/128 | — |
| se-ppo-control [LiftPegUpright] | 13 | 632.985 | 15,360,000 | 127/128 | — |
| se-ppo-env512-bineps [LiftPegUpright] | 11 | 1306.474 | 15,360,000 | 128/128 | — |
| se-ppo-env512-bineps [LiftPegUpright] | 12 | 1073.905 | 15,360,000 | 128/128 | — |
| se-ppo-env512-bineps [LiftPegUpright] | 13 | 1115.383 | 15,360,000 | 128/128 | — |
| se-td3-utd16 [LiftPegUpright] | 11 | 1547.495 | 7,680,000 | 126/128 | — |
| se-td3-utd16 [LiftPegUpright] | 12 | 1563.314 | 7,680,000 | 126/128 | — |
| se-td3-utd16 [LiftPegUpright] | 13 | 1585.257 | 7,680,000 | 128/128 | — |

## Source records and updates

[Exact run data and source digests](experiment-runs.json). Rebuild with `python scripts/build_experiment_ledger.py` from this checkout and its retained local archives. The generator reads recorded aggregates only; it does not train, evaluate, deserialize native checkpoints or change cloud resources.

Detailed reports: [official StackCube baseline](stackcube-results-20261026.md), [early sweeps](stackcube-overnight-results-20261027.md), [full-horizon investigation](stackcube-settling-20261027.md), [CAPG](stackcube-capg-results-20261027.md), [SAC](stackcube-sac-results-20261027.md), [50M screen](stackcube-precision-results-20261027.md), [50M replication](stackcube-precision-replication-results-20261028.md), [local BC/prior checks](stackcube-amp-local-dataflow-20261027.md), and [residual exploration hypothesis](research-discovery-bc-residual-exploration-v2-20261028.md).
