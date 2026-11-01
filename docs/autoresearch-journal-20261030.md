# AutoManip research journal — 2026-10-30

User mandate: continue autonomous research until asked to stop. Preserve the historical benchmarks, account for assistance and compute, and test mechanisms cheaply before robot training. The original tenfold sample-efficiency objective remains unmet. The first cumulative-learning code milestone is infrastructure, not an achieved learning benefit.

## Current ideas and the evidence deciding between them

The broad questions are reusable corrections, preservation during adaptation, action-consequence representations learned from old experience, and reliable acquisition of useful trajectories. Only one mechanism chain is being pursued at a time; these are not a queued algorithm sweep.

Residual control is established prior work: [Johannink et al.](https://arxiv.org/abs/1812.03201) combine a base controller with learned corrections, and [Silver et al.](https://arxiv.org/abs/1812.06298) study residual policy learning. Here a simple output-layer correction can be fused exactly into the existing MLP, with no new inference runtime. That is head-only fine-tuning, not an invention. The local fusion and frozen-parameter checks pass; they do not guarantee retention.

The more relevant warning is [forgetting during RL fine-tuning](https://arxiv.org/abs/2402.02868): early visitation can exclude states on which a policy previously worked. This motivated separating hidden-feature preservation, old-state functional preservation, and the reliability of the proposed learning direction. [TRPO](https://arxiv.org/abs/1502.05477) provides prior context for trust-region/natural-gradient geometry; none of the probes below is a complete implementation or monotonic-improvement claim.

## Robot diagnostics: existing competence was stronger than the proposed deficits

The unchanged 50M source already passed the [north-strip test](cumulative-preflight-results-20261030.md): 124/128 original-task and 122/128 target successes. The registered stopping rule prevented unnecessary adaptation.

A subsequent fixed-force recovery probe failed during the sham calibration at step20: ManiSkill's `Actor.apply_force` calls `cuda_rigid_body_force`, which the pinned SAPIEN `3.0.0b1` backend does not expose. This consumed **160 calibration transitions**, with no push, no policy assessments and no learning. Its logs/report are preserved. The partial trajectory was not flushed by that original diagnostic; those160 state/action rows cannot be reconstructed from the retained report. This is a missing asset, not invented data. The later probe saves a partial trace on ordinary errors.

Without changing the physics dependency, a **separately registered instantaneous-impulse task** adds `J/m_actual` to destination-cube velocity once, before control step20, then applies/fetches rigid-body data. `J=.08 N·s` along scene ±y, balanced by episode seed. This is not an implementation of the failed finite-force task. The supported rigid-body data interface is documented by [SAPIEN](https://sapien-sim.github.io/docs/api/sapien.pysapien.physx.html); actual pinned GPU execution supplies the validation below.

| Fixed source policy | Successes | Execution failures |
|---|---:|---:|
| Sham | 125/128 | 0 |
| Impulse | 116/128 | 0 |

All128 destination cubes remained on the table. The initial observation trajectories matched **exactly through the pre-impulse state**. There were12 sham-only successes,3 impulse-only successes and113 successes in both conditions. The impulse was not uniformly adversarial: it rescued the three original failures. Median destination displacement relative to sham immediately after the affected control interval was about5.04 cm. Free-cube calibration displacements were7.47–25.94 cm, with all eight cubes remaining on the table.

Ten of the12 pushed failures reached placement at some point; eight were still geometrically on top at the last step, but only two of the12 had the cube-static flag. One remained grasped. These are trajectory diagnostics, not established causes. The existing policy still passed the registered90% gate. No impulse-strength search or adaptation training followed.

The impulse worker took66.465 s; its wrapper70.405 s. Completed new experience across the force failure and impulse probe was **13,760 transitions** (160 +800 calibration +12,800 assessment). Both exact apps were verified stopped with no tasks. Reported cost for the two was **$0.05800341**, subject to metering lag. Including the earlier north-strip attempts gives **27,360 completed transitions and $0.11153757 reported cost** for these new GPU diagnostics. All final/confirmation sets remain untouched.

Evidence: `artifacts/cumulative/recovery-force-failure-20261030/`, `artifacts/cumulative/recovery-impulse-20261030/`; source-bound plans under `cumulative/experiments/recovery-*-v1.json`; billing/state receipt `.runtime/recovery-impulse-probe-20261030/closeout.json`.

## Reuse the known collapse, rather than search for a policy deficit

The next probes return to the [actual first-update collapse](first-update-probe-results-20261028.md). **This is the historical assisted BC actor, not the stronger50M actor above.** Three archived data draws share the same initial actor, random critic and calibrated Gaussian scale; they are not independent training seeds. The old approved experiment established immediate damage from large mean movement but did not establish a useful small forward direction.

All new CPU probes use existing numeric NPZ assets with `allow_pickle=False`; no new teacher data, simulator query or native checkpoint loading is required. Source costs remain49,971,200 teacher transitions /2,295.408 trainer seconds, plus25,600 demonstration transitions and5,000 BC updates. Old data is prior information, not free from-scratch initialization.

### 1. Does freezing hidden features itself protect old behavior?

Reconstructed the original first update from each saved initial state and original minibatch. Literal production-versus-independent CPU calculations matched bitwise, including gradients, Adam state, targets and RNG. Saved GAE targets matched exactly; CPU versus archived GPU parameter differences were below9.67e−7. This is CPU operator parity and finite GPU agreement, not a new bitwise CUDA test.

Compared full-network and head-only mean steps at the same current-distribution Gaussian KL=.01, on64 disjoint current-rollout episodes. Old-state diagnostics use t30:50 from the first64 original BC training episodes,1,280 states, without fitting on them.

| Archived draw | Old-state KL, head / full at matched current KL |
|---|---:|
| 90282001 | 1.282 |
| 90282002 | 1.247 |
| 90282003 | 1.253 |

**Rejected.** Head-only updates changed old late-state behavior about26% more on average. Keeping features fixed did not create functional locality. The smaller unscaled head step would have confounded this finding without KL matching. No corresponding GPU test launched.

Cost:3.141 s local CPU,9 optimizer calls /14,400 fitting-row uses,586,560 actor-forward rows and20,544 critic-forward rows. No new environment interactions. [Registration](../cumulative/experiments/readout-locality-cpu-gate-v1.json); result `artifacts/cumulative/readout-locality-cpu-v1/report.json`.

### 2. Can explicit old-state geometry protect behavior without consuming the fitted direction?

For a frozen hidden representation with bias feature `phi`, whiten the mean correction by the fixed Gaussian standard deviations. Solve

`U = G (C_current + lambda C_old + ridge I)^−1`,

then scale to current mean-KL=.01. `G` is the original minibatch score/advantage gradient. Compare lambda0 and1, with the same features, data and ridge. This compares a current-only natural-gradient readout control against the added old-state metric; it does not credit the memory term for changing optimizers relative to Adam.

The candidate's old-state KL was9.72%,9.46%,9.87% of the current-only control, while retaining95.83%,96.19%,95.67% of its fitted linear gain. Actual actor KL matched the closed-form calculation. The registered geometric gate passed in all three draws.

This is an approximately90% reduction in **action-distribution change on those old states**, not tenfold sample efficiency or robot success. The fitted signal might still be noise. Cost:0.632 s CPU, six small linear solves, zero optimizer calls or new interactions. [Registration](../cumulative/experiments/old-state-metric-cpu-gate-v1.json); result `artifacts/cumulative/old-state-metric-cpu-v1/report.json`.

### 3. Does that fitted signal reproduce on independent data?

Before spending on prepared forward/reverse robot rollouts, evaluate each fixed direction using the other two archived collections. Keep each direction's original fitting-minibatch advantage normalization constants fixed. Also measure within-draw labels from disjoint episodes. Recomputed fitting gains matched the original algebra.

Both methods had six positive cross-draw estimates, but their mean cross-draw gain was only **0.28% of the fitted gain**. The old-state metric did not solve this reliability problem. The registered gate required at least10%, plus sign checks, and failed. Thus the384,000-transition paired GPU probe was **prepared but not launched**. Its13 safe policy artifacts remain preserved; export parity was within1.40e−6.

These are GAE/score-estimator projections, not true advantages, an unbiased conditional continuation-variance decomposition, or measured task-return improvements. There are only three independent collections from one policy. The finding weakens use of the in-sample surrogate as an efficacy gate; it does not prove that the policy gradient is zero or that PPO is generally invalid.

Cost:0.592 s CPU, zero updates/interactions. [Registration](../cumulative/experiments/cross-draw-direction-cpu-gate-v1.json); result `artifacts/cumulative/cross-draw-direction-cpu-v1/report.json`.

### 4. Is the original minibatch too small for this direction estimate?

With curvature, features, normalization constants and current KL unchanged, replace the1,600-row gradient by the mean over the already-collected51,200-row rollout. Cross-draw gain rose **2.10×**, from0.000445 to0.000935. However it still retained only **3.39%** of the fitted gain, below the registered10% reliability criterion. All six cross-draw projections were positive, but the full gate failed. No GPU probe launched.

The larger direction uses32× more score rows for one estimate; this is explicitly counted and is not a compute-matched comparison of complete learners. A future gradient-accumulation learner would need comparison with an ordinary PPO epoch processing the same rows, not one minibatch. Cost:0.706 s CPU, six linear solves, zero new interactions. [Registration](../cumulative/experiments/gradient-aggregation-cpu-gate-v1.json); result `artifacts/cumulative/gradient-aggregation-cpu-v1/report.json`.

### 5. Does inverse feature covariance explain the weak direction?

Held the full-rollout gradient and current KL fixed, comparing inverse-covariance and identity feature metrics. Identity improved its validation-to-fitting ratio to22.93%, but its absolute independent gain was **19% lower** and only4/6 signs were positive. It failed the registered requirement to improve independent gain by2× with5/6 positive cells. Only about5–6% of inverse-metric movement lay below the prespecified low-eigenvalue threshold. This weakens the proposed dominant low-eigenvalue explanation; it does not establish that preconditioning is generally harmless or harmful.

Cost:0.035 s CPU, cached sufficient statistics, no new score rows, optimizer calls or interactions. [Registration](../cumulative/experiments/feature-metric-cpu-gate-v1.json); result `artifacts/cumulative/feature-metric-cpu-v1/report.json`.

### 6. Does a properly separated value baseline improve the signal?

Compared a constant baseline with a linear value baseline on the existing actor features. Entire even/odd episodes were separated: a trajectory's baseline was fitted only on the opposite group. For independent collection validation, predictions came from the fitted training collection, never validation reward labels. Both methods used the **same finite50-step Monte Carlo return**, gamma.9 and zero terminal tail. That is an explicit target change from historical continuing-return GAE, not a reconstruction claim; only the baseline differs within this matched test.

The state baseline reduced the score-weighted residual second moment to34.8–35.6% of the constant control. Its independent directional gain increased by only20.6%, with validation-to-fitting ratio7.03% versus3.92%. All six cross-draw signs were positive, but the prespecified2× benefit and10% reliability thresholds were unmet. Predictive value MSE alone therefore did not admit a GPU test.

Cost:0.881 s CPU, six value fits over153,600 previously collected rows, six actor-direction solves, no new interactions. Peak process RSS852.7 MB. [Registration](../cumulative/experiments/crossfit-value-cpu-gate-v1.json); result `artifacts/cumulative/crossfit-value-cpu-v1/report.json`.

## Screening limits and next causal question

The10% validation-to-fitting cutoff is a **prespecified prioritization heuristic**, not a law of RL, a confidence interval, a success-rate threshold, or proof that a rejected method cannot work. Six cross-checks reuse three independent data collections; they are not six independent experiments. Ratios can be small simply because a direction was chosen to improve its fitting objective. No screen should become a substitute for measuring task returns, and the usefulness of this screen itself needs calibration.

The immediate estimator question is whether replacing sampled future rewards with a learned next-state value yields a useful direction, or merely lower-variance critic bias. Any such direction must be evaluated against independent Monte Carlo score estimates, not just agreement with its own TD targets. Terminal observations absent from the historical archive must not be fabricated: a separately declared finite-horizon target can assign terminal value zero, but cannot claim reconstruction of the original continuing-return objective. No new critic implementation or robot admission follows merely from the failed baseline gate.

Historical training-run counts remain72 runs across40 families. The probes above are diagnostics, not additional full training experiments or a demonstrated learning mechanism. Autonomous research remains active; the journal and continuation checkpoint should record the next registered test and any negative evidence.

## Two distinct GPU mechanisms: actual update utility and temporal exploration

The user explicitly removed cost hesitation and asked for different experiments, rather than counting replications as different ideas. We registered two separate questions and ran them on L40S. No historical task, evaluator, results or confirmation/final partitions were changed. Both probes used the unchanged official normalized dense reward in a new public diagnostic adapter; the historical scorer still disables rewards. Complete trajectories, failed behavior, work journals and source provenance are archived locally under `artifacts/cumulative/`. These were scoring interventions, **zero training optimizer calls**.

### 7. Calibrating the CPU screen against actual robot behavior

[Pre-outcome protocol](../cumulative/experiments/proxy-calibration-v1.json). Three old data collections generated fixed readout directions from **one assisted BC actor**, not three independently trained actors. We assessed unchanged, archived harmful large PPO mean step, and constant/state-baseline MC directions both forward and reverse. Each branch had512 fresh public reset episodes per collection, deterministic and stochastic modes. Stochastic comparisons kept the original Gaussian standard deviation and paired underlying noise; noise was added before action clipping. All paired initial observations matched exactly and every rollout completed without execution errors.

The negative control reproduced material damage: deterministic success fell from567/1536 (36.91%) to244/1536 (15.89%). In stochastic mode:

| Fixed actor branch | Final successes | Success rate |
| --- | ---: | ---: |
| Unchanged | 414/1536 | 26.95% |
| Archived large PPO mean step | 187/1536 | 12.17% |
| MC constant baseline, forward | 438/1536 | 28.52% |
| MC constant baseline, reverse | 386/1536 | 25.13% |
| MC state baseline, forward | 473/1536 | 30.79% |
| MC state baseline, reverse | 381/1536 | 24.80% |

Both MC methods met the registered **proxy false-negative** criterion: their stochastic discounted return gains against unchanged and reverse had positive98.75% batch-bootstrap lower bounds, with no material success loss. Constant forward-minus-unchanged return was0.06074 [0.03398,0.09312]; state was0.09332 (full interval in the archived analysis). The state method's point success gain was3.84percentage points. State-minus-constant return was0.03259; its95% interval was positive but its98.75% interval included zero, so strong comparative superiority is not established.

**Scientific correction:** the prespecified10% held-out/fitted ratio rejected genuinely useful local directions here. Earlier CPU gate failures remain recorded, but this ratio must no longer serve as a hard admission rule. It can describe fitting optimism; it has not earned authority to prevent a bounded robot test. This does not establish that every failed proxy is useful, nor that these small steps form a successful iterative learner. The actor is assisted and the bootstrap conditions on these three fitted directions; there is no from-scratch or training-seed generalization claim.

Cost:921,600 diagnostic transitions; worker times226.18,225.28,230.21s in parallel; zero optimizer calls. Archive: `artifacts/cumulative/proxy-calibration-20261030/`,375 remote files /384,203,612bytes. Analysis: `analysis-v1.json`.

### 8. Does temporally correlated exploration preserve useful behavior?

[Pre-outcome protocol](../cumulative/experiments/temporal-noise-probe-v1.json). Two fixed actors (strong50M teacher and weaker BC clone), each under deterministic, IID Gaussian, stationary AR(1)rho.9 with the same marginal variance, and an IID control matched to the AR process's50-step integrated latent-noise variance. This is a coverage intervention, not PPO training; no IID likelihood was incorrectly assigned to correlated actions. The same declared BC noise-scale vector was used for both actors, not a claim to restore teacher-native exploration.

| Actor | Deterministic | IID | AR(1), same marginal variance | IID, matched latent-sum variance |
| --- | ---: | ---: | ---: | ---: |
| Teacher50M | 496/512 | 495/512 | 488/512 | 405/512 |
| BC clone | 194/512 | 138/512 | 107/512 | 4/512 |

Latent AR/IID mean-square ratio0.9899 passed calibration; observed lag-one correlation0.8989. Yet BC success decreased by6.05percentage points with AR relative to IID. The registered improvement criterion failed. Smoother actions alone therefore did not improve this frozen clone's coverage at the declared scale. The much larger IID noise was severely destructive. This is negative evidence for this particular intervention, not proof that temporal exploration can never improve learning from scratch.

Cost:204,800 diagnostic transitions; worker times79.71,79.46s in parallel; zero optimizer calls. Archive: `artifacts/cumulative/temporal-noise-20261030/`,90 remote files /128,158,690bytes; `analysis-v1.json` includes paired intervals and stage diagnostics.

## Reassessment after actual measurements

The robot tests overturn the use of the10% ratio as a blocking screen and favor investigating the useful small update over increasing exploration coherence. The next important distinction is whether useful movement comes primarily from constrained update geometry, a fitted state baseline, or replacing a bootstrapped GAE target with sampled future rewards. A matched readout-direction comparison can isolate reward target from baseline class before a complete learner changes several factors at once. Full learning curves, matched tuned baselines and subsequent transfer/retention tests are still required; no10× sample-efficiency result exists.

### 9. Target versus baseline under matched update geometry

[Pre-outcome protocol](../cumulative/experiments/credit-target-probe-v1.json). We reused the public rollout worker unchanged, with fresh reset/noise seeds, original actor/std,512 episodes per branch per old data collection, and mean KL=.01. Seven branches: unchanged and forward/reverse of historical full-rollout GAE, GAE with a cross-fitted state baseline, and finite MC with the same state-baseline class. State baselines used identical actor features, ridge and episode folds. GAE labels were the archived production returns; MC labels had gamma.9 and a zero tail at50. These target estimands differ deliberately.

The two prespecified primary contrasts used97.5% paired batch-bootstrap intervals and required at least0.03 mean discounted-return gain. MC-state minus GAE-state was **0.03030 [0.01077,0.05215]**, passing this local material-gain criterion. GAE-state minus legacy GAE was **0.02819 [0.00631,0.05090]**: positive return evidence, but below the registered0.03 material threshold, so that criterion failed. No threshold was relaxed.

Final success was428/1536 unchanged,434 legacy GAE forward,464 fitted-baseline GAE forward, and473 MC forward. **MC versus matched GAE success is unresolved**: the difference was0.59percentage points with a97.5% interval spanning−0.98 to+2.28points. Legacy GAE forward also beat reverse in return, weakening the assertion that its direction is wholly useless. The data instead support a useful constrained direction plus modest target/baseline improvements in this assisted setting. They do not establish a cause of delayed learning from scratch.

All paired initial observations matched exactly; all21 branch assessments completed with zero execution errors. Cost:537,600 new public diagnostic transitions, zero optimizer calls; workers132.47,132.76,135.42s. CPU preparation3.19s, six baseline fits and three small direction solves, no new fitting interactions. Archive: `artifacts/cumulative/credit-target-20261030/`,225 remote files /224,172,864bytes plus source provenance and `analysis-v1.json`. An initial progress read overlapped a non-atomic report write and failed JSON parsing; we obtained completed function receipts and two identical stable reads before analysis. **No physics was repeated.**

## Return to actual from-scratch learning curves

The assisted first-update chain has now supplied enough evidence to test learning rather than continue producing related one-step variations. The next registered experiment is [from-scratch PPO target comparison](../cumulative/experiments/scratch-credit-v1.json). Two methods, three seeds each: the existing strong precision-constant recipe (gamma.9, entropy.003, full-horizon,8 PPO epochs) versus replacing its complete GAE target by finite50-step Monte Carlo returns. Architecture, PPO update code, initialization, collection and reward/task are shared. The candidate changes both trace/bootstrapping target semantics as one intervention; it is **not** a lambda-only causal attribution.

Each trial has10.24M training interactions and a600-second trainer-process ceiling. Early actor checkpoints are fixed at0,0.512,1.024,2.048,4.096,6.144,8.192,10.24M. Both methods use identical added work accounting, blind first64-lane trajectory sampling and safe actor exports. Trainer images contain no evaluator/benchmark files, demonstrations or pretrained weights. Independent scorers receive safe JSON actors from a read-only candidate volume and use the unchanged physical evaluator on paired fresh public development episodes, with no feedback to training and no confirmation/final queries.

Primary evidence will be per-seed learning curves and common-grid success AUC, with T80/T90 requiring two consecutive threshold crossings and unreached crossings censored. Equal interaction/time ceilings are not proof of equal consumed compute: actual transitions, optimizer calls, learner time, exports and outer worker time must be reported. A material follow-up screen asks for at least2× earlier T80 across all three seed pairs without reduced final success or greater measured learner compute. The original10× objective remains unmet. Held-out task validation remains future work.

### 10. The from-scratch result overturned the local target preference

The six runs completed. Finite-MC PPO scored0/128 at every checkpoint in all three seeds. Strong GAE ended116/128,0/128,0/128, so its best seed is not representative. Finite MC used more optimizer calls and learner time. The [complete report](scratch-credit-results-20261030.md) retains curves and actual work:58,880,000 training transitions plus307,200 scoring transitions. A locally better assisted first update did not predict a better from-scratch learning procedure. No speedup was established.

### 11. Remaining-time value structure reduced covariance, but did not improve learning

We inspected12 retained training-data cells: two checkpoints from each of the six runs. Whole-episode cross-fitting and a fixed shuffled-clock control showed that remaining discounted horizon improved finite-MC value estimates. A follow-up computed trajectory-gradient covariance for all146,192 actor mean/logstd parameters, rather than only the readout or per-step losses. The clock reduced covariance by at least25% against state and shuffled-clock controls in all12 cells. It also beat the actual neural-MC baseline in all six MC cells. These are batch properties; GAE's continuing target is not ground truth for finite-horizon prediction.

That mechanism passed its offline checks and therefore received one bounded learning test. Both new methods used cross-fitted linear values on current actor features, with the same parameters and fits; only one scaled features by remaining discounted horizon. Neural-critic training was removed in both. **Both scored0/128 at every checkpoint in all three seeds.** The registered material AUC screen failed. Actual work:60,313,600 training and307,200 scoring transitions. [Full report](feature-value-results-20261030.md). Lower offline covariance was insufficient for this online PPO procedure; no parameter sweep follows automatically.

### 12. Narrowing the behavioral bottleneck

Blind training traces show that the failed GAE policies often achieve stacking geometry but rarely open their fingers sufficiently. A CPU action-support diagnostic rejected the stronger claim that their Gaussian means were trapped far outside the action clamp: median probability of escaping the lower clamp was approximately.50 and.62 for the two failed8.192Mpolicies. The corresponding median probability of an opening command above+.28 was only.000052 and.000196. A clamp-repair sweep is not justified by these data.

The next [registered causal diagnostic](../cumulative/experiments/release-control-probe-v1.json) compares unchanged behavior, five fixed-time opening commands, five geometry-triggered opening commands, and the same targeted opening with zero arm delta. It uses two frozen failed policies, fresh public episodes and a409,600-transition ceiling, with no optimization. All assigned episodes count, including those never reaching the geometric trigger; a late fallback equalizes command budgets. Pre-intervention trajectories must match their paired unchanged reference. A positive result would identify a repairable action-sequence bottleneck in these policies; it would not be a learned skill or a benchmark improvement.

### 13. Correcting an avoidable validity-design mistake

That replay design failed before a complete comparison. Repeated reset-and-replay was an avoidable choice because an earlier lift probe had already demonstrated hidden-state divergence despite matching measured resets/actions. The underlying error was sanitized by the frozen evaluator and the failed batch was incompletely retained; its precise cause cannot be proved. We cancelled the peer, preserved71,104confirmed transitions plus64unresolved inflight transitions, and verified the apps stopped. No threshold was relaxed and partial scores earned no scientific conclusion. [Correction report](release-control-design-correction-20261030.md).

The replacement randomized intervention groups within each64lane rollout, independent of geometry and outcomes. It required no physical counterfactual replay. Five-step opening improved final success only modestly, failing the registered20-point material screen. However targeted opening produced630/1,024ever-success episodes versus27final successes. Exploratory first-loss traces showed loss of staticity while geometry usually persisted. [Full randomized result](randomized-release-results-20261030.md).

### 14. A maintenance mechanism with direct intervention evidence

A new predeclared2×2factorial gave every group identical initial opening, then independently randomized continued opening and zero arm delta. Final success:32/1,024resume actor,336keep open,162zero arm,596both. Combined gain55.08percentage points,98.75%interval[51.27,58.79]. The interaction was+12.70points[7.03,18.55]. Opening alone and combined maintenance passed the20-point material screen; arm alone did not. This establishes a repairable maintenance bottleneck conditional on two fixed failed policies, not a learned skill or efficiency gain. [Report and trajectory figure](release-maintenance-results-20261030.md).

The two randomized probes each used204,800physics transitions and zero optimizer calls. Separate CPU reconstructions exactly reproduced all409,600executed action rows from recorded observations, fixed actor/std, assignments and masks, without rerunning physics. No diagnostic trajectories enter learning.

An additional inference-only audit examined15,894pre-action states where maintained stacks were already successful. Original mean commands still moved the arm (clipped-mean RMS.170and.294) and strongly closed the gripper (median closing probabilities.998and.995). Stochastic variance contributed most of the arm command energy, but changing variance alone would not remove the mean's closing/motion. This does not identify physical counterfactual outcomes; it prevents misdescribing the issue as noise alone.

### 15. Essential checks before another learner

A [future-innovation scalar control-variate test](future-innovation-toy-results-20261030.md) passed mean-gradient and whole-trajectory covariance screens, including clipped dynamics and deliberately biased negative controls. It used no robot data or physics. The failed online value experiment cautions against treating another variance property as a task improvement.

We also registered a small assisted stopping toy: learn a persistent correction trigger from randomized choices and final outcomes, with no provided state threshold. This first version failed in all three toy seeds. Its rollout-calibration gate correctly withheld the candidate; both learned and shuffled-label candidates retained baseline behavior. We are diagnosing model underconfidence versus inadequate state/features before proposing any robot training. The base controller and zero-action correction are explicitly supplied priors in this toy, not a from-scratch claim.


### 16. Training-only randomized correction outcomes and support

The randomized training corpus used 204,800 new transitions and two fixed acquired
parents. Frozen actor features predicted held-out correction outcomes better than
constant and time-only controls. The raw learned trigger nevertheless intervened
at unsupported initial states. A training-derived support restriction prevented
this extrapolation. These are prediction/support results, not policy improvements.
See [training report](random-switch-training-results-20261030.md).

### 17. Closed-loop correction selection failed its strongest controls

Fresh public prototype evaluation used 51,200 transitions: original Gaussian
actor 0/256, random-time correction 108/256, raw learned trigger 50/256, supported
trigger 100/256. Support helped relative to raw selection but did not beat random
timing. No score or public trajectory was fed back into fitting.
See [selector report](learned-correction-results-20261030.md).

A subsequent finite-horizon waiting test passed an analytic CPU toy, then failed
on fresh robot episodes: state continuation 131/256 (51.17%) versus a fixed time
learned from training outcomes 146/256 (57.03%). This assessment used a further
51,200 transitions. The primary paired interval included zero; the registered
overall criterion failed. The supplied correction and parent acquisition remain
explicit assistance. Close the assisted selector branch without another tuning
variation. See [waiting report](continuation-correction-results-20261030.md).

### 18. User-approved change to a branching mechanism test

After reviewing the six-page Grow, Plan, Distill proposal, the user approved a
minimal planning-to-distillation investigation. First verify same-lane branching
under the pinned physical kernel, then test whether planner improvements survive
distillation. No backward curriculum, learned terminal model, controller change,
or architecture expansion is admitted yet. All new interactions, including the
diagnostic, must stay within the proposed one-million ceiling. First two runtime
attempts stopped before any physics (index dtype; BatchedRNG deepcopy); both are
preserved and stopped. A corrected attempt uses identical scientific settings,
94,720 diagnostic transitions and zero optimizer calls. Its scientific result
is pending at this journal entry. [Protocol](branching-validity-plan-20261030.md).


### 19. Direct restore failed; no-op writing exposed a concrete simulator effect

The completed augmented-state diagnostic used94,720transitions and zero optimizer
calls. Both physical-only and augmented methods passed zero of24replay batches.
Immediate state errors were at most1.907micrometres with matching joint/cube
velocities, but1,533/1,536replay-lane instances pervariant later exceeded fixed
physical tolerances. Three repeats share states and are not independent samples.
[Full report](branching-validity-results-20261030.md).

The followup used19,200transitions in six fresh subprocesses. Capture-only control
trajectories were bit-identical through all50steps. Every condition matched before
step15. Writing the current state back at15changed next-action motion in both
restore variants and repetitions. Hidden causes remain unresolved. This isolates
a state-write effect in the tested setting, not a general failure of planning or
RL. Fresh-process prefix replay is a possible alternative, with every replayed
prefix/setup cost counted; it is not yet a validated training method. No student
was trained. [No-op result](branching-noop-results-20261030.md).

Allfourcreatedapps(includingtwozero-transitionfailedpreflights)werereadbackstopped
withzerotasks. Totalnewbranchingphysics113,920;zerooptimizer/confirmation/final
queries. Twenty-sixselectedCPUtests pass; all1,684protectedfiles and84historical
scientifictrainingrecords remain unchanged. No10xsample-efficiencyresult exists.
