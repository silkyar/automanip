# AutoManip: scientific diagnosis and one proposed mechanism test

2026-10-28. **For review; no new GPU training, simulator probes or sweep launched.** The objective is at least 10× fewer environment interactions to 80–90% success against strong matched baselines, with total compute and all assistance accounted for. The [37-paper literature map](literature/survey-20261028.md) broadens the candidate mechanisms beyond PPO.

## What the existing experiments establish

The [ledger](experiment-ledger.md) now contains 61 completed scientific training runs across 29 families: 60 RL runs and one BC fit. The difference from the strategy message's 59-run snapshot is the already-completed, subsequently reconciled control/critic pair. No extra training was launched to reconcile its scoring records.

| Observation | Evidence | Scientific interpretation and limit |
|---|---|---|
| Reliable stacking is achievable | Official 25M PPO: 32.23% mean confirmation; tuned full-horizon 25M: 94.08%; constant 50M: 95.90%, each across three training seeds | Establishes a useful baseline and evaluation pipeline. Most gain is recipe/task alignment; doubling the tuned budget buys about 1.82 percentage points. It does not establish a learning mechanism with dramatic interaction savings. |
| Learning can start late | One strong L40S control: 0/128 at 5.12M, 98/128 at 10.24M, 111/128 at 15.36M, 117/128 at 20.48M | These sparse points cannot distinguish discovery, credit, retention or target-quality bottlenecks. This is one seed, not a population learning curve. |
| Training seeds differ substantially | The gamma .95 family gives 10/128, 92/128 and 32/128 at the same 25M budget | Large variance makes a promising single-seed screen unreliable. It does not identify the source of variance. |
| Useful cloned behavior was lost during PPO | Shared BC actor: 49/128; both PPO continuations: 0/128 after 5.12M new transitions | Strong observed failure with a controlled initialization, but assisted and single-seed. We did not capture enough intermediate task outcomes to date the loss or identify its cause. |
| Lower noise also changed optimization | Residual-noise PPO executed exactly 100 joint updates over 100 rollouts; logged approximate KL .218–6.013 versus .1 limit. Standard-noise arm executed 22,782 updates | The supposed exploration-only intervention also changed actor displacement and critic fitting. The pre-update stop does not undo an accepted large step. Both variants failed; KL alone is not a causal explanation. |
| More value fitting was insufficient | From-scratch 10.24M control 98/128, critic-completion variant 29/128; 31,269 versus 51,200 total optimizer calls | Rejects the tested extra-value-fitting recipe. Does not refute PPG, corrected replay or the fresh-critic explanation specifically in BC continuation. |
| An adapted SAC screen did not solve stacking | 0/128 after 262,432 interactions and 1,292 seconds; 129,216 learner rounds | An expensive adaptation under a time cap, not a strong step-matched refutation of off-policy learning. We still lack a strong modern replay baseline under our exact task contract. |

Sources: [precision confirmation](stackcube-precision-confirmation-results-20261028.md), [BC pilot](stackcube-bc-exploration-results-20261028.md), [reconciled critic pair](../runs/stackcube-kl-value-screen-20261028/reconciled-summary.json), [SAC screen](stackcube-sac-results-20261027.md), and exact ledger source records.

## Why many runs produced little sample-efficiency progress

**The selection objective mostly rewarded terminal quality.** Fixed 25M/50M endpoints, gamma/entropy/reset families and quality replication answered whether a recipe eventually works. They were useful baseline work, but did not directly optimize or explain the experience needed for early competence. Later short-budget variants were often compressed recipes rather than tests of a mechanism that changes information gained per transition.

**An experiment family often changed more than its label implied.** Noise calibration changed effective KL and update counts. Early actor stopping also stopped critic fitting. Environment count, minibatch size, collection cadence and optimizer history interact. A single visible configuration change can create multiple causal pathways. Labeling it “exploration” or “more critic learning” hid the variables we needed to measure.

**The cheap tests were not always diagnostic of the robot bottleneck.** Toy improvements established that a component can work somewhere, not that StackCube fails for the same reason. An action-MSE improvement did not predict closed-loop retention. A stronger gate needs a predicted signature in real training data, a competing explanation and a negative control before robot-scale commitment.

**Evidence did not accumulate around one unresolved question.** Many changes were screened once, followed by another family. Negative results should have narrowed a causal model. Instead, broad algorithm changes sometimes replaced the question before its key observation had been explained. The failed SAC adapter and critic completion are particularly narrow negatives; they cannot eliminate the broader replay literature.

**Comparators and accounting were not consistently strong enough for mechanism claims.** Different hardware, assistance, budgets and optimizer counts prevent simple headline comparisons. Infra work, approvals and scoring reconciliation also reduced throughput, but more pods would not fix weak experimental questions. The independent evaluator and preserved failures are strengths; a private final holdout and multi-seed early learning curves are still missing.

## Define the 10× denominator correctly

The 50M recipe is a high-quality endpoint, not evidence that 50M transitions are necessary for 80% or 90%. In the one strong control curve above, the first observed threshold crossings bracket **T80 in (10.24M, 15.36M]** and **T90 in (15.36M, 20.48M]**. Temporary crossing and sustained competence are also different; sparse snapshots cannot establish sustained thresholds.

For that seed only, a candidate reaching 80% by 1.024M or 90% by 1.536M would be a sufficient tenfold improvement against the lower ends of these brackets. This is an illustrative conservative bound, not a preregistered project threshold or a multi-seed claim. Before any efficacy campaign, freeze threshold/cadence rules and collect adequately resolved curves for strong matched baselines. Report censoring when a seed never crosses. Count diagnostics and evaluation queries separately from acquisition steps, with both included in total compute.

## Strongest opportunity and selected question

The broad opportunity is **learning more from rare useful experience without corrupting its targets or destroying the acquired behavior**. Replay, corrected multistep targets, imagination and policy-preserving objectives offer different routes. The literature does not pick the route for us. [REDQ](https://arxiv.org/html/2101.05982), [Retrace](https://papers.neurips.cc/paper_files/paper/2016/file/c3992e9a68c5ae12bd18488bc579b30d-Paper.pdf), [PPG](https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf), [MBPO](https://arxiv.org/html/1906.08253).

The next diagnostic should answer: **Does PPO initially move a useful actor too far in a useful direction, or does its estimated direction already damage behavior?** The BC failure supplies an existing nonzero-skill starting point and a striking optimizer signature. It is a cleaner causal starting point than diagnosing a flat from-scratch learning curve. It is not yet evidence that retaining cloned behavior solves from-scratch sample efficiency.

Competing hypotheses:

- **H1, displacement:** low Gaussian variance plus adaptive optimization produces an excessive mean-policy step. A small forward step is useful; the full step destroys behavior.
- **H2, direction/targets:** noisy or biased advantages, a fresh critic, or reward/terminal-success mismatch produce a locally harmful direction. Merely shrinking the step preserves behavior without improving it.
- **H3, collection/coverage:** stochastic rollouts already leave the BC policy's support before updating; poor recovery coverage limits learning. This can coexist with H1/H2 and must be measured before the update.

Existing [CPU algebra/probe](research-discovery-gaussian-kl-20261028.md) shows why H1 is plausible: equal-variance Gaussian KL is one half the sum of `(mean displacement / standard deviation)^2`, and fresh Adam steps need not scale down with gradient magnitude. But that probe used synthetic advantages, not actual robot GAE, and also exceeded the KL limit under its default-noise condition. It is **not** a reproduction of production collapse. These are known optimizer/trust-region effects, not a new learning principle. [Adam](https://arxiv.org/pdf/1412.6980), [PPO](https://arxiv.org/pdf/1707.06347), [TRPO](https://proceedings.mlr.press/v37/schulman15.pdf).

## One decisive experiment: fixed-direction response at the first update

This is one controlled mechanism test, not five training recipes. Its claim is limited to the initial update; an absent effect rejects that initial-step explanation, not every possible later accumulation mechanism.

**Prerequisite, before new robot collection:** reconstruct the production first-update calculation locally with fixed finite arrays. Verify equality with the unmodified update for parameters, optimizer state and advantages; verify exact Gaussian KL, interpolation endpoints and executed action clipping. These checks are proposed, not already run. If reconstruction fails, stop at the discrepancy rather than run robots.

**Fixed initial condition:** the existing production BC actor, original fresh critic/optimizer recipe, and residual-calibrated initial noise. This deliberately studies the assisted failure; disclose the 49,971,200-step teacher, 25,600 demo transitions and 5,000 BC updates. No additional teacher data or policy pretraining. The frozen task, rewards, horizon and protected evaluator remain unchanged.

**Collection:** use three preregistered public training collection seeds, each with one 1,024-environment × 50-step rollout: 153,600 new transitions total. These are three data draws from one existing actor, not three independently trained policies. Measure stochastic stage progression and final success before updating. Use actual sampled actions, rewards, masks and GAE; do not assign teacher rewards to newly sampled actions.

**One direction per collection seed:** reproduce the first ordinary joint Adam update and extract its actor-mean displacement Δ. Hold critic and noise fixed for the response branches so the deterministic task comparison isolates mean movement. Log all joint-update changes separately. Choose 64 KL-check episodes with no rows in the prescribed first fitting minibatch, using indices only, before inspecting task outcomes. If fewer are available, stop rather than change the ordinary proposal's fitting data. The KL-check states are diagnostic training data, independent of the later task-probe episodes.

**Five branches from exactly the same starting actor:** no update; full +Δ; the largest passing positive fraction on the grid `2^(-k), k=0,…,12` with held-out mean-policy KL ≤.01; a positive fraction on that grid with KL ≤.001; and the negative counterpart of the latter. If full +Δ already meets a cap, reuse identical branches rather than enlarge Δ; if no fraction passes, record a failed displacement gate. Fractions use analytic KL only, never task success or reward. Record both KL directions, per-state tails and action saturation: average latent KL does not bound all executed-action changes.

**Public diagnostic rollouts:** evaluate all branches deterministically on the same 64 fresh public training probe episodes per seed, retaining the full 50-step horizon. Count every branch rollout: at most 48,000 additional transitions, giving **201,600 total new simulator transitions**, including collection and probes. No privileged state restoration, branch simulator queries, protected development or confirmation episodes. Reuse common reset seeds, not simulator snapshots. Record paired stock final success, original dense return and stage histories; these are diagnostic training probes, not independent benchmark claims.

**Decision rules fixed before outcomes:**

- Support H1 as an initial-step explanation only if full +Δ loses at least 10 percentage points versus no-update, the bounded nonzero forward step recovers at least half that loss, and small-forward minus small-reverse dense return has a positive paired 95% interval. Direction of the main effects must agree in at least two of three collection seeds. Treat episodes as paired within seed; report seed-specific intervals and cluster-aware aggregate uncertainty. Three seeds provide limited population inference.
- A bounded step that only matches no-update without evidence of a useful forward direction is **retention without demonstrated improvement**; it does not pass the mechanism gate.
- If small-forward is reliably worse than small-reverse, reject the locally useful-direction prediction and investigate target/credit or objective alignment. The original dense return and terminal-success response can disagree; report both rather than select the favorable metric.
- If the full first step causes no material loss, reject it as a sufficient explanation for the observed collapse. If uncertainty spans the stated effects, report inconclusive. Neither outcome triggers an automatic follow-up sweep.
- Poor stochastic collection before any update supports a coverage/noise concern, but does not prove it is the only cause. This test cannot identify which component of H2 is wrong, or explain a collapse that only develops much later.

**Resources proposed for review:** one L40S worker, at most 600 container seconds, at most five CPU minutes for reconstruction, at most 201,600 new transitions. Record actual gradient calls/rows, inference, KL checks, wall time and provider charges; initial placeholder dollar envelope $1, to be checked against the provider rate before any submission. Stop on a cap without filling missing outcomes with extra runs. No resources are reserved or launched by this document.

If supported, this warrants a separately reviewed from-scratch retention/improvement method comparison against competent PPO and replay baselines, with equal interaction and total-compute reporting. It does not warrant a tenfold claim. That requires multi-seed T80/T90 gains, explicitly charged data/model/teacher costs, and held-out-task validation. If rejected, refine the causal account around targets or coverage using the recorded evidence before selecting another algorithm.
