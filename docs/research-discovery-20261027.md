# AutoManip discovery journal — 2026-10-27

Status: candidate-side literature and CPU falsification phase complete. No robotics
training or evaluation was run by this agent. No GPU, infrastructure, credentials,
confirmation seeds, evaluator sources, benchmark protocol, or private holdout were
accessed. The pending 94.0% candidate remains **unconfirmed in this journal**.

## Mandate and architecture

Optimize transitions to reliable 80/85/90% final-step success, against the strongest
independently confirmed reference, not the original weak PPO recipe. A tenfold gain
is a research target, not an expectation. A reference that has not reached 90%
within its measured budget supplies no finite denominator for a claimed 10x
improvement at 90%. Report it as censored; do not extrapolate.

Keep candidate implementation in `research/discovery/`, a standalone training
entrypoint per learner, small reusable replay/target kernels, plain JSON
configuration, and append-only decisions/results. Do not add a learner framework
to the existing package yet. Candidate trainers produce the **existing safe actor
artifact** through a trusted export review; the evaluator never imports these
modules. Preserve the current clipped-linear-output, tanh-hidden-layer MLP schema.
No changes to evaluation, physics, success, horizon, observation/action interface,
or protected artifact loader are proposed here.

Training can use a different critic, replay and objectives; actor inference must
remain representable by the current artifact. Standard TD3's final output tanh is
not silently interchangeable with the current clipped linear output. An initial
TD3-family control must document that adaptation and test saturation gradients.
SAC/FastTD3 results from other domains do not establish performance here.

Read public evidence: the two StackCube reports, public seed-1 comparison and
mechanism audit, historical public aggregate results, and pinned official PPO
source. Starting repository commit observed:
`97215850fbcc192c4ca7be5878b48e9fb066322e`.
The CPU result records its own exact source digest and Python version.

## Hypothesis 1 — Preserve experience, not just network weights

**Observation.** Public half-budget PPO control reaches 12.2% mean at 12.49M
transitions. Full-horizon seed 1 is only 8.6% at 10.24M, then 88.3% at 15.36M.
This suggests a learning takeoff worth understanding; it does not identify its
cause. The official PPO implementation has no persistent replay buffer. It uses
50-step rollouts, independent Gaussian noise and updates until epochs or KL stop,
then discards that rollout as training data. High late value explained variance
does not measure whether early rare transitions were learned efficiently.

**Mechanism.** Replay permits more learning from a fixed set of simulator
transitions, including early useful episodes. A twin-critic off-policy control
also exposes whether a putative new mechanism merely recovers established sample
efficiency. More gradient computation is an explicit cost, not free experience.

**Closest prior art / novelty.** [TD3](https://proceedings.mlr.press/v80/fujimoto18a.html)
uses twin critics, delayed actor updates and target-policy smoothing.
[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) supplies a strong stochastic
off-policy alternative. [REDQ](https://arxiv.org/abs/2101.05982) studies high update
to data ratios with critic ensembles; [FastTD3](https://arxiv.org/abs/2505.22642)
combines parallel simulation and large-batch TD3-family learning for humanoids.
Replay itself and these methods are established. The proposed contribution is
currently **a fair local control and mechanistic investigation**, not novelty.

**Prediction.** A competent replay learner moves the initial rise left on the
transition-indexed curve; compute per transition rises. If it changes only
throughput, it is not a sample-efficiency mechanism.

**Falsification.** At matched observations/reward, seeds and budgets, no reduction
in transitions to a predeclared threshold or AUC versus the confirmed anchor;
high critic overestimation or behavior-value mismatch despite increasing updates.
One failed untuned TD3 run does not falsify off-policy learning generally.

**Small test / outcome.** The tabular test below establishes what data reuse can
and cannot recover. It is not a TD3 or StackCube experiment. A neural toy with
recorded actor outputs, twin critics and exactly the permitted output clamp is
the next implementation gate before simulator smoke tests.

**Next decision.** First build a minimal off-policy control outside the frozen
modules, with reset-boundary-safe replay and full accounting. Do not jump from a
PPO-only comparison to a discovery claim.

## Hypothesis 2 — Recursively propagate endpoint information through stored episodes

**Observation.** Success-ever substantially exceeds final success in public
diagnostics. Full-horizon training improves final development success, suggesting
post-acquisition state coverage matters. Early takeoff might also depend on
propagating information from rare useful trajectory suffixes. Neither observation
proves delayed-credit failure; the reward is already dense.

**Mechanism.** Retain complete ordinary training episodes, including failures.
Test backward recursive critic targets against uniform one-step replay on exactly
the same stored data and update count. A recursive target can communicate a
suffix's information beyond one target-network edge per update. Simply reversing
minibatches with frozen targets cannot do this. Avoid success-only value fitting,
and keep a uniform-data control. Multi-step off-policy targets require an explicit
correction/truncation design, not uncorrected Monte Carlo targets on arbitrary old
behavior. A deterministic actor makes naive density-ratio Retrace inappropriate;
resolve that before proposing a robotics candidate.

**Closest prior art / novelty.** [Episodic Backward Update](https://proceedings.neurips.cc/paper_files/paper/2019/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html)
already propagates episode values backwards. [PSER](https://arxiv.org/abs/1905.12726)
propagates replay priorities through predecessors. [Retrace](https://arxiv.org/abs/1606.02647)
addresses safe multi-step off-policy returns. [RUDDER](https://arxiv.org/abs/1806.07857)
instead redistributes delayed returns. The raw backward-replay idea is a
rediscovery, not a new algorithm. A future uncertainty/contact-aware extension
would still require a much closer novelty check and component ablations.

**Prediction.** Against uniform replay, recursive targets reduce critic learning
lag on observed rare suffixes before any robot performance gain; improvement
should survive matched gradient steps and disappear if recursion is disabled.

**Falsification.** A benefit exists only with changing target freshness, extra
updates, outcome selection or overoptimistic value estimates; or no benefit
against ordinary n-step/return targets on the same data. Learning only an endpoint
classifier without better policy learning is also a null robotics result.

**Small decisive tests / outcome.** `research/discovery/credit_toy.py`, standard
library only, completed in roughly 0.25 CPU seconds on Python 3.13.7. For one
50-transition deterministic trajectory (reward only on the last transition,
gamma=.9), Bellman updates to recover at least 90% of true initial value:

| Target used within replay sweep | Forward | Shuffled, 256 toy seeds | Reverse |
|---|---:|---:|---:|
| Live online value | 2,500 | mean 1,463.87; SD 136.89 | 50 |
| Frozen snapshot | 2,500 | 2,500 | 2,500 |

All orders share identical data. This is an **optimization-order effect in a
deterministic tabular problem**, not a 29x or 50x robotics sample gain. The frozen
target null is decisive against a batch-order-only implementation.

In 256 toy data sets of 64 Bernoulli outcomes at true probability .2, uniform
estimation averages .20013 (across-seed SD .04915), while success-only estimation
is 1.0. Selecting observed successful outcomes changes the estimated quantity;
this does not refute [SIL](https://arxiv.org/abs/1806.05635), which has a different
actor-critic objective. It rejects using success-only replay as an unbiased value
estimator without correction or coverage.

Finally, a 50-step two-branch exact enumeration at gamma=lambda=.9 recovers an
expected action-gap fraction 1.0 with an exact critic versus .005726 with a zero
critic. This follows [GAE](https://arxiv.org/abs/1506.02438): the coefficient of a
remote residual alone is not proof of expected advantage loss. Thus “the trace
decays exponentially, so replace GAE” is unsupported without conditional critic
error measurements.

**Next decision.** Do not implement reverse batch ordering as the candidate.
Before GPU work, test a recursive target in a stochastic continuous-action neural
toy, against uniform one-step and n-step replay, with target freshness and gradient
counts held fixed. Require unbiased failure coverage and off-policy stress tests.

## Hypothesis 3 — Temporally coherent, state-dependent training exploration

**Observation.** Entropy improved public PPO results, but current exploration
samples independent noise each control step. We have no public measurement of
noise correlation, coverage, or whether the initial learning delay is exploration
rather than credit assignment. Saturation alone is not evidence of bad exploration:
the better full-horizon policy has high gripper saturation.

**Mechanism.** Coherent exploration can produce sustained interventions instead
of mutually cancelling joint commands. Re-sampled state-dependent noise leaves
the deterministic actor and evaluation interface unchanged. A learned local
controllability model is a possible later extension using the same stream; do not
assume local smooth dynamics remain valid at contact.

**Closest prior art / novelty.** [gSDE](https://arxiv.org/abs/2005.05719) already
provides periodically resampled state-dependent exploration. Thus coherence is
not novel. Controllability-weighted exploration would also need comparisons to
[controllability-aware skill discovery](https://proceedings.mlr.press/v202/park23h/park23h.pdf)
and model-based exploration before any novelty claim.

**Prediction.** More grasp/transport coverage early per transition, with equal
per-step noise energy and no persistent loss of release/stability coverage.

**Falsification.** Only higher noise energy explains coverage, coherent noise
reduces fine-control success, or endpoint coverage improves without learning-curve
gain. Correlation introduces latent exploration state, so ordinary PPO per-step
likelihood ratios must not silently be assumed valid.

**Small test / outcome.** No experiment run. Lower priority until rollout
coverage and action autocorrelation establish an exploration bottleneck. Use the
off-policy learner for this first, where correlated behavior noise does not
require pretending the behavior policy factorizes per step.

**Next decision.** Defer GPU trials. This is a conditional next hypothesis, not
another dimension in a predetermined hyperparameter sweep.

## Proposed smallest research campaign — review required, no spending yet

1. CPU: minimal replay learner and neural stochastic toy; one-step versus ordinary
   n-step versus recursive targets. Verify replay handles reset boundaries,
   terminal/truncated states and failed outcomes. Test output-clamp gradients and
   exports against the existing public schema. Run a small simulation smoke only
   after the trusted owner approves integration; no evaluator modifications.
2. Preserve/reproduce the strongest confirmed PPO reference. Request trusted
   transition-indexed development evaluations on a common preselected cadence;
   candidate agent receives aggregate development outcomes only. Parent owns the
   pending promotion and confirmation decision. Do not query confirmation for
   screening or create a private holdout as part of this candidate phase.
3. If CPU and smoke gates pass, screen the PPO reference, uniform-replay control,
   and one recursive-target candidate with seed 1, at most **660 trainer seconds
   each**, max 12.4928M transitions. Do not interpret an unreached threshold as
   numerical time to success. Abort only crashes/invalid artifacts immediately;
   scientific futility requires the predeclared development-curve criterion.
4. If replay is viable and the candidate beats the uniform control at matched
   transition checkpoints, run the candidate and appropriate strongest comparator
   on **three paired training seeds**, at most **1,320 trainer seconds each** and
   24.9856M transitions. The screening seed is disclosed as adaptive; report the
   two additional seeds separately. A proper independent multi-seed replication
   can be proposed later instead of pretending this is already confirmatory.
5. If recursive targets win, ablate recursion and compare conventional n-step
   replay at equal update count before a scientific improvement claim. If only
   ordinary replay wins, preserve it as the new competent baseline and change
   direction. No transfer claim until the method is frozen and tested on another
   manipulation task without task-specific retuning.

Proposed common checkpoint targets (all multiples of 51,200): 0, .512M, 1.024M,
1.536M, 2.048M, 2.56M, 5.12M, 10.24M, 15.36M, 20.48M, 24.9856M. This gives early
resolution for the ambitious 2.5M target. The trusted owner should review cadence,
episode counts, reliability rule, promotion gate and any resulting protocol
addition **before** the campaign; this journal does not amend the protocol.
At 128 development episodes per checkpoint, 50 steps each, all 11 checkpoints
cost 70,400 separate evaluation transitions per run, logged separately. Bracket
crossings between checkpoints; no linear interpolation presented as measured
time to threshold. Report AUC with specified integration and censored curves,
80/85/90% crossings, per-seed outcomes and lower tail, then fresh confirmation
only on the chosen candidate checkpoint at its recorded training budget.

Budget ceiling proposal: **20 Modal credits**, not authorization to spend.
Screening has at most .55 trainer GPU-hours; six followup runs at most 2.20, total
2.75. Reserve .75 GPU-hours for evaluation and .50 for image/startup/retry/smoke,
giving **4 GPU-hours maximum**. At actual total billed rate `r` credits/GPU-hour,
estimated cost is `4r`; proceed only if the quoted hardware/CPU/memory costs fit
20 credits and impose a hard spend stop. This is not a live price quote; USD and
credit conversion must be obtained by the infrastructure owner. Actual spend in
this phase is **$0, zero paid API calls, zero GPU hours**. Existing Runpod jobs and
Modal integration are unaffected. No unrelated jobs may be stopped to fund this.

Every run must log environment transitions (screen max 37.4784M total; followup
max 149.9136M total), simulator calls, resets/episodes, actual optimizer steps,
minibatch presentations, actor/critic gradient steps, replay age, target update
count, training seed, initialization, actor hashes, code/config digests, software
pins, hardware, simulation/inference/update/reset/evaluation time, wall clock,
GPU-hours and provider cost. Zero teacher, demos, planning lookahead or pretrained
weights. Reusing previous-policy trajectories across runs would be extra data and
requires a separate track; within-run ordinary replay does not create transitions.

## Reproduce this phase

```sh
python3 -m research.discovery.credit_toy --output research/discovery/credit-results.json
python3 -m unittest discover -s tests -p 'test_discovery_*.py' -v
```

Four meaningful mechanism/counterexample tests pass. The result file contains
full toy seed counts, compute accounting and source digest. No claim is made that
these CPU checks validate a robot policy, a learning algorithm on ManiSkill, or
the security of a local single-user repository.
