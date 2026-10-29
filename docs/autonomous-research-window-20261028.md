# Autonomous research window,2026-10-28

User mandate:3–4 hours from16:32:33UTC, continuing through status questions;
shared Modal cap$100, no new Runpod spending. Goal remains at least10× fewer
interactions to80–90% success against strong matched baselines. This journal
records mechanisms and negative evidence, not a leaderboard improvement.

## Completed: replay moment scope

The [preregistered study](replay-moment-study-plan-20261028.md) tested joint
current/successor normalization against split normalization and ordinary target
SAC after CPU mathematical/source tests and an execution smoke. Three fresh
seed1 runs took586–587 process seconds each and1,881,024 transitions in total.
All14 independent early development scores were0/128. No replication gate or
confirmation query followed. [Full result](replay-moment-results-20261028.md).

The useful distinction is that numeric stabilization is not useful learning.
Joint values stayed relatively controlled but rarely grasped. Split normalization
has an exact warmup per-head mean constraint; a fixed-policy scalar control
test produced true-Q errors6.6–7.0 despite a small self-TD residual. Joint avoided
that pathology but did not beat the competent target-network toy control, and
its action-gradient errors were worse in all3 toy seeds. An independent robot
source review found no concrete scoring/export defect that excuses the zeros.

## Checked: policy-gradient noise and temporal correlation

Primary reading expanded to Q-Prop, Stein control variates and the contrary
Mirage analysis. [Review](literature/gradient-variance-followup-20261028.md).
A cheap analytic/Monte Carlo probe showed why action-only corrections cannot
remove continuation noise, and how same-sample gating introduces bias. A second
synthetic gate verified the nested-variance estimator, independent baseline
calibration and within/R correction. These are estimator checks, not learning.

The attempted public nested-rollout probe produced **no valid variance data**.
Repeated seeds did not replicate vector slots; a separately reviewed physical
state-copy version passed physical parity but failed its derived-observation
tolerance. Both stopped before any env.step, were archived and verified stopped.
There was no third setup attempt. All setup/state queries and provider time are
recorded in the [execution report](continuation-noise-execution-20261028.md).

A related [temporal exploration review](literature/temporally-correlated-policy-scoring-20261028.md)
showed that stationary Gaussian marginals do not justify independent-action PPO
likelihoods. A two-decision CPU check reproduced a gradient-component sign error
under naive scoring of correlated noise. Correct AR policies need historical
mean/scale derivatives and appropriate critic context. This rejects a tempting
implementation shortcut; no correlated-noise robot learner was launched.

## Stronger historical clue: the fast replay recipe lost early competence

Re-reading existing development reports revealed that old SAC already grasped
128/128 and lifted115/128 by102,400 transitions, while fast target SAC grasped
0/128 at the same count. Their initial actor bytes are identical. The historical
run spent515s there, versus71s recently; this is not matched compute. More actor
and critic work, different gamma/EMA/architecture/optimizer and temperature
timing are confounded.

At102k the historical run had49,216 actor/critic updates and alpha.002519. The
fast one had6,088 critic/2,029 actor updates and alpha.2336. Its target mean was
64.47 versus3.50. In an exact scalar idealization, changing UTD, gamma and EMA
together slows the target backup's interaction-relative relaxation about160×;
restoring gamma alone leaves16×. That is a recipe-clock plausibility check, not
a reconstruction of neural learning or proof of the causal mediator.

Following [independent review](replay-clock-regression-review-20261028.md), one
[gamma-only sufficiency screen](replay-discount-sufficiency-plan-20261028.md)
was admitted: fast target SAC gamma.99→.9, one fresh seed1,256k/240s ceilings,
unchanged underlying learner/task,3 once-only frozen development points. Primary
102k grasp≥64/128 and lift≥32/128 supports substantial recovery;≤8 each weakens
sufficiency; otherwise partial/inconclusive. Cached gamma.99 control scores are
reused. No final-stack,10×,general principle or replication claim follows from
passing this stage gate. **Completed:**256k transitions in151.354 process seconds.
The primary102k grasp/lift were0/128;256k grasp4/128, lift0/128; stacks0/128 at
all3 points. The predefined gate weakens discount-alone sufficiency. Artifacts,
3 trusted reports and exact sources were archived and app stopped/zero-task
verified. This does not identify temperature, update exposure or architecture
as the remaining cause.

## Further cheap check: constant values versus useful gradients

The [offset diagnostic](value-offset-probe-20261028.md) used the same controlled
scalar system with a +1 reward shift, a value-coordinate change and an explicitly
oracle-initialized positive control. All12 fits/12,000 optimizer calls completed
in8.14s total CPU process wall time. Value errors rose to15.7–15.9, while
action-gradient RMSE rose only18–32%. Its prespecified interference gate was
partial/inconclusive. A fixed coordinate change matched the raw shifted model
to numerical precision; the informative initialization matched the unshifted
residual problem. Neither supports claiming that a fixed centering trick alone
improves learning. [Primary-method distinctions](literature/value-offset-normalization-review-20261028.md).

A default-denied initial-temperature-only sufficiency test is prepared from the
original gamma.99 fast control, with alpha1→.005 and no other learner change.
CPU paired actor/critic/target initialization and first-target algebra passed;
independent source review was interrupted by model capacity and has one bounded
retry pending. It has not been submitted at this update. No initial-value oracle
or constant-shift toy data would enter that robot learner.

## Accounting and preserved scope

Before the gamma screen, provider-reported total shared-session spend was
$13.53572685; its$2 cap is reserved separately. Metering may lag. Earlier teacher
and training histories remain preserved, not charged as free assistance. No
public diagnostic data entered a policy update. Task/evaluator/seeds/interfaces
remain frozen; publication figures/video are separate historical artifacts.

CPU diagnostic process times: moment algebra.0792s; scalar critic fits17.3927s
including the explicitly repeated measurement correction; archived-snapshot
inspection1.5064s; action-noise algebra.02843s; nested estimator gate.24131s;
AR-score check.08639s; state-copy data-structure gate.02799s; scalar replay-clock
check.12564s. Receipts preserve synthetic rows/updates and limits separately.
These are process timers, not total research or import/wall time.
