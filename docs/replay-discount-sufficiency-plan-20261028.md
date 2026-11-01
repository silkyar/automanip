# Does discount restoration recover early grasping?

Preregistered2026-10-28, before this candidate exists. One bounded experiment in
the user-authorized autonomous window; no broad sweep or new algorithm.

## Observation and competing hypotheses

The historical adapted SAC learned grasp128/128 and lift115/128 by102,400
transitions; the recent fast target-SAC learned grasp0/128 and lift0/128 at that
same count. Both began with identical actor bytes. The historical run took515s
to that point, versus71s in the recent one, so this is **not matched compute**.
The older recipe changed many factors: gamma.9 versus.99, target EMA.01 versus
.005, UTD.5 versus1/16, actor delay1 versus3, batch1024 versus2048,32 versus64
environments, LR.0003 versus.001, Adam beta1.9 versus.5,3 versus2 hidden critic
layers, warmup4000 versus5000 and temperature sampling timing. No causal claim
follows from that historical contrast.

At102k the old temperature was.002519 versus.2336, and old target mean3.50 versus
64.47. A fixed-policy scalar recurrence with instantly fitted critic shows that
the combined discount/EMA/update clocks could make target-backup relaxation
about160× slower in environment-transition units. Restoring discount alone
reduces that gap to16×. The .12564s CPU check used640,000 scalar assignments and
zero robot transitions/neural updates. This establishes plausible conditioning,
not actual neural targets, entropy dynamics or the cause of the robot regression.
An [independent review](replay-clock-regression-review-20261028.md) recommends
the narrow sufficiency test below, not identification of the lag mechanism.

-Horizon sufficient: changing only the fast recipe's gamma to.9 recovers much
 of the early grasp/lift behavior despite its unchanged lower update exposure.
-Horizon insufficient: the fast recipe still fails early grasp/lift; remaining
 update/temperature/representation differences deserve investigation.

## Exact intervention and budget

One fresh training seed1, fast **target-SAC only**, gamma.99→.9. All other
learning/data/initialization settings and underlying source remain unchanged.
A tiny launcher subclasses only configuration validation to admit the smaller
256,000-transition/240-process-second diagnostic budget. It calls the unmodified
`replay_robot.train` and `replay_learner.Learner`; no reward/task/curriculum change,
teacher, pretraining, state copying, planner or extra training observation.
The candidate does not import the public noise/state-copy diagnostics.

Use cached once-only frozen gamma.99 development reports at51,200/102,400/256,000
as the control; no baseline rerun or repeated scoring. Score those3 prespecified
gamma.9 exports once each through the unchanged independent evaluator. If a
checkpoint is missing because of censoring, report it as missing; do not replace
it with a favorable final point or extend the lease. No confirmation query.
Validate that initial actor bytes match the cached control before scoring.

Primary endpoint is102,400 transitions:

-Grasp≥64/128 **and** lift≥32/128 supports discount restoration being sufficient
 for substantial early-stage recovery in this seed/recipe.
-Grasp≤8/128 **and** lift≤8/128 weakens that sufficiency hypothesis.
-Other outcomes are partial/inconclusive; report both metrics without relabeling
 a borderline result as a passed gate.

51k/256k are declared context;102k cannot be replaced after seeing outcomes.
Report stack success, all stage diagnostics, alpha, actual calls/rows/time and
the complete curve too. A passed grasp gate is not a stack success improvement,
a10× learning result, a multi-seed finding or evidence of a particular mediator.
Changing gamma changes the discounted objective as well as conditioning;
temperature and policy feedback can mediate any recovered behavior.

Single L40S trainer≤240 child seconds /300 wrapper seconds plus30startup;
one CPU coordinator≤1200seconds, up to3 frozen development calls≤120seconds
plus30startup each. One$2 reservation under the shared$100 cap, no Runpod use.
Archive every source/protocol/input/report and verify stopped/zero tasks. No
automatic replication, third noise-setup attempt or next algorithm is admitted.
