# Does a smaller initial entropy weight recover early grasping?

Registered2026-10-28 after the gamma-only result, before candidate training.
This continues the same historical replay-regression investigation, not a new
algorithm sweep. The gamma.9 intervention produced grasp0/lift0 at102,400 and
grasp4/lift0 at256,000, so its predefined sufficiency gate failed. The original
gamma.99/alpha1 fast control scores remain cached and must not be repeated.

The historical SAC reached grasp128/lift115 by102k, with alpha.002519. The fast
control's alpha was.2336, about93× larger. Its much lower actor/temperature update
exposure can prolong an initially strong entropy incentive. However, high alpha
also changes soft critic targets, and the old recipe spent far more compute.
This evidence does not isolate exploration from value-scale/target effects.

A2k-update scalar temperature check used the fast recipe's actual2,029-update
clock, fixed mean logprob−1, target entropy−8 and identical Adam. It produced
alpha.234076 from1 versus.001170 from.005. The final ratio remained.005000002.
Cost:2.0603 CPU seconds /4,058 scalar optimizer calls,0 neural/robot calls.
Entropy and actor feedback were fixed, so this does not predict robot learning.

## One-factor intervention

Start from the **gamma.99 fast target-SAC control**, not the gamma.9 candidate.
Change exactly one learner line: initial alpha1→.005. This value is the order
observed at the historical51k checkpoint, not fitted to confirmation data.
The learned target entropy−8, temperature optimizer, update cadence, architecture,
all other optimizer settings, replay, data, reward, task and initialization seed
remain unchanged. Initial actor/critics must be identical; only log-alpha differs.
No demonstrated/pretrained/teacher or public-probe data enters training.

H1: the initial temperature transient is sufficient to explain substantial
early-stage regression. H2: lower alpha is insufficient or harmful because
remaining update/critic/coverage issues dominate; it may remove needed exploration.
This tests sufficiency, not a unique mediator or a general exploration principle.

One fresh seed1,256,000-transition /240-child-process-second ceilings. Independent
frozen development exactly51,200/102,400/256,000 once per unique policy hash. Reuse
cached control reports and safe-policy/global claim infrastructure. No confirmation,
baseline rerun, resampling, censored-final substitution or automatic extension.

Primary102k gate, fixed before any candidate outcome:

-Grasp≥64/128 and lift≥32/128 supports substantial recovery in this recipe/seed.
-Grasp≤8/128 and lift≤8/128 weakens temperature-start sufficiency.
-Other results are partial/inconclusive. Do not choose a later endpoint instead.

Report stack success, full early curves, every stage and alpha/Q/action statistics.
Passing a grasp/lift gate is not successful stacking,10× efficiency, robustness
across training seeds or permission to promote a method. Losing the gate stops
this intervention; no extra training until a new causal case is reviewed.

The candidate image bakes `replay_learner_lowalpha.py` under the canonical import
name `replay_learner.py`; this mapping is explicit in its plan. The rest of the
core actor/robot code is byte-identical to the cached control. A tiny launcher
only admits this smaller diagnostic budget and records the alpha intervention.
Root must verify the exact one-line diff, paired initialization and first-target
algebra on CPU, plus independent source review, before admission.

Resource envelope: one L40S trainer≤240 child seconds /300 wrapper+30startup,
one CPU coordinator≤1200seconds, up to3 frozen development calls≤120+30startup
each; one$2 reservation under shared$100 and remaining autonomous window. No
Runpod spending. Archive and stopped/zero-task closeout required.
