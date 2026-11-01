# Randomized parallel intervention on grasp states

Registered as a **new unpaired randomized diagnostic**, not a repeat or revision
of the failed replay-prefix experiment. That experiment remains failed: identical
measured resets/commands did not maintain the required physical match. Its512
transitions and all raw evidence remain counted. No tolerance is loosened and
no old episode is replaced. This protocol must receive independent review and
new exact-source/budget admission before any robot query.

The unresolved scientific question is unchanged: does the frozen low-alpha
policy fail to lift mainly because of its learned arm command, an insufficiently
closed/load-bearing grip, or failure of the proposed upward controller to deliver
motion? The crossed entropy-route and coordinate-entropy gates do not answer it.

## Design and assignment

One batch of128 environments,50steps, exactly6,400 public transitions maximum.
Fresh declared public seeds93304000–93304127, one vector reset only. Assign32
lanes to each A/B/C/D using
`default_rng(93305000).permutation(repeat(arange(4),32))` **before reading any
observations**. Save assignment and algorithm before the reset. Assignment is
independent of task state; do not select/rebalance by observed positions, grasp
or success. ManiSkill's batched seed semantics are retained, not claimed to make
identical rows or one-to-one independent RNG streams.

All128 lanes execute the same fixed safe policy before eligibility. First five
consecutive official grasp flags, completed by elapsedstep25, define that lane's
next-action intervention start. The condition does not affect its own actions
before eligibility. Eligibility is therefore a pretreatment variable under the
simulator's independent-subscene semantics. Retain every ineligible lane and
every failed attempt; no replacement, repeat, state copy, paired-state claim,
prefix replay, extra reset or extra queried seed.

| Group | Arm after eligibility | Gripper after eligibility |
|---|---|---|
| A | Frozen policy | Frozen policy |
| B | Frozen policy | Close/action−1 |
| C | Analytic upward servo | Frozen policy |
| D | Analytic upward servo | Close/action−1 |

The actor is exactly weights
`98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a`.
Upward controller and public Panda URDF are unchanged from the reviewed paired
protocol: own intervention-start TCP x/y/orientation and z+6cm; translation cap
15mm, rotation cap.05rad, damping.05, unit Jacobian row weights, targets.01rad
inside limits, normalized joint deltas divided by.1/clipped±1. Frozen action,
observation, horizon, physics, task and reward interfaces remain unchanged.
Live FK agreement must pass1e-4m/1e-3rad throughout. Save all observations,
actions, rewards, grasp/success/termination flags, targets, eligibility,
assignment, attempted/actual counts and analytic query counters incrementally.

## Outcomes and limits

For every lane report ever/final grasp, eligibility time, cube height relative
to reset and intervention start, TCP tracking/rise, finger positions and drops.
Keep all32 assigned-lane reset-relative outcomes as well as eligible-lane
denominators. Save first threshold-crossing times, cube rise and grasp flag at
the first delivered TCP dose, and drop sequence: separate maximum heights are
not by themselves proof of secure simultaneous load-bearing. Report eligibility
times/remaining exposure; do not rebalance chance differences or select only
post-treatment dose responders.
Primary mechanical lift is cube rise>=3cm from intervention start. Delivered
dose is TCP rise>=4cm. Report exact numerator/eligible denominator and Wilson
95% working-Bernoulli intervals per group, plus all32 assigned lanes; intervals summarize these
cases and are not population generalization from one vector batch/one policy.
They are not guaranteed95% causal/randomization coverage. Thresholds below are
directional mechanical screens, not proofs.

At least8 eligible lanes in **each** group are required for directional
interpretation; otherwise no directional gate passes. C with delivered dose
and lift in>=75% versus A lift<=25% supports an immediate learned-command
barrier in these selected states. If D delivers dose+lift in>=75% but C lift<=25%,
with C dose>=75%, closure is implicated under the same upward servo. If D
delivers dose>=75% but lifts<50%, load-bearing/contact geometry remains suspect.
If D dose<75%, report insufficient dose and do not diagnose grip failure. B is
the grip-only policy-response contrast, not an isolated static-force experiment.
Mixed cases are inconclusive. Report numerical differences and uncertainty
without an individual counterfactual or exact paired effect. No endpoint change,
tuning, extra sample, training continuation, protected scoring or promotion.

This is an assisted mechanical experiment: public kinematics plus the existing
own256k-transition/195.43s policy prefix are disclosed. Randomization improves
identification of average intervention effects; it does not turn the hybrid
controller into a learned method or establish10x learning efficiency.

Resource envelope: oneL40S worker, child<=270s/function<=330s+30startup,6,400
pre-step transition cap; one$1 reservation within the same user-authorized total
$100Modal cap. No Runpod. Worker receives only the fixed safe policy, three
safe-policy reader files, own diagnostic/controller helpers, public task runtime
and read-only physics cache. No evaluator/benchmark-seed package or credentials.
Archive all results, independently verify STOPPED/zero tasks, then close the cap.
