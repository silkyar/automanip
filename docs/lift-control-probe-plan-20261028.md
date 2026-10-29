# Can the recovered grasp support a deliberate upward motion?

Registered before new robot actions,2026-10-28. User explicitly authorized
continuous autonomous work until asked to stop; the earlier4h limit is revoked.
The existing shared$100 Modal cap/no new Runpod spending remain. This admits
only a separately reviewed **assisted mechanism diagnostic**, not a learned
method, policy promotion or training sweep.

## Evidence and competing hypotheses

The low-initial-alpha policy scored121/128 grasp but0 lift/stack at256k. In its
aligned64 recent training episodes,56 had sustained grasp signals without3cm
lift. Hand and object moved together mainly horizontally; drops were secondary.
Joint commands are not Cartesian directions, and contact flags do not establish
load-bearing grip strength. It is premature to add a lift reward or declare that
the arm cannot move upward.

- H_command: the learned arm command is the main immediate barrier; a controlled
  upward motion with the policy's gripper setting can raise the cube.
- H_grip: the observed contact is not reliably load-bearing; closing the gripper
  is necessary in addition to an upward motion.
- H_dose: the proposed upward controller does not deliver sufficient actual TCP
  motion. A failed lift then cannot answer the grip question.

A public-URDF FK/Jacobian gate passed on6400 saved poses, with maximum position
error6.73µm, orientation2.50µrad, finite-difference Jacobian error2.21e−10.
Its5mm upward step is kinematically feasible but says nothing about physical
tracking/contact. The gate's serialization failure and corrected repeat were
preserved and counted. [Gate and limitations](panda-fk-diagnostic-gate-20261028.md).

## Fixed2×2 public intervention

Use the fixed safe deterministic256k low-alpha actor, weights
`98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a`.
Eight declared fresh public seeds93284000–7;8 parallel environments; four
conditions;50 steps each =1600 explicitly counted extra public transitions.
All data are diagnostic-only and excluded from any policy update.

| Condition | Arm after eligibility | Gripper after eligibility |
|---|---|---|
| A | Frozen policy | Frozen policy |
| B | Frozen policy | Close, action−1 |
| C | Analytic upward servo | Frozen policy |
| D | Analytic upward servo | Close, action−1 |

A first records baseline actions and identifies each lane's first run of5
consecutive official grasp signals, with eligibility no later than step25.
Its next pre-action state is the intervention start. A lane with no eligible
state is retained and never replaced. All conditions reset the same8-seed
**vector batch**, not duplicate rows. Replay A's recorded actions before that
lane's fixed intervention time; after it use the declared condition and normal
closed-loop observations. No state-copy API, state search or new seed is used.

Verify initial and pre-intervention matching: positions≤1e−4m, arm/finger
qpos≤1e−4 in their units, qvel≤1e−2, corresponding quaternion rotation-angle
differences≤1e−3rad. Record all residuals; if any matched lane fails, preserve
the invalidity and report no paired causal conclusion. Observation agreement is
not a proof of identical hidden solver state. This is a new replay-prefix
intervention, not a third attempt at the stopped nested-noise setup.

The analytic controller uses the explicitly disclosed public Panda URDF and
task-defined base transform, **structural model assistance**. At eligibility,
target the current TCP x/y/orientation and TCP z+6cm. Each step uses a geometric
6×7 Jacobian, translation-error vector capped at15mm, rotation-error vector
capped at.05rad, and damped least squares
`dq=J^T*(J*J^T+.05²I)^−1*error` with unit row weights. Joint targets stay.01rad
inside URDF limits; normalized deltas are clipped to the unchanged±.1rad action
range. The ordinary controller/physics still applies them. Record requested
and achieved motion; no force model, contact planning or adaptive tuning.
Live analytic FK must remain within the original1e−4m/1e−3rad gate against
observed TCP poses. A helper extracted from the checked equations must pass
CPU parity before admission. Count every analytic FK/Jacobian/IK evaluation.

## Measurements and falsification

Report all8 lanes and eligibility separately, never only successful cases.
Primary mechanical outcome: cube rises≥3cm from the intervention-start height;
also report the existing reset-relative lift diagnostic and final/ever grasp,
TCP rise, finger positions, target tracking and drops. A delivered upward dose
means TCP rises≥4cm from its intervention start. No StackCube success-rate or
learning-efficiency estimate follows from these selected public conditions.

At least4 matched eligible lanes are needed for directional interpretation.
If C delivers upward dose and cube rise in≥75% of eligible lanes while A does
not, this supports H_command in these states. If D does so and C does not,
gripper closure is implicated under the same upward controller. If D delivers
the TCP dose but mostly fails cube rise, its load-bearing grip/contact geometry
is suspect under this intervention. If TCP dose fails, classify H_dose/uncertain;
do not diagnose grip failure from it. B is the declared grip-only contrast and
may expose closed-loop policy response; it does not isolate a pure static force.
Use exact counts, not population claims from8 cases. No endpoint replacement,
resampling, repeated run, controller tuning or subsequent training is automatic.

Assistance includes the existing256k/195.43s own-policy prefix and public URDF;
it must not be hidden as a from-scratch learner improvement. Task, reward,
50-step horizon and independent evaluator stay unchanged. The hybrid controller
is not exported to or accepted by the evaluator. No protected episodes/credentials
or evaluator package are available to the public diagnostic worker.

One L40S process≤270s /function≤330s+30startup,1600-transition pre-step cap,
one$1 reservation under the shared cap. Root code/pin/budget review and an
independent review precede admission. Preserve raw observations/actions/rewards,
validity/dose/failure records, sources and policy hashes; verify stopped/zero
tasks. No new training or protected evaluator call is admitted.
