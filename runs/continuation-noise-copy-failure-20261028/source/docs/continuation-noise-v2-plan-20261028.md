# Explicit state replication for the public noise diagnostic

2026-10-28. New reviewed protocol version, not an automatic retry of the first
attempt. The [original plan](continuation-noise-probe-plan-20261028.md) remains
unchanged and archived with its failed run. Its replication guard stopped before
any env.step:0 transitions/optimizer/scorer calls; maximum initial-observation
spread1.83884. It provides no variance or learning result.

Pinned source explains the failure: reset seeds one Torch RNG for the vector
batch, while StackCube draws different tensor rows for positions/quaternions.
Repeating a seed list reproduces the batch, not identical environments. This is
a simulator-specific setup finding, not a general RL principle.

The scientific question, two frozen actors, four public seeds,64 first actions,
16 continuations, independent16/48 action-group baseline/measurement split,
50-step reward-only returns, primary gamma.99/.9 sensitivity, estimators and
interpretation limits remain as in the original plan. No new policy training,
algorithm, protected evaluation or promotion is admitted.

The only new setup uses the pinned simulator's public state API **for diagnostic
replication**. At each prescribed reset, capture the first slot's registered
physical state and copy it into all1024 slots. Reset the joint-delta controller
buffers to the copied articulation state. Verify copied physical state and policy
observations; after the shared first action verify each16-way successor group.
Choose each first action using the canonical slot0 mean/variance, broadcasting
the resulting action identically within its16-way group. Subsequent policy
actions use each slot's own observation. This avoids introducing first-action
differences from tolerated physical scene-coordinate roundoff.
Select slot0 regardless of its contents; there is no search for favorable states,
success-conditioned reset, fitted model, planner or candidate learning.

This changes the previous prohibition on state restoration explicitly. The
probe must record all additional state reads/writes/observation reads, not call
them free training data. Expected totals on completion:

-8 reset calls,16 physical-state read calls,8 batched state-copy writes,
 8 controller resets,8 extra observation reads;
-8192 diagnostic episodes,400 vector step calls,409,600 public transitions;
-0 policy/critic optimizer calls,0 protected evaluator queries.

State dicts contain registered dynamic actor/articulation local poses and
velocities; static geometry is recreated by the unchanged task. The pinned
`set_state_dict` applies GPU state and refreshes articulation kinematics. The
joint-position controller reset refreshes its start/target position buffers;
each actual step then assigns fresh drive targets through the normal controller.
Task elapsed steps are already zero after reset. This does not promise perfect
replication of unexposed PhysX solver internals.

Predeclared tolerances account for GPU scene-coordinate roundoff:1e−5 maximum
absolute physical-state/observation difference before actions,1e−4 maximum
absolute successor-observation difference within first-action groups. These are
finite-tolerance replications, not mathematical equality. They are orders of
magnitude below the observed unreplicated spread. Record all residuals; stop
without a scientific result if either gate fails. Do not loosen them after
outcomes or launch a third setup version in this window.

The state API, scene API and both controller modules were downloaded from the
same pinned upstream commit and matched installed local bytes; runtime will
verify all4 hashes before copying. A CPU data-structure check passed (.02799s):
replication preserves source data, uses independent cloned rows and rejects
nonfinite/wrong-shape/unexpected fields. This is not a GPU API test.

Use a distinct one-attempt output volume and launch record. Preserve all rewards,
first/future Gaussian noises, policy hashes, source/protocol, validity residuals,
query counts and failures. Same500-second process /550-second function+30startup
caps and$2 maximum reservation; original app must be archived and stopped before
root admits this version. Root admission also requires independent source review.
No unattended retry or further training follows from its output.
