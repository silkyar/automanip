# Branching validity before planning and distillation

The next mechanism question is whether simulator search can produce useful
action targets which a policy retains without the planner. First establish
whether the existing pinned simulator supports sufficiently reproducible
branches. The previous repeated-reset prefix design failed its matching gate;
that result remains valid and is not evidence that explicit restoration works.

This diagnostic compares uninterrupted trajectories with repeated restoration
of the same exposed state. Two existing weak 8.192M-transition parent policies
generate fresh training-only trajectories. No hand-designed release/hold option
is applied. Their acquisition costs remain disclosed; this is not from scratch.

At control steps 0, 15, 30 and 40, capture the official physical state, exposed
controller buffers, drive targets/velocities, generalized forces, elapsed clock,
and random generators. Replay the original executed float32 action tape to step
50, using either physical-state-plus-clock restoration or the augmented snapshot.
Three repeats, two parents, and 64 lanes require exactly 94,720 new transitions.
One L40S worker, no retries, 300-second function ceiling, zero optimizer calls.

The augmented method passes only if **every** replay matches its uninterrupted
reference: componentwise position <=0.1 mm; rotation <=0.001 rad; qpos <=0.0001;
qvel <=0.01; cube linear velocity <=0.01 m/s; angular velocity <=0.05 rad/s;
identical grasp and stock-success flags at every post-action step. All assigned
lanes count. Initial restoration errors and later dynamics errors are separate.
Tolerances will not be relaxed after looking at the results. Preserve raw traces
even on failure. No claims about restoring hidden contact/solver caches.

If the augmented method fails, diagnose it before distillation. If it passes,
register the small planning-to-distillation test, keeping all new interactions
within the proposed one-million-transition ceiling including this diagnostic.
The planner and distilled student must be scored separately against the original
policy and an equal-cost ordinary-rollout learning control. No learned terminal
value, reverse curriculum, new controller, or action-chunk architecture yet.
Historical task/evaluator/protocol/results remain unchanged. No private evaluation.
