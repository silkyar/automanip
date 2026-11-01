# Writing the current state back changed its next motion

The narrow followup isolates a state-write effect under the pinned simulator.
Two fresh-process capture-only runs were **bit-identical for the entire 50-step
trajectory**, including positions, orientations, joint/cube velocities, and
grasp/success flags. All six runs were also bit-identical through step 15.
At step 15, physical-only and augmented variants each wrote their current state
back without rewinding. Both variants exceeded the registered physical
tolerances on the immediately following control step in both repetitions.

| Maximum first-action discrepancy from capture-only | Physical write | Augmented write |
| --- | ---: | ---: |
| Position component | 0.346 mm | 0.346 mm |
| Rotation | 0.004084 rad | 0.004084 rad |
| Joint position | 0.0001121 | 0.0001121 |
| Joint velocity | 0.10490 | 0.10493 |
| Cube linear velocity | 0.04756 m/s | 0.04755 m/s |
| Cube angular velocity | 1.4375 rad/s | 1.4370 rad/s |

Later grasp/success flags differed in three of 64 lanes after physical writes and
two after augmented writes. Final success did not differ; this weak parent rarely
finished successful. The four comparisons per variant reuse two controls and
two intervention repetitions; they are not independent population replications.

This supports the claim that the tested write operation can perturb subsequent
dynamics. It does **not** establish whether contact caches, sleep/wake behavior,
roundoff, articulation setters or another internal detail causes it. It is not
a general finding about RL, GPU nondeterminism, or all ManiSkill tasks. The
earlier clone gate stays failed and distillation has not started.

The capture-only control establishes a possible constructive alternative on
these 64 initial states: reproduce each branch's common action prefix in a fresh
process, retaining its actual physical history instead of teleporting state.
Before using it to train a policy, validate the chosen branching procedure and
include every repeated prefix, setup and candidate rollout in compute accounting.
This may lose the sample/time advantage the original proposal assumed. Exact
controls in this one experiment are not a global reproducibility guarantee.

Work: 19,200 new environment transitions, six fresh children on one L40S function,
zero optimizer calls, no policy inference or learned parameter changes. Child
times 9.51, 6.66, 6.12, 5.76, 5.67, 6.03 seconds; wrapper 52.36 seconds. Two CPU
analysis tests passed in 0.038 seconds, including a deliberately mismatched-prefix
negative control that blocks causal attribution. No confirmation/final evaluation.

The archive contains 26 remote files /3,624,490 bytes plus exact sources/inputs.
Total new branching diagnostic physics is **113,920 transitions**, including the
preceding 94,720-step test; preceding failed preflights contributed zero. All
historical training records and frozen task/protocol remain unchanged.

[Protocol](../cumulative/experiments/branching-noop-v1.json);
[numeric result](../cumulative/results/branching-noop-20261030.json);
archive: `artifacts/cumulative/branching-noop-20261030/`.
