# Learning when to wait did not beat a learned fixed time

The assisted stopping branch is closed. The state-dependent continuation model
failed the registered overall comparison against the strongest simple control.
No policy was promoted and no from-scratch sample-efficiency improvement exists.

| Rule | Parent 2 /128 | Parent 3 /128 | Pooled final success |
| --- | ---: | ---: | ---: |
| Fixed time learned from training outcomes | 83 | 63 | 57.03% |
| Fixed payoff-confidence threshold | 63 | 49 | 43.75% |
| Time-only continuation value | 65 | 45 | 42.97% |
| State-dependent continuation value | 71 | 60 | 51.17% |

All rules share the same acquired parent policies and supplied persistent
zero-arm/open-gripper correction. Learned rules share the payoff model and
support restriction. Only training corpus 956 entered fitting; fresh public
958 episodes supplied the assessment, with no parameter updates during rollout.
The learned fixed times were steps 37 and 27 for the two parents, chosen before
assessment using seven time bins in the training corpus.

State continuation minus learned fixed time was **−5.86 percentage points**,
98.75% block-bootstrap interval [−12.11, +0.78]. It did improve over time-only
continuation by +8.20 points [1.17, 16.02], but that partial result does not meet
the overall requirement. State minus fixed threshold was +7.42 points
[−3.13, 16.02]. Both state-rule parent groups preserved ever-grasp above 95%.

The preceding analytic optimal-stopping toy passed, and the fitted outcome
model predicted held-out correction outcomes better than simple constants.
Neither establishes effective robotic decision-making. These results reject
this particular learned stopping procedure, not every possible option learner.
No threshold, support, projection or architecture sweep follows this failure.

Compute: 51,200 public assessment transitions, zero assessment optimizer calls;
worker times 26.25 and 27.58 seconds. CPU preparation used 202 regression fits
plus two support-metric solves, **204 total linear solves**, 104,448 parent
feature forwards and 1.976 seconds. The 204,800-transition training corpus and
two 8.192M parent prefixes remain additional, previously counted assistance.
No confirmation or final holdout was queried. The Modal app was read back
stopped with zero tasks. Exact traces, inputs, sources and audits are under
`artifacts/cumulative/continuation-correction-20261030/`.

[Registered protocol](../cumulative/experiments/continuation-correction-probe-v1.json);
[numeric result](../cumulative/results/continuation-correction-20261030.json).
