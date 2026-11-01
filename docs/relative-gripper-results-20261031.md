# Correct coordinates and cumulative displacement unlocked search labels

| Training-only oracle coverage | Parent 2 /64 | Parent 3 /64 | Total /128 |
| --- | ---: | ---: | ---: |
| Original absolute quiet gripper proposals | 0 | 1 | 1 |
| Anchor-relative targets | 11 | 8 | 19 |
| Current-relative displacement proposals | 50 | 39 | 89 |

The primary current-relative variant passed the preregistered coverage screen.
All prefix, repeated-continuation, paired-arm-command and first-action matching
checks passed. The first relative action is identical for anchor/current modes;
later integration distinguishes them. Eight proposals per variant use identical
noise and arm commands, without any geometry/stage-conditioned action selection.

The original gripper normalization maps a to 0.015+0.025 a metres. Only 5.47%and
2.73%of absolute quiet proposals initially requested an opening relative to the
measured finger; relative proposals did so 53.13%and 49.22%of the time. Median
first displacement changed from−3.44/−3.16 mm to+0.092/−0.043 mm. Thus centering
removed a strong closing bias. Current-relative proposals then accumulated
small motions, producing much more final-success coverage than fixed anchored
targets. This supports a local sampler-coordinate explanation; it is not a
general learning principle or an improved policy score.

The procedure uses acquired 8.192 Mparent policies and a fixed step 30 search point.
Both assistance and the generic action proposal prior are disclosed. The oracle
selects among simulated futures in training; it cannot be deployed as a policy.
Successful current-relative trajectories supplied student labels only after
the primary fidelity/coverage screens passed.

Work:115,200 newtransitions, zero optimizer/confirmation/final queries. Two L 40 S
wrappers 151.34 and 164.76 seconds;148 remote files/48,757,074 bytes plus exact inputs
and sources archived. App read back stopped with zero tasks. Cumulative
branching physics at this point 344,320. Four actuator-toy/conversion tests and
two analysis tests passed before interpreting the result. Historical task,
evaluator, policy schema and benchmark records remain unchanged.

[Protocol](../cumulative/experiments/relative-gripper-search-v1.json);
[numeric result](../cumulative/results/relative-gripper-20261031.json);
[policy-compression experiment](readout-distillation-plan-20261031.md).
