# Saturation is present, but the saved critic does not ask the actor to undo it

The [registered CPU snapshot diagnostic](actor-gradient-snapshot-plan-20261029.md)
used exactly 1,904 existing grasp-flagged successor observations and the frozen
256k low-alpha actor/two online critics. It ran in .669 seconds, with zero
simulator transitions, optimizer calls or protected evaluations. Safe-policy
inference agreed within 3.28e-15. Entropy differentiation, the Q/tanh chain rule
and loss decomposition agreed to at most 5.21e-18; the fixed finite-difference
direction check differed by at most 3.72e-9.

The second arm joint has deterministic action mean +.97994 and median tanh
derivative .007982. Its current critic's mean bounded-action gradient is +.53475,
so the critic locally favors increasing this already positive action. Tanh
attenuates that gradient before the Gaussian mean, but simply undoing that
attenuation would not make the same surrogate ask for the opposite command.
Joint deltas are not Cartesian lifting directions.

For 16 fixed noise draws per saved observation, the second joint's mean-space
ascent contributions were:

| Contribution | Mean across sample-state rows |
|---|---:|
| Current min-Q | +.010168 |
| Entropy | -.008136 |
| Combined | +.002032 |

The combined median is -.00503 and only 28.15% of these sample-state rows have a
positive combined component. The positive mean therefore does not describe a
typical row. Nor are these independent episodes, complete network gradients,
or Adam updates. At the gripper, Q contributes +.002813 toward increasing its
normalized setting, whereas entropy contributes -.004611; their net mean is
-.001798. It is incorrect to describe current entropy as uniformly encouraging
the observed saturation or open-gripper action.

This weakens the specific story that a useful opposing critic direction exists
and only tanh prevents following it. It does not establish that the critic is
wrong: physical counterfactual returns were not measured, and the failed lift
probe supplies no such ground truth. The snapshot also cannot explain the
historical early effect of alpha or substitute for the full replay distribution.

The accounting receipt records 1,904 actor rows, 1,904 safe NumPy policy rows,
64,800 critic-head forward rows and each autograd request's row count. All input
and source hashes are in
`research/discovery/actor-gradient-snapshot-20261029.json`. An independent review
checked algebra, signs, row accounting and these interpretation limits.

A focused literature check supports keeping action saturation, entropy and critic
quality separate. The SOP paper studies bounded-action exploration and output
normalization in MuJoCo; that is motivation for a hypothesis, not validation on
StackCube. [Wang et al., 2020](https://arxiv.org/html/1910.02208v4).
TD3 studies how approximate critic optimization can bias actor updates; our
current twin critics already use a minimum and delayed actor updates, so citing
that mechanism alone does not identify a missing remedy.
[Fujimoto et al., 2018](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf).

The more direct unresolved question remains the one raised by the successful
grasping intervention: reducing alpha changed both the actor's entropy incentive
and the critic's soft Bellman targets. A narrowly controlled route intervention
would distinguish these mediators before adopting a saturation fix or a new
algorithm. No further robot run follows automatically from this snapshot.
