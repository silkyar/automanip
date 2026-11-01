# Future action-noise control: essential test

A scalar example passed the registered bias and variance checks. This is a mechanism check, **not a robot sample-efficiency result**. No robot transitions or optimizer updates were used.

The environment sums 50 Gaussian-driven scalar actions and rewards the final position. A clipped-action variant adds model mismatch. An independent set of4,096 trajectories fits time-dependent state baselines and linear action-noise coefficients;32,768 fresh trajectories assess the gradient of shared mean and log standard deviation. Both parameters are included in variance accounting. Controls fitted using a trajectory's own future innovations are prohibited.

For a future step `u`, a coefficient predicted from history before sampling that step's Gaussian innovation gives a control `c_u = beta_u * epsilon_u` with conditional mean zero. Subtracting discounted future controls from an earlier return preserves the expectation of its raw policy-score gradient. A current-action control needs an analytic score correction. This statement does not establish unbiasedness after PPO clipping or minibatch advantage normalization.

| Dynamics | Discount | Future-only / baseline gradient covariance trace |
|---|---:|---:|
| Linear | 1.0 | 0.1441 |
| Linear | 0.9 | 0.5016 |
| Clipped | 1.0 | 0.1489 |
| Clipped | 0.9 | 0.3305 |

All four cells met the predeclared ratio≤.75 and gradient-mean agreement within four standard errors. Two deliberately incorrect implementations were detected as biased: omitting the current-action correction, and making future coefficients depend on their own innovations. The analytic linear positive control has covariance trace4,275 for ordinary state-baseline Monte Carlo and600 for future-only control; the full oracle control has zero variance. These oracle numbers use known dynamics and are not available to our robot learner.

The experiment generated1,843,200 scalar transitions, reused across the four cells; it fit200 two-coefficient value models and200 scalar control models in0.727seconds total. Three unit checks passed. [Protocol](../cumulative/experiments/future-innovation-toy-v1.json); exact source, plan and all estimates are retained under `artifacts/cumulative/future-innovation-toy-v1/`.

This is consistent with existing trajectory-control-variate reasoning, not a novel algorithm claim. [Cheng, Yan and Boots](https://proceedings.mlr.press/v100/cheng20a.html) analyze correlations along trajectories and construct recursive controls. [Tucker et al.](https://research.google/pubs/the-mirage-of-action-dependent-baselines-in-reinforcement-learning/) provide relevant negative evidence: action-dependent controls offered limited variance benefit in their tested domains, and implementation bias explained some apparent gains. Neither paper establishes a benefit for our task.

The completed feature-value experiment already showed that lower offline gradient variance need not improve online learning. Accordingly this toy does not authorize another robot training variation. A useful follow-up would need evidence that a model learned from the robot's own counted training data predicts action consequences well enough to improve the actual update, with compute and any estimation bias measured.
