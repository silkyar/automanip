# Why test a finite Monte Carlo target now?

Read October30: Schulman et al., [Generalized Advantage Estimation, version6](https://arxiv.org/html/1506.02438v6), sections2–3, especially equations14–18. GAE trades variance against error from an approximate value function. Its lambda1 limit uses sampled reward-to-go minus the state baseline; reducing lambda can lower variance while introducing error when the value estimate is inaccurate. These are established mechanisms, not a newly invented algorithm.

Our inference is narrower: a modest assisted-policy target effect justifies checking whether it survives actual from-scratch learning. It does not predict success. Complete returns can be noisier, particularly before useful trajectories become common. Their benefit can therefore disappear when policy, critic and exploration evolve together.

For the current gamma.9/lambda.9 estimator, a TD residual at lag30 has explicit weight(.81)^30≈0.00180. A sampled reward at lag30 has weight(.9)^30≈0.04239 in the MC target. This arithmetic **does not establish23.6× learning efficiency**: intermediate TD terms, critic accuracy, correlations and normalization matter.

The registered comparison replaces the whole original bootstrapped GAE target by complete50-step MC with zero terminal tail. It changes trace and boundary treatment together. It cannot attribute a result specifically to lambda or terminal bootstrap. Both arms retain ordinary PPO clipping, minibatch advantage normalization and uniform timestep sampling. Thus neither is claimed to be an unbiased gradient procedure for the discounted start-state objective; the paper distinguishes its discounted gradient approximation from that objective.

See [registration](../../cumulative/experiments/scratch-credit-v1.json) and [research journal](../autoresearch-journal-20261030.md). The criterion is measured learning curves with actual compute, not target loss or novelty.
