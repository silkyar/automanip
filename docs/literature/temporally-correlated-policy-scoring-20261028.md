# Temporally correlated exploration requires a conditional policy score

2026-10-28. Primary-paper review and analytic derivation only; no code/probe execution, GPU calls or algorithm admission. The stopped public probe collected zero transitions and supplies no variance finding.

**Primary method.** [Korenkevych et al., ARP (2019), §5/Eq 11](https://arxiv.org/html/1903.11524) makes noise history explicit through previous states/actions. It explicitly differentiates every current-parameter mean/scale evaluation, including evaluations at historical states. Its augmented-state formulation matters; Gaussian noise marginals alone do not establish a correct policy gradient. Appendix E describes a zero-history startup transient and warns about mid-episode parameter changes. These are not automatically stationary initial samples.

**Our AR(1) derivation.** Let latent actions satisfy `u_t=mu_theta(s_t)+sigma_theta(s_t)*x_t`, with stationary standardized noise `x_t=rho*x_(t-1)+sqrt(1-rho²)*epsilon_t`. For fixed rho and recorded history, the exact conditional Gaussian is:

`m_t=mu_t + rho*sigma_t*(u_(t-1)-mu_(t-1))/sigma_(t-1)`

`v_t=(1-rho²)*sigma_t²`.

Past states/raw actions are held fixed in likelihood evaluation; historical **network outputs are not constants**. Recompute them under the candidate parameters. The score includes derivatives through both mean/scale occurrences and conditional variance. Caching the behavior policy's residual noise and detaching it during updates scores a different model. Using `Normal(mu_t,sigma_t)` ignores history and generally scores neither this conditional distribution nor the state-conditional marginal: physical states already depend on past noise.

**Two-step counterexample, derived here.** Set one shared mean theta, sigma 1, and independent standard normals e1/e2:

`u1=theta+e1`; `u2=theta+rho*e1+sqrt(1-rho²)*e2`.

Both marginals are `N(theta,1)`. Nevertheless, the joint score is

`S=e1+sqrt((1-rho)/(1+rho))*e2`,

whereas summing independent-marginal scores gives `(1+rho)*S`. For reward `G=u2`, the true derivative is 1; that naive estimator has expectation `1+rho`. In the second conditional, `d m2/d theta=1-rho`; detaching the historical mean incorrectly replaces it by 1. This is an analytic bias example, not a training result.

**PPO and bounded actions.** A history-conditioned PPO surrogate needs old/new **conditional** densities on the same observed histories. Correct conditionals do not make clipped PPO an unbiased trajectory-gradient estimator or replace full-path importance sampling. Episode boundaries and rollout cuts must preserve/reset the declared noise context; shuffled minibatches cannot discard it.

If the simulator executes `clip(u)`, retain and score the sampled raw u. Scoring the clipped action with a Gaussian density ignores boundary probability masses, and clipped history cannot reconstruct latent residuals. An invertible tanh transform instead uses its change-of-variables density; its parameter-independent Jacobian cancels in old/new ratios at fixed latent action. Neither case permits pretending the temporally correlated latent sequence is independent.

**Inherited state and critic caveat.** Future exploration depends on persistent noise/history, even when the physical observation is Markov for the plant. An exact bootstrapping critic generally needs that augmented context; a physical-state-only critic can average incompatible continuation distributions. A state-only Monte Carlo baseline remains valid because it is history-measurable; that does not make state-only GAE/TD bootstrapping exact. ARP reported better empirical stability with a state-only critic despite its augmented theory—an approximation, not proof of equivalence.

**Evidence and counterevidence.** ARP's sparse Square and UR5 reaching results favor coherent exploration at high control rates; at 125 Hz the UR5 learned in only 50% of four runs while Gaussian failed within five hours. Dense MuJoCo gains were mostly similar/slight, except Swimmer. At low rates, highly correlated noise can get stuck at boundaries; correlation needs task-specific tuning. High-rate experiments also scaled batches, so compare interactions and compute separately. This is not StackCube contact/release evidence. [Paper results](https://arxiv.org/html/1903.11524).

[Colored Noise PPO (2024)](https://arxiv.org/html/2312.11091) reports 20,480 runs across 16 tasks and prefers intermediate correlation on average; six tasks selected white noise, and pink was inferior to the best setting in nine. Its marginal-Gaussian argument does not by itself establish exact history-conditional scoring (our inference; no author-code audit here). [TD3's original experiments](https://arxiv.org/html/1802.09477), meanwhile, found no OU exploration benefit in their setting. Neither result settles correctly scored ARP on our task. Temporal correlation changes trajectory coverage and covariance, not merely a removable current-action noise term. No robot implementation is recommended or authorized by this note.
