# Action-dependent variance reduction: what the proposed probe can establish

2026-10-28. Literature/estimator review only: no new probe, training, simulator or infrastructure execution. Read Q-Prop §§3–4/appendix, Stein §§3.1–3.5/§7.4, and Mirage §§3–4, beyond abstracts. The existing CPU algebra source was inspected, not rerun.

**Primary evidence.** [Q-Prop](https://arxiv.org/pdf/1611.02247v3) subtracts a critic's first-order action Taylor approximation from the score-function residual and restores its analytic expectation gradient. Critic accuracy alone is unnecessary; useful correlation and correct correction matter. Its conservative/adaptive coefficients require scrutiny, not automatic unbiasedness.

[Stein](https://arxiv.org/html/1710.11198v4) permits nonlinear action-dependent functions through a reparameterized derivative correction, subject to integration/boundary conditions. Simply subtracting learned Q is not that estimator. Appendix 7.4 explicitly acknowledges bias from fitting the baseline on the current estimator batch. Its PPO derivation uses KL-penalized importance weighting, **not the clipped minimum surrogate**. Clipping changes the residual's active set when advantages change; an unclipped zero-expectation identity does not automatically preserve the clipped gradient. At the old policy the first unclipped gradient is a narrower case.

[Mirage](https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf) finds limited additional action-baseline benefit on its LQG/MuJoCo tests and attributes earlier gains to implementation choices: asymmetric normalization of Q-Prop residual versus correction; inadequate state baselines; and fitting/using baselines on the same samples. These are contrary evidence, not a theorem that action dependence never helps. Same-sample gating or coefficient fitting can correlate a supposedly zero-mean variate with its weight. Independent fitting/cross-fitting is needed; normalization must treat estimator terms consistently.

**Our conditional analysis.** At fixed state, write `Y=z(G-b)`, with first-action latent `u=mu+sigma*z`, frozen positive diagonal sigma, `z~N(0,I)`, and bounded action `tanh(u)`. Then z is the Fisher-whitened **local latent-mean** score. It is not the standardized bounded action or full-network natural gradient. Future policy parameters remain fixed. This measures first-decision sensitivity of the chosen finite 50-step reward return; it excludes shared-parameter later decisions and SAC's entropy objective.

Define `W=E_a tr Cov(Y|a)` and `B=tr Cov_a(E[Y|a])`. Total single-trajectory variance is `B+W`. A fixed action-only control variate cannot remove W: conditional continuation noise remains. It may reduce B, but a sampled correction adds its own noise. Across states another variance component remains. Root's synthetic bound/gating probe illustrates these limitations, not robot efficiency.

**Proposed nested design.** Two frozen existing actors × 4 public reset states × 64 first actions × 16 independent continuations × 50 steps equals **409,600 additional simulator transitions**. Baseline fitting consumes 102,400; measurement 307,200. Zero training/protected-evaluator queries does not mean zero data cost. Predeclare states, actor hashes and RNG streams; reuse rewards for gamma .99 primary/.9 sensitivity without selecting favorable results.

For each state/actor, reserve the first 16 action groups for an independent scalar baseline. The trace-optimal scalar is `b*=E[||z||²G]/E[||z||²]`, estimated there; a plain return mean is a different comparator. Freeze b before the remaining 48 groups. No measurement-driven gating, baseline selection or noise tuning. Conditional-on-fit inference is valid; baseline-fit uncertainty still matters for general claims.

For 48 measurement actions and 16 repeats, let `barY_i` be each action's mean and `S_i` its unbiased within-action covariance. Estimate:

`What=mean_i tr(S_i)`

`Bhat=tr sampleCov_i(barY_i) - What/16`.

Without the subtraction, continuation noise masquerades as action variance. Retain negative finite-sample estimates/intervals rather than silently clipping them. Variance of the grand mean is `B/48 + W/(48*16)`, not `(B+W)/768`. An ideal action-only ceiling `(B+W)/W` concerns this conditional estimator, assuming an exact correction; it is not a training speedup forecast.

**Validity, uncertainty and stopping.** Reset all 1024 slots with the same explicit public seed per state; verify observation identity and repeated-first-action successor parity before independent future noise. Observation agreement alone does not prove hidden simulator-state identity. Fail on mismatch, invalid actors, nonfinite returns or accidental shared continuation noise across action groups.

Use independent action groups as confidence-interval units, with within-group uncertainty and the 16-group fitting uncertainty handled explicitly; do not treat 768 trajectories as independent first actions. Report component/gradient-mean intervals and paired discount sensitivities. Avoid ratios with near-zero denominators and uncorrected squared-mean “signal.” Four states support conditional case studies, not a representative state-distribution covariance estimate.

Stop the action-baseline promise if continuation variance dominates and its ceiling excludes a useful reduction. Stop inference if components/gradient direction remain unresolved at this allocation; do not reinterpret noisy ratios as opportunity. Initial-state results may miss later contact bottlenecks. Even favorable B only establishes local reducible estimator variance, not a learnable control variate, PPO compatibility, successful policy improvement or 10× sample efficiency. No algorithm implementation is recommended yet.
