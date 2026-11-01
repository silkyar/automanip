# Can global entropy pressure hide a concentrated action coordinate?

CPU-only gate after the crossed entropy-route tests. Use exactly the existing
1,904 grasp-flagged successor observations and frozen low-alpha256k actor;
zero fitting, simulation or protected evaluation. No GPU method is admitted.

For the diagonal Gaussian followed by tanh, integrate each normalized action
coordinate's expected log density using fixed32- and64-point Gauss-Hermite
quadrature. The Gaussian part is analytic. Verify convergence within1e-4 and
report both orders; do not tune quadrature after seeing the result.

Compare the scalar controller's conditional signal `sum Elogpi_i -8` with the
hypothetical per-coordinate signals `Elogpi_i -1`. A coordinate blind spot is
descriptively present on this selected state set if >=75% of states have both
positive coordinate signal and negative scalar signal, with the corresponding
coordinate and scalar averages having those signs. These are directions of the
instantaneous loss gradient on these inputs, not actual historical Adam steps,
full-replay estimates, independent episodes or proof the scalar objective is
wrong. Scalar constraints intentionally allow unequal allocation.

Before proposing an equal-per-coordinate controller, evaluate an analytical
negative control: accurate quadratic Q, two Gaussian actions with curvatures100
and1, fixed joint entropy, unconstrained means at the optimum. Optimal scalar
entropy allocation uses variance inversely proportional to curvature. Equal
coordinate entropies give expected task cost5.05times larger at the same joint
entropy. Compute this and verify both joint entropies and optimality conditions.
If this negative control holds, a blind-spot observation alone does **not** pass
a gate for a general equal-per-coordinate remedy. A future method must explain
how it gains useful coverage under critic error without sacrificing this case.
The unsquashed quadratic is a deliberate mathematical counterexample, not a
StackCube model or a robot-performance result. All analytic evaluations count
as extra CPU compute; they are not extra environment experience.
