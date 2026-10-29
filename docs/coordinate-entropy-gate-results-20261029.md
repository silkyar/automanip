# Coordinate-entropy explanation did not pass its CPU gate

No action coordinate met the [registered blind-spot criterion](coordinate-entropy-gate-plan-20261029.md).
On the same1,904 saved grasp states, expected scalar entropy pressure averaged
+1.28969: gradient descent on the temperature loss would raise alpha on these
inputs. Only29.52% of states had a pressure to lower it. The saturated second
joint had coordinate pressure+2.95036, but the fraction simultaneously asking
that coordinate's alpha to rise while scalar alpha fell was26.89%, below the
fixed75% gate. All eight coordinate gates failed.

Fixed32/64-point Gauss-Hermite quadrature agreed within7.41e-8. No Monte Carlo
episode independence or historical optimizer behavior is inferred. These are
conditional signals from one frozen actor on selected old training states, not
the actual replay/Adam trajectory. Existing logged low-alpha training also shows
temperature falling to about.001484 at133,120 transitions and rising afterward;
it is incorrect to claim it never responds to concentration.

The exact unsquashed quadratic negative control also passed: curvatures100/1,
equal joint entropy2.837877, optimal variances.1/10 versus equal1/1. Expected
costs are10 versus50.5, a5.05ratio. This mathematical counterexample does not
model StackCube, but prevents treating equal coordinate entropy as a general
improvement. Accurate value functions can rationally allocate less exploration
to a sensitive action. Grasp closing may be such a precision-sensitive action.

Cost:.06075s diagnostic timer,1,904 actor rows and1,462,272 analytic squash
integrand evaluations; zero fitting, robot interactions or protected scoring.
Receipt:`research/discovery/coordinate-entropy-gate-20261029.json`. No GPU
coordinate-entropy method, architecture change or promotion follows.

The [independent primary-literature review](literature/scalar-temperature-coordinate-entropy-review-20261029.md)
also identifies prior work on selective entropy allocation and warns that equal
noise weights do not imply equal entropy or equal effective variance. No novelty
or robot-efficiency claim is made here.
