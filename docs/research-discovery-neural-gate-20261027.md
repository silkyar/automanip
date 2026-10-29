# Neural replay gate — 2026-10-27, version 1

**Decision: stop this recursive-replay candidate branch before a robot pilot.**
The neural toy reproduces an established multi-step optimization tradeoff, with no
consistent final accuracy advantage over a competent four-step control. No GPU
pilot or spending is proposed on the strength of this result. The first journal
and tabular result remain unchanged.

## Observation and mechanism

The first gate found that backward update order helped live tabular targets but
failed completely with frozen targets. The next question was whether explicit
backward target recursion helps a nonlinear critic under stochastic dynamics and
policy shift, beyond known n-step returns, with target freshness held fixed.

The candidate is deliberately **standard V-trace**, not a claimed invention.
[IMPALA / V-trace](https://proceedings.mlr.press/v80/espeholt18a.html) is the closest
prior work. It truncates trace coefficients while correcting current TD residuals
with importance weights. The latter weights are untruncated in this test, so the
target policy is preserved. A raw recursive-target improvement over one-step TD
would not establish novelty.

## Prespecified small test

Implemented in `research/discovery/neural_credit_toy.py` before reading results.
Six paired toy seeds, four methods, two fixed policy shifts, one shared data set
per seed, no adaptive settings or reruns:

- A continuous action `a` lies in [-1,1]. Behavior is uniform. Target density is
  `pi_k(a)=(1+k*a)/2`, with `k=0` or `.75`. The exact target/behavior density ratio
  is `rho=1+k*a`, in [.25,1.75] for the shifted case. Both have full common support.
  Target action mean is `k/3`; behavior mean is zero. This is a genuine distribution
  shift but not a learned or changing actor.
- Twelve steps: `x_next=.9*x+.2*a+Normal(0,.03^2)`. Initial x is uniform on
  [-.25,.25]. Only the final transition has reward
  `1-(x_next-.4)^2+Normal(0,.2^2)`. Gamma=.95. The toy includes stochastic dynamics,
  continuous state/actions, and noisy delayed outcomes; it has no contacts.
- Each seed collects 128 complete trajectories, 1,536 transitions. All failures
  and successes are retained. Success diagnostic `abs(x_final-.4)<.15` is never
  used for sampling or fitting. Data contain 17–25 successes and 103–111 failures
  per seed. Exact data digests are in the result.
- Critic input is `(elapsed_fraction,x)` and its architecture is 2→32→32→1 with
  tanh hidden layers. All four methods use identical initialization, Adam at .002,
  300 gradient updates, batch size 256 and the **same minibatch index stream**.
  Targets for the entire stored data set are recomputed every 20 updates from
  that learner's current critic and frozen during those 20 updates. No candidate
  gets a fresher target, extra optimization steps, or live reverse SGD targets.
- Diagnostic V is analytic under the known target policy using exact mean and
  variance propagation. It is evaluated on a common 12×41 grid (all phases,
  x in [-.6,.6]), and **never used as a training label**. Some grid states are
  uncommon in behavior data; this is a common coverage/generalization diagnostic,
  not an on-policy task-success score.

Let `delta_t=r_t+gamma*Vbar_{t+1}-Vbar_t`, terminal Vbar=0.

1. One-step: `G_t=Vbar_t+rho_t*delta_t`.
2. Four-step: per-decision IS TD-residual sum through at most four transitions,
   `G_t=Vbar_t+sum_j gamma^(j-t)*(product_(i=t..j) rho_i)*delta_j`.
3. Full-episode: the same known per-decision IS target, through episode end.
4. Recursive V-trace: `G_t=Vbar_t+rho_t*delta_t+gamma*c_t*(G_(t+1)-Vbar_(t+1))`,
   `c_t=min(1,rho_t)`. Residual rho is never truncated. Only future trace products
   are truncated; this differs from clipping residual rho, which could change the
   fixed-point policy. No trace crosses a reset or terminal boundary.

On-policy, methods 3 and 4 are mathematically equal. That equivalence is an
explicit negative control, including a unit test. Off-policy, candidate trace
truncation trades propagation length against variance; it is not a new mechanism.
Density ratios are resolved for **this toy**. This does not solve density ratios
for arbitrary deterministic robotic actors or old PPO data.

**Prediction:** recursive targets might reduce critic error earlier than one-step
targets, while truncation might protect against full-episode IS variance.
**Falsification gate:** no advantage against the competent four-step control at
matched optimizer/data budgets, or apparent gains explained entirely by already
known multistep-return behavior. The toy is a mechanism screen, not an algorithm
benchmark; six seeds do not justify precise significance claims.

## Outcomes

Final analytic-grid value MSE after **300 equal gradient updates**, mean ± across-
toy-seed SD:

| Method | Same target/behavior policy | Shifted target k=.75 |
|---|---:|---:|
| One-step | .005907 ± .001207 | .005553 ± .001073 |
| Four-step IS | .005081 ± .001159 | .006111 ± .001540 |
| Full-episode IS | .007300 ± .002336 | .009048 ± .002498 |
| Recursive V-trace | .007300 ± .002336 | .006329 ± .001990 |

The candidate and full-episode control coincide on-policy to numerical precision,
as predicted. Under shift, V-trace beats full-episode IS in 5/6 seeds, consistent
with an established variance reduction role. But it beats four-step IS in only
2/6 seeds, with mean final MSE **higher by .000218**. It also beats one-step in only
2/6 shifted seeds, mean final MSE higher by .000776. This weakens a general claim
that carrying endpoint information farther necessarily improves a neural critic.

Mean MSE over the 15 common optimizer checkpoints, updates 20–300:

| Method | Same policy | Shifted policy |
|---|---:|---:|
| One-step | .055780 | .071209 |
| Four-step IS | .020199 | .025955 |
| Full-episode IS | .008596 | .010418 |
| Recursive V-trace | .008596 | .022135 |

Long returns do accelerate early **fixed-data value fitting**. Full-episode IS is
best by this history-average diagnostic, despite its worse final MSE. V-trace has
a modest history-average gain over four-step in the shifted condition, but its
final error does not improve. This is an optimization/variance tradeoff, not a
reliable Pareto improvement. These history averages are **not** environment-step
learning AUC, robot success, or evidence of using fewer experiences.

One fixed data size was used. No actor was trained and no policy threshold was
measured. Therefore there is **no experience/sample-efficiency claim**. A robot
pilot is not warranted for this candidate based on this gate. Keep the tested
target kernels as conventional reference implementations; do not promote a new
algorithm or start a hyperparameter sweep to rescue this branch.

## Compute, reproducibility and tests

Existing `.venv311` Python with torch 2.6.0, CPU explicitly selected, two threads.
No dependencies installed. Experiment: 3.93 wall seconds, 7.08 process CPU seconds,
48 critic fits, 14,400 gradient updates and 3,686,400 minibatch presentations total.
Data: 9,216 toy transitions across 768 toy episodes, shared across methods/shifts.
Zero actor gradients, robot/simulator queries, robot transitions, robot resets,
teacher/demonstration/planning data, GPU seconds, paid API calls or dollar spend.
The analytic oracle is toy diagnostic information, not information given to a
robot learner. Full per-seed curves, data/code digests and accounting are saved in
`research/discovery/neural-credit-results-v1.json`.

```sh
.venv311/bin/python -m research.discovery.neural_credit_toy --output research/discovery/neural-credit-results-v1.json
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_*.py' -v
```

Nine discovery tests pass: four original counterexamples plus five neural tests
for on-policy equivalence, independent forward/backward off-policy expansion,
episode boundaries, analytic Bellman identity under target shift, and unfiltered
failure coverage. The optional neural tests skip when torch is unavailable.

## Updated decision and next hypothesis

The method idea was narrowed through prior-art checks into V-trace and did not
clear the competent-control accuracy gate. **Stop it as a discovery candidate.**
No robot pilot proposal or new GPU budget is submitted. The earlier 20-credit
campaign outline remains unapproved and is not automatically activated by these
tests. The pending 94% candidate remains unconfirmed here; no confirmation data
were accessed.

The next useful action is a diagnostic-driven direction choice using public
training rollouts: distinguish insufficient early state coverage from a poor
conditional value model, rather than inferring either from a long episode or
short eligibility trace. Ordinary off-policy learning remains a necessary strong
baseline, independent of this rejected recursive-target proposal.

Public artifact clarification from the trusted owner also removes an unnecessary
restriction in the first proposal: a two-hidden-layer tanh actor with final tanh
can be expressed by existing schema hidden widths `[h1,h2,8]`, putting the final
action tanh in the third hidden layer followed by an 8×8 identity linear output.
Thus a standard tanh-output TD3 actor need not be forced to use a linear mean.
This toy did not implement/export an actor; any future use must test exact action
parity through that construction. No protected loader inspection or change is
needed, and hidden ReLU would still require separate contract review.
