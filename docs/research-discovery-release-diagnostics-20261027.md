# Release-window diagnostics before another learner change

Decision: collect narrowly scoped **training-only** evidence before choosing a
reliability intervention. Operational target is **99%+ final-step success across
multiple independent training seeds**, with protected scoring unchanged. This is
a target, not a measured achievement. Training-time reduction follows reliability;
report larger future budgets separately. The existing SAC screen keeps its
original 500k-transition/22-minute lease and exact source unchanged.

The owner declares full-horizon PPO independently confirmed and eligible as the
reference. No protected episodes, evaluator code, protocols, seed registries or
confirmation diagnostics were accessed here. CAPG remains stopped after its
declared AUC/85% gate failure. The recursive/V-trace candidate remains stopped.
There was no GPU launch, paid API use or infrastructure action in this phase.

## Observation and limits

Permitted public development summaries show control final success 121/128 with
five `released_not_stable` and two `grasped_not_lifted`; CAPG final success 118/128
with ten `released_not_stable`. CAPG at 10.24M had placement 127/128 and release 0.
These are descriptive development outcomes from one training seed. They do not
establish the cause of either release delay or final failures. In particular,
stage-ever counts do not retain alignment, velocity or remaining time at release.

The pinned public PPO has a state-independent learned diagonal Gaussian std,
ordinary per-step sampling, global latent entropy regularization and a scalar
critic. Public aggregate entropy, clipping, gradient norms and stage coverage do
not establish inadequate exploration, harmful action noise, or value-model error.
Deterministic deployment also means stochastic training noise cannot directly
explain a final deterministic rollout's disturbance without evidence of how the
mean policy was learned. CAPG changed boundary-mass likelihoods and PPO ratios,
while retaining Gaussian sampling and latent Gaussian entropy. The CAPG paper's
score-estimator variance result is not a guarantee for a clipped PPO surrogate,
nor does the negative screen establish any particular causal explanation.
[CAPG primary paper](https://proceedings.mlr.press/v80/fujita18a.html).

## Public reward audit and CPU falsification

Inspected the installed `mani_skill==3.0.0b21` public
`mani_skill/envs/tasks/tabletop/stack_cube.py`, SHA256
`ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440`.
Earlier public provenance maps this release to upstream commit
`6226567ce326625fa6eacf9aeb3f4008e4339f7e`; remote task-byte fetching failed, so
this audit establishes the installed file's identity, not independently verified
remote-byte equality. The inspected public file and
[upstream task documentation](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/stack_cube.html)
both explicitly include release and static-placement incentives. The documentation
is a newer release; the installed-file audit is the pinned evidence.

The task replaces the reward when the cube is on its target cube with
`6+(ungrasp_reward+static_reward)/2`; successful states receive 8. Normalized dense
reward divides by 8. Thus **a missing release/stability reward is falsified**.
Whether the learner visits the geometry-gated branch at the right time, and
whether it learns the relevant action consequences, remain open questions.

`research/discovery/stackcube_reward_audit.py` extracts only that public reward
method into a CPU-only process, without importing the simulator. Six constructed
inputs give:

| Constructed input | Normalized reward |
|---|---:|
| On stack, grasped, moving | .783498 |
| Same pose/velocity, released | .833498 |
| On stack, released, slower | .867536 |
| Successful release/static stack | 1.000000 |
| Off stack, grasped | .567235 |
| Same off-stack pose, released | .250000 |

All four prespecified checks pass: release helps on-stack, slowing helps on-stack,
success overrides to 1, and off-stack release can reduce reward. These constructed
conditions isolate the immediate reward function. They do not represent feasible
counterfactual robot trajectories, long-term values, or measured policy behavior.
The reward itself remains unchanged. Record:
`research/discovery/stackcube-reward-audit-v1.json` (audit 0.0239 seconds after Torch
import; the process including import completed in seconds, well below five
minutes). Zero robot transitions, optimizer updates, simulator queries and dollars.

## Smallest passive diagnostic

Implemented `research/discovery/release_diagnostics.py`; it is **not connected to
any trainer**. It accepts one complete existing 50-step training episode plus its
initial state. All outcomes are retained. No extra episode, planning query,
teacher, action substitution or new actor observation is requested.

At each actual pre-reset successor, record the existing public upstream flags
`is_cubeA_on_cubeB`, `is_cubeA_grasped`, `is_cubeA_static`; XY offset magnitude,
absolute stack-height error and cube linear/angular speed from public training
task state. Those last measurements are telemetry only: they must not be fed to
the actor/critic or turned into rewards. Preserve the actual final successor
before automatic reset and explicit true-terminal/truncation masks. Record
training seed, update index, episode index and step; these are public training
identifiers, never protected episode seeds.

The module reports each grasp→release transition, geometry immediately before
and after release, velocity at release, remaining episode steps, time from first
on-stack state to first release, later geometry loss and resumed grasping. It
also reports loss of the public upstream success predicate after first attainment.
These descriptive events distinguish **late release**, **release while outside
the stack geometry**, and **loss after an initially achieved stack**. They do not
attribute a later loss to the robot, physics, noise or another cause. Preserve
raw magnitudes alongside flags rather than introducing new success thresholds.
The output is training telemetry, never a replacement for trusted scoring.

Using already computed pre-update `V(s)`, reward, correct `V(next_state)`, raw GAE
advantage and actual minibatch-normalized advantage, compute conditional
`delta=r+gamma*(1-true_terminal)*V(next_state)-V(s)`. Group by pre-action public
phase: off-stack; on-stack/grasped; released/moving; released/static. Report
count, signed/absolute residual, raw advantage and normalization sign changes.
For the normalized-advantage field, use each row's actual minibatch in the first
optimizer epoch, before that row's first update; later epochs can reshuffle its
minibatch. This is a diagnostic convention, not a changed objective.
Episode-level aggregation is required before uncertainty estimates because steps
from one trajectory are correlated. A TD residual is noisy and bootstrapped: a
large residual is **not ground-truth critic bias** or proof of a causal bottleneck.
It identifies where a conditional value-calibration investigation is warranted.

For a later collector, additionally retain existing Gaussian mean/std and sampled
versus clipped executed action near each release, per coordinate, without new
random draws. This reveals mean/noise/saturation differences at encountered
training states without pretending they are causal counterfactual rollouts.
The analyzer intentionally does not fabricate these unavailable measurements.

Four CPU tests pass on synthetic complete episodes: placement/release coverage
versus timing, geometry at release versus later loss, correct truncation bootstrap
versus true-terminal masking, and rejection of cross-episode/nonfinite data.
Synthetic flags/scalars test bookkeeping only; they establish no robot dynamics,
failure prevalence or learning gain. Test runtime 0.001 seconds. Together with the
reward audit this phase used zero simulator transitions and no GPU.

## Mechanism selection and closest prior art

Conditional next hypothesis: a single state-independent variance vector may
poorly balance exploration before grasping with precise actions near release.
A training-only state-conditioned diagonal variance network could change that
allocation while keeping the deterministic mean actor and safe artifact schema
unchanged. It would consume only existing state 48, with independent per-step
Gaussian draws and correctly recomputed likelihood/entropy. This differs from
temporally correlated noise, whose latent temporal state would require different
likelihood treatment. It is a conventional policy-family extension, not a novel
algorithm, and **is not selected for launch on current evidence**.

Closest primary work is [gSDE](https://arxiv.org/abs/2005.05719), which combines
state-dependent exploration with periodically resampled noise, and
[adaptive Gaussian exploration](https://proceedings.mlr.press/v108/papini20a.html),
which explicitly studies learning-speed/final-policy tradeoffs. Our provisional
per-step variance proposal does not implement either full method. Likewise,
[GAE](https://arxiv.org/abs/1506.02438) already addresses critic/advantage variance;
conditional residuals alone cannot establish a new credit-assignment problem.
[Tucker et al.](https://proceedings.mlr.press/v80/tucker18a.html) show why claimed
variance mechanisms require careful implementation controls. No novelty claim.

Falsification: if release-window telemetry shows no material conditional
noise/advantage mismatch, defer this variance hypothesis. If geometry is already
lost at release, investigate its observed timing/alignment first; if geometry is
maintained but final static status fails, inspect velocity/time traces. If public
phase coverage is high while conditional prediction errors dominate, investigate
critic calibration before exploration. None of these branches justifies a
blanket reward addition or reopening CAPG.

## Smallest next paired pilot proposal — not launched

First attach the passive collector to a **future** full-horizon PPO run after
owner review; do not modify the running SAC source, dependencies or records.
Use fixed environment slots chosen before collection, all complete episodes from
those slots, no success-based selection. Log summaries at existing checkpoints
and retain at most 256 complete episodes/checkpoint to bound storage. With no new
draws/optimization/simulator steps, compare a tiny seeded instrumented run against
an identical uninstrumented run: actions, rewards, updates and final weights must
match on deterministic CPU fixtures. On GPU, document nondeterminism rather than
silently relaxing the comparison. Measure telemetry overhead separately.

If the resulting evidence supports a specific learner change, the smallest
performance pilot is one paired training seed, reference versus that single
prespecified mechanism, with random initialization and unchanged public task,
reward, horizon, state/action interface and protected scoring. For an initial
same-budget screen retain **at most 22 trainer minutes and 24.9856M transitions per
arm**, common checkpoint cadence and the owner's existing development rule.
Primary result is final-step reliability and the per-failure diagnostic change;
learning AUC/thresholds are secondary. One paired seed cannot establish 99% across
training seeds. Any apparent win needs new independent training seeds and
owner-controlled confirmation, separately budgeted. A larger-budget reliability
study would be a distinct proposal, never described as an equal-budget gain.
Even 128/128 would have a one-sided exact 95% binomial lower bound of only
`0.05**(1/128)`, about 97.69%, conditional on one trained policy. Thus the small
development screen cannot certify 99% reliability; the owner must separately
specify confirmation precision and account for variation across training seeds.

Hard proposed combined pilot spending ceiling: **$3**, including evaluation,
startup and overhead. Forty-four trainer minutes plus at most six evaluation and
ten startup/overhead minutes fit one billed resource-hour only if the owner's
live **all-inclusive** rate is ≤$3/hour. This is a conditional ceiling, not a live
price quote or launch authorization; if that quote/cap cannot be enforced, defer
or submit a separately costed plan. There is no retry allowance. No learner pilot
should begin until the narrow diagnostic resolves enough uncertainty to choose
one mechanism. Actual expenditure here remains $0.

```sh
.venv311/bin/python -m research.discovery.stackcube_reward_audit \
  --source .venv311/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop/stack_cube.py \
  --output /tmp/automanip-new-reward-audit.json
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_release_diagnostics.py' -v
```

The reward audit refuses to overwrite an output. Prior journals and experimental
artifacts are preserved. SAC source 1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca
was not edited or imported by this phase.
