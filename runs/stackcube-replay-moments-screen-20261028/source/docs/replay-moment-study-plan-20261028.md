# Autonomous window: replay-moment mechanism study

Preregistered before robot outcomes, 2026-10-28. User authorized autonomous
thinking/experimentation from16:32:33 to20:32:33UTC, without requiring feedback.
Shared Modal total cap remains $100 (last reported12.04770834 spent); no new
Runpod spending. This is one mechanism study, not an unrelated algorithm sweep.

## Question and competing explanations

Can replay-based control learn useful StackCube behavior with substantially less
experience while remaining within a600-second training-process budget? More
specifically, does using common normalization moments for current/replay and
next/current-policy inputs materially improve learning over separating those
moments in an otherwise identical target-free critic?

The prior SAC screen is weak negative evidence: it used1024-row batches and
.5 critic/actor updates per transition (512 replay rows/transition), requiring
129,216 learner rounds for262,432 transitions. It did not produce stacks within
its time cap. It is not a refutation of replay or a competent modern algorithm
comparison. The assisted first-update probe established a retention concern,
not a useful direction or from-scratch efficiency result; no assisted data is
used in this study.

- **H1, moment scope matters:** joint normalization preserves distinctions lost
  by independent half-batch moments; its critic leads to better early policy
  learning than the otherwise identical split control.
- **H2, target-free machinery is insufficient:** joint/split critics behave
  similarly or both fail, indicating this mechanism is not sufficient in this
  adaptation/budget. Neither outcome refutes published CrossQ generally.
- **H3, ordinary replay suffices:** target-network SAC matches/beats both, so
  purported gains should not be attributed to normalization.

All are predictions, not explanations already established for our robot.

## Prior art and essential cheap gate

[CrossQ paper](https://arxiv.org/html/1902.05605v4) and
[author code](https://github.com/adityab/CrossQ/tree/a318a266679bec3f9d93c44e029860a6da077c89)
are pinned at commit`a318a266679bec3f9d93c44e029860a6da077c89`. Critic normalization
precedes the first dense layer and follows each hidden ReLU; actor training uses
critic running statistics without updating them. The source's finite-epsilon
BRN expression is distinguished from textbook BRN. Critic-only normalization is
an explicit safe-actor adaptation. No algorithm novelty is claimed.

The [CPU gate](../research/discovery/replay-moment-probe-20261028.json), run in
.079 seconds with zero interactions/optimizer calls, demonstrates shifted-batch
aliasing, gradient coupling and the finite-epsilon caveat. This proves an
operator distinction, not reliable targets or robot improvement. See the
[independent review](research-window-replay-review-20261028.md).

A direct port additionally failed a constant-feature gradient check (NaN at
zero variance, both warm and postwarm). The common implementation now leaves
sqrt forward values unchanged and defines its otherwise singular zero-boundary
derivative as zero. Six CPU tests pass, including this negative condition,
shared pooled-EMA clock, detached Bellman target with retained cross-moment
gradient, paired initialization, critic-evaluation immutability and safe actor
export. This safeguard is implementation hygiene, not a new learning method or
a claim of bitwise JAX parity.

## Fixed comparison

Three arms, first training seed1:

| Arm | Critic training |
|---|---|
| target-sac | Two ordinary critics, Polyak target networks, no normalization |
| joint-moments | Target-free critics, one concatenated current/next pass, joint moments |
| split-moments | Same target-free pass, each half uses its own moments; pooled running-stat update once |

Joint versus split is the causal comparison. Both use the same update rule and
clock for pooled EMA, not two sequential EMA updates. Their deeper running
buffer values can differ because normalized activations differ. The ordinary
SAC arm is an adapted replay comparator, not a verbatim CrossQ or universal
best-tuned SAC baseline. Avoid method-wide claims from this initial screen.

Fixed common settings: actor two256-wide tanh layers with squashed mean; critic
two256-wide ReLU layers, two Qs; gamma.99, Adam learning rate.001/beta1.5,
policy delay3, initial alpha1 with learned temperature/target entropy−8;
BRN momentum.99 in the Flax old-weight convention, epsilon.001,
warmup100,000 **critic updates**, correction caps r3/d5. Target SAC tau.005.
All start from paired random initialization; no teacher/demos/pretraining,
learned dynamics, planner, state restoration or extra observation inputs.

Vectorized replay adaptation:64 environments, collect128 transitions then eight
critic updates, batch2048. That is UTD1/16 and128 replay rows per new transition,
not the paper's UTD1/batch256. It is fixed across the three arms to keep compute
competitive. This changes update cadence/noise and BRN's transition-relative
warmup; report both units. It must not be mislabeled an official CrossQ result.
At the planned1.024M ceiling this cadence yields at most64,000 critic updates,
so the normalized arms remain in the **BN warmup phase** throughout this initial
screen. It tests early mixed moments, not postwarm BRN stability.

Exact frozen StackCube task/reward/physics: state48/action8,
`panda_wristcam`, `pd_joint_delta_pos`, full50-step episodes. Timeouts bootstrap
from the actual final observation; next-episode reset states never enter those
targets. These are continuing-return learning targets, not a changed success
rule. The restricted deterministic actor exports through the already checked
safe[256,256,8]+identity schema; trusted evaluation code remains frozen.

## Gates, curves and resources

First perform a bounded one-GPU execution smoke across all three arms, at most
5,120 transitions each (15,360 total) and180 seconds. It verifies real wrapper
boundaries, finite gradients/targets, exact update/moment counters, accounting,
numeric artifact schema/parity and pin/source integrity. It is not a scientific
success estimate. No protected evaluation in the smoke.

After smoke passes, the initial scientific screen uses at most1,024,000
transitions and600 trainer-process seconds per arm. Graceful saving and outer
hard timeouts preserve deadline-censored results. Record process wall time,
actual transitions, actor/critic/alpha calls, sampled rows, forward work, replay
and simulator time, source/config/software/GPU identities. No gradient call is
treated as an environment sample.

Prespecified frozen development checkpoints:51,200;102,400;256,000;512,000;
1,024,000, plus the actual final point if censored. One independent score per
unique safe policy hash globally; reuse durable existing reports if identical.
No confirmation query in this screen. Report threshold brackets/censoring;
do not use the50M quality endpoint as a required-experience denominator.
Retain complete recent public training episodes, behavior densities, actions,
rewards, masks and actual successors at checkpoints for target diagnostics.

Proceed to paired additional training seeds only if joint beats BOTH controls
by at least10 percentage points at a common observed checkpoint and its mean
success over the common learning-curve checkpoints is higher. A final-only
crossing, more compute, censored mismatch or toy success does not pass this
gate. Promotion still requires multiple training seeds and independent
confirmation. A10× claim additionally needs strong matched T80/T90 reference
curves, competitive total compute and held-out-task validation.

If all arms remain poor, inspect preserved stage trajectories and target/entropy
statistics, identify the distinguishing negative evidence and refine the same
question before another run. Do not immediately switch to another algorithm.
If split beats joint, preserve that counterevidence; inspect moment conditioning
and gradient effects instead of claiming improved numerical calibration.

Initial envelope: smoke$1 plus scientific screen$6 (up to three trainer workers,
18 bounded development calls and a CPU coordinator), held under the existing
shared$100 total. New allowances require root review against remaining window,
metering and scientific evidence. No automated broad queue or confirmation
sweep is admitted by this plan.
