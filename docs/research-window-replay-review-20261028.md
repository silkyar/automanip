# Review: joint replay moments and target reliability

2026-10-28. One from-scratch mechanism under review; no algorithm switch proposed.
This agent inspected sources and ran only the specifically authorized CPU algebra
gate. Root owns learner implementation, simulator use, infrastructure and scoring.

## Source basis and claim boundary

Primary paper: [CrossQ v4, March 2024](https://arxiv.org/html/1902.05605v4).
Sections 3.2/4.4 motivate mixing current replay and current-policy successor inputs;
the critic-only normalization ablation is relevant to our safe actor. Its published
recipe also changes target networks, width and optimizer settings. Paper success
is therefore not evidence that mixing moments alone improves our task. Appendix A.3
timings exclude synchronous evaluation/logging, unlike a full provider lease.

Author source is pinned to
[`a318a266679bec3f9d93c44e029860a6da077c89`](https://github.com/adityab/CrossQ/tree/a318a266679bec3f9d93c44e029860a6da077c89).
Local manifest: `research/discovery/upstream/crossq/manifest.json`. The CPU gate
verified the manifest and actual hashes of `sbx/sac/policies.py`, `sbx/sac/sac.py`
and `train.py`. Relevant implementation facts:

- The critic normalizes concatenated observation/action input **before** the
  first dense layer, then after each hidden activation.
- Critic loss computes both halves in one training forward pass and detaches
  the Bellman target afterward. Next rows still influence current-row gradients
  through shared normalization moments.
- Actor loss calls the critic in inference mode with stored running statistics;
  it does not advance critic moments or warmup counters.
- CrossQ defaults include two 2048×2048 critics, actor delay 3, Adam beta1=.5,
  and no lagged target parameters. BRN uses old-stat weight .99, epsilon .001,
  correction limits r∈[1/3,3], d∈[-5,5], and 100,000 training forwards of warmup
  per normalizer. These units are learner calls, not environment transitions.

Our narrow hypothesis should be **shared moments avoid an inconsistent mapping
between replay and successor inputs during learning**. “More reliable Bellman
targets” is a stronger hypothesis: neither the paper's distribution argument nor
this CPU gate establishes that the resulting targets are closer to true values.
Normalization can preserve an irrelevant shift or suppress a meaningful one.

## Causal comparison

| Arm | Role and legitimate contrast |
|---|---|
| Target-SAC | Competent conventional replay control with lagged twin targets and no incompatible BN-target-statistics shortcut. Same safe actor, task, data budget and declared learner budget. Candidate versus this arm tests the adapted recipe, not moments alone. |
| Joint moments | Target-free normalized twin critic; current and successor rows share moments in a single concatenated pass. |
| Split moments | Identical target-free normalized architecture and pass; only the forward normalization moments are computed separately within the two halves. This is the mechanistic negative control. |

The last two arms must share initialization, hyperparameters, replay sampling,
next-action sampling and optimizer/update counts as far as possible. Both compute
running EMA from the union **once** per normalizer/update. Pooled variance includes
the between-half mean term; averaging two within-half variances is wrong. Two
separate training forwards would also change EMA cadence and ordering. Root's
same-pass/shared-EMA design avoids those confounds.

At the same initial state and fixed batch this isolates moment scope. Within a multilayer critic, earlier normalization already changes later hidden
activations, so actual running-stat values can legitimately differ even in the
first forward at identical weights; later learning adds further differences. After online policies diverge, replay contents
also diverge; a paired-seed learning contrast is not an identical-data optimizer
comparison. Do not claim otherwise.

Match critic widths if the question is normalization rather than capacity. A
small target-SAC critic against a wide candidate confounds that contrast. Do not
silently give the control a harmful target-BN combination. If canonical SAC
optimizer/actor cadence differs from the matched recipe, name the differences:
there is no single control that simultaneously isolates every published change.
The historical short, slow SAC screen is not this competent comparator.

## CPU mechanism gate: completed, not robot evidence

Source: `research/discovery/replay_moment_probe.py`, SHA256
`d5ef313a3eeabe8f9558f7ff30577f047362472d5587654ca6609d60924a4d4f`.
Receipt: `research/discovery/replay-moment-probe-20261028.json`.
Runtime .0792s in the script (1.11s process wall time), zero simulator transitions,
optimizer calls, fitted rows, teacher rows or GPU use; cost $0. The test computes
two eight-output Jacobians, equivalent to 16 scalar reverse-mode backward calls.
This is float64 Torch algebra transcribed from source, not execution/parity of
the author's Flax package. No fitting or oracle training labels are involved.

| Gate | Measurement and interpretation |
|---|---|
| Shifted BN aliasing | Two identical four-row patterns shifted to means −2/+2, each variance 5. Shared variance 9 retains normalized means −.6667/+.6667. Separate centering maps both means to 0 and makes the halves identical. Without a shift, joint/split outputs match. This establishes information loss in this constructed feature, not erroneous Q-values. |
| EMA isolation | Starting mean .25/variance 2, old-stat weight .99: union update gives mean .2475/variance 2.07. Averaging half variances gives 2.03; two ordered mean updates give .245225. Both are unintended extra interventions. |
| Textbook unclipped BRN | At epsilon 0 and .001, shared/split forward outputs match common-running normalization within 4.45e-16. |
| Different training Jacobians | Despite that forward equality, current-output sensitivity to next-half inputs has Frobenius norm .37256 for joint and 0 for split; full Jacobians differ by up to .39660. Detached BRN corrections leave BN moment derivatives active. This is intended autodiff, not a finite-difference equivalence claim. |
| Pinned finite-epsilon negative condition | The author's custom-moment implementation only reduces exactly to the textbook expression as epsilon becomes negligible. At epsilon .001, ordinary-feature joint/split difference is .0001727. With deliberately low within-half variance it reaches .62614; official correction clipping increases the difference to .87433. These counterexamples are recorded, not tuned away. |

The textbook identity is worth stating precisely:

`z=(x-mu_batch)/sigma_batch`,
`r=stopgrad(sigma_batch/sigma_running)`,
`d=stopgrad((mu_batch-mu_running)/sigma_running)`.

Without clipping, `z*r+d=(x-mu_running)/sigma_running` in forward values.
The pinned code instead forms custom mean/variance then adds epsilon in the
final normalization. Its low-variance behavior is not algebraically identical.
An adaptation must explicitly choose source fidelity or the standard expression;
it must not claim both byte-level fidelity and an identity the finite-epsilon
source does not satisfy. No upstream production file was changed here.

This gate passes **existence and isolation of a moment-scope effect**, not a
target-quality or policy-improvement gate. The cheapest remaining target-quality
test, if needed, would freeze a policy in a small finite-horizon linear-quadratic
system with analytic soft values, fit all critics using only replay Bellman
updates, and use analytic values solely to measure held-out target/Q errors.
Zero-shift and both shift signs would prevent choosing a favorable fixture.
No such fit was run: it is unnecessary for the completed algebra checks and
would still not establish contact-dynamics transfer.

## Implementation risks for the owned learner

1. **Warmup and momentum convention.** Flax .99 means .99×old+.01×new. PyTorch BN
   `momentum=.99` means the opposite. Each normalizer counts one update per critic
   step, not per twin head or hidden layer. If G updates follow N parallel
   transitions, 100k updates require roughly 100k×N/G transitions after startup.
   A low-UTD vector pilot may remain entirely in BN warmup. Report that regime;
   changing warmup units is a separate declared deviation.
2. **Gradient paths.** Detach sampled successor actions from the actor in critic loss; concatenate both
   critic halves with differentiable moments; detach the resulting target. Doing
   successor critic computation wholly under no_grad, or detaching moments,
   changes the tested operator. During actor loss, freeze critic parameters but
   retain gradients with respect to action; use running-stat inference mode and
   do not update counters. Avoid batch-dependent actor optimization from a
   training-mode normalized critic.
3. **Safe actor.** Keep two 256 tanh hidden layers and no actor BN. Squashed-Gaussian
   SAC's deterministic output is tanh(latent mean), not clip(latent mean). The
   existing reviewed SAC exporter represents this exactly with safe hidden sizes
   `[256,256,8]` and a final identity 8→8 linear layer. Preserve that numerical
   behavior, log-probability Jacobian and bounds; keep stochastic std/critic
   machinery outside the safe actor artifact. Label the method CrossQ-inspired,
   since actor activation/normalization differ from the full author recipe.
4. **Replay boundaries.** Use actual pre-reset successor observations. Preserve
   full 50-step episodes, original rewards/controller, ignored stock early
   terminations and the declared timeout-bootstrap convention across every arm.
   Mixing reset-state successors or masking timeouts inconsistently can dominate
   any normalization effect. A lower self-bootstrapped TD loss is not validation.
5. **Numerical checks.** Record finite variances, correction clipping, current/next
   moment gaps and training-versus-running-stat predictions. Epsilon .001 may
   dominate nearly constant coordinates. A finite value is not necessarily a
   calibrated value; entropy contributions also change the scale of soft Q.

## Accounting and falsification

Report acquired transitions, vector rounds, warmup acquisition, replay capacity
and age, batch size, sampled replay rows, critic optimizer calls, actor calls and
temperature calls separately. Actual critic UTD is `critic_calls/transitions`;
replayed fitting rows per acquired transition is a different number. With B
replay rows and two heads, a joint current/next pass evaluates 4B head-row examples;
target-SAC also evaluates current and target rows, with different backward cost.
Actor-loss critic inference and next-action sampling are additional work.

Charge wide-critic memory/compute, initialization, compilation, transfer,
checkpointing, diagnostics and scoring inside the declared total compute. A
1200s cap can censor transition-matched comparisons differently; report both
actual transitions and full elapsed costs. A faster split implementation must
not receive extra gradient calls unnoticed. No demonstrations or prior weights
belong in this from-scratch comparison.

Predeclare interpretation before outcomes:

- If moment scope changes no outputs/gradients in the intended warmup regime,
  reject the implementation/isolation claim before robot spending. The current
  scalar gate finds the expected distinction.
- If joint and split arms have indistinguishable learning within the bounded
  uncertainty, there is no evidence that shared moments are necessary there.
  Do not extend or retune solely to obtain separation.
- Joint beating split but not competent target-SAC shows a weakness of split
  normalization, not useful sample-efficiency progress.
- Joint beating both supports this adapted learning recipe at its tested budget.
  It does not by itself prove more accurate targets: forward calibration,
  normalization conditioning, gradient coupling and later data coverage compete
  as explanations. After unclipped postwarm BRN, equal forward values with
  different learning particularly implicate gradients rather than value-scale
  alignment, subject to the pinned epsilon caveat.
- Reject a claimed gain that exists only in critic loss, Q magnitude, return
  without final task competence, extra learner compute or altered task semantics.
  Independent training seeds and protected verification remain necessary for a
  robust gain. Earlier negative PPO critic completion does not resolve this
  different off-policy target question.

No robot success prediction follows from the CPU result, and no additional
method is proposed if this fixed mechanism fails.

```sh
.venv311/bin/python -m research.discovery.replay_moment_probe \
  --output /tmp/new-replay-moment-probe.json
```
