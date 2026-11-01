# Next isolated assisted-track test: residual-scaled initial exploration

Status: source ready and five CPU contract tests pass; **future proposal only**.
No current pilot source was changed, no GPU launched and no protected scoring,
private seeds, infrastructure or credentials accessed. The frozen v1 files and
their receipts remain intact. This remains a labeled prior-assisted track with
teacher, demo, BC and student compute counted separately.

## Review findings and hypothesis

The v1 PPO Gaussian starts at **logstd −.5, hence std .60653** in every action
coordinate. It does not start at unit std. The BC initializer changes only the
mean; its fresh critic and global std retain the public PPO initialization.
Consequently, good deterministic cloning does not imply that initial stochastic
training trajectories preserve that behavior. An unfitted value function can
also affect GAE and joint gradient clipping. These are distinct hypotheses, not
established explanations of a pilot outcome. The critic does not literally
overwrite mean weights, and this proposal leaves it unchanged.

The supplied public teacher audit further rejects a blanket “less noise is
better” claim: its final per-coordinate stds range from .169 to 1.312, with two
coordinates above the student's .60653 default. These teacher values are
descriptive only and **never enter calibration or student initialization**.

Other reviewed risks remain: the full state-pair discriminator can distinguish
state marginals, and its unseen-random style reward was high despite zero local
task successes. Neither discriminator confidence nor low BC action MSE certifies
reliable stacking. Generic action ambiguity is also not established here: the
collected labels come from one deterministic feed-forward teacher. The observed
BC/teacher gap does not identify multimodality, covariate shift or accumulated
control error as its cause.

**One mechanism:** initialize the existing eight Gaussian std parameters to

`sigma_i = clip(sqrt(mean_training_rows[(mu_BC(s)_i - executed_demo_action_i)^2]), .05, exp(-.5))`.

Only the BC **training** episodes are used. Validation episodes are excluded.
The .05 floor and original-std ceiling are fixed before the experiment, not
swept. PPO then learns logstd normally. Mean, critic, reward, entropy coefficient,
scheduler, architecture, likelihood calculation, optimizer and policy artifact
schema stay unchanged. This isolates initial sampling rather than combining it
with critic warmup or a different prior representation.

This is a conventional, residual-scaled exploration heuristic, not calibrated
Bayesian uncertainty. Clipped deterministic action labels and approximation bias
affect the residuals. Closest primary context includes
[adaptive Gaussian exploration](https://proceedings.mlr.press/v108/papini20a.html)
and the documented difficulty of offline-to-online improvement in
[AWAC](https://arxiv.org/abs/2006.09359) and
[RLPD](https://arxiv.org/abs/2302.02948). We implement none of those complete
algorithms and claim no novelty. [GAE](https://arxiv.org/abs/1506.02438) is relevant
to the separate value-model concern, which this test deliberately does not change.

## CPU diagnostic and falsification

Existing Mac CPU training data only: 223 BC-training episodes/11,150 rows calibrated
the sampler;24 held-out demo episodes/1,200 rows were used only for an offline
action-distribution diagnostic. The data and its BC actor remain explicitly
non-benchmark, non-production artifacts.

The eight measured stds were:

```text
.06720532196028443  .07451047695331992  .07462261684374780  .08221141931603906
.10957110156167456  .09158898053173271  .10900616393609368  .08283186082968728
```

These numbers are **not production constants**. The payload recomputes them from
the reviewed production dataset and its shared BC actor.

Exact clipped-Gaussian integration on held-out demonstration states gives:

| Initial sampler | Expected executed-action MSE to demo labels | Expected clipping fraction |
|---|---:|---:|
| Original std .60653 | .26744 | .24057 |
| Training-residual stds | .01613 | .12616 |

This confirms a substantial change in the action distribution, not an RL gain or
a predicted success rate. Clipping itself is not evidence of failure. Diagnostic
runtime was .335 CPU seconds; zero new simulator transitions, optimizer updates,
GPU seconds or dollars. Full episode indices, exact values and digests are in
`research/discovery/bc-residual-exploration-probe-v2.json`.

The essential counterexample is also recorded: if a cloned mean 0 requires a
corrective action in (.3,.6), that event has probability .14916 under std .60653
but only 9.87e-10 under the .05 floor. Lower imitation deviation can therefore
suppress necessary correction discovery. This is a constructed counterexample,
not a claim that StackCube requires that interval. It falsifies unconditional
improvement and prevents using the offline MSE result as a promotion criterion.

Five tests pass: per-coordinate RMS/bounds; exact clipped moments against sampled
values including saturation; training-only calibration with unchanged mean,
critic, state_dict and RNG; the lower-noise counterexample; and exact removable
patches with unchanged Agent code and initialization before optimizer creation.
The original probe source snapshot is preserved because later changes added only
runtime identity/time accounting and an embedded source digest.

## Ready source and matched experiment

Payload:
`research/discovery/bc-residual-exploration-v2/assist-bc-ppo-residual-std.py`

```text
payload SHA256: c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200
v1 base SHA256: 7a4de8deecf0adc15d9f8beaef1bf6b9874ca9eb7b0122efbab5d9b1bc80d427
patch SHA256:   32475453cf5080e0cfdd0c0ed11170f785a27e947c3ca7eb7e744faa8f51692a
```

Removing `BC_RESIDUAL_STD` marked blocks recovers the exact frozen BC+PPO v1
payload. The production demo reader retains all version/backend guards; it does
not accept the local CPU smoke dataset by default. Calibration occurs after the
shared BC mean is loaded and before the ordinary Adam optimizer is created.
No new model, parameter or serialized buffer is introduced.

`AUTOMANIP_EXPLORATION_INITIALIZATION` logs dataset and BC actor digests, patch/base
source identities, exact training and excluded-validation episode indices, raw
residuals, all eight old/new stds, extra BC forward rows and calibration wall time.
Initial scalar tags use `audit/assist_init/*`; ordinary per-update
`audit/logstd_dim_0` through 7 remain unchanged. The owner should also retain KL,
optimizer-step, value-loss and gradient/clip audits. Include initialization time
in job/compute accounting; upstream throughput timing starts later.

The smallest next comparison is **original BC+PPO versus residual-init BC+PPO**:
same reviewed production dataset, BC artifact/report, training seed, mean/critic
initialization, PPO settings, transition ceiling, wall cap and scoring cadence.
Use the existing strong BC+PPO CLI, changing only payload/experiment name. For
example, a new paired screen may request 2,048,000 student transitions and 300 trainer
seconds per arm, subject to the owner's declared lease/save reserve and shared
budget. This document does not launch it. A completed v1 comparator may be reused
only if its production artifact, data, source/CLI, software, hardware and budget
match; otherwise run the matched control.

Prediction: less early stochastic departure from the clone preserves more useful
training trajectories and improves the transition-indexed learning curve without
losing final-step quality. Falsification: no earlier threshold/AUC improvement,
lower final quality, or evidence that suppressed exploration blocks correction.
Offline action MSE is not the outcome. The trusted owner retains scoring and
promotion rules. One adaptive paired seed cannot establish reproducibility.

First isolate this sampler change without AMP. If it helps, apply the same sampler
to both BC+PPO and BC+PPO+AMP before attributing an advantage to adversarial reward.
Do not combine critic warmup, new discriminator features or an entropy schedule
into this first comparison. All teacher/demo/BC prefixes and additional calibration
work remain counted. The owner's reported 50M three-seed development aggregate is
98.6979%; no protected confirmation outcomes are inputs to this proposal.

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_bc_residual_v2.py' -v
.venv311/bin/python -m research.discovery.bc_residual_exploration_v2 \
  --base research/discovery/amp-prior-assisted-v1/assist-bc-ppo.py \
  --method assist-bc-ppo-residual-std --output /tmp/new-residual-std-payload.py
```
