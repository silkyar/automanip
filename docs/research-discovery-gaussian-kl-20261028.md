# Gaussian mean-step/KL diagnostic after the failed BC screen

Status: bounded CPU diagnostic complete; frozen BC/residual-std/from-scratch
payloads remain unchanged. No GPU launch, simulator query, protected episode,
private seed or infrastructure access occurred. These results do not establish
robot learning or repair either failed BC arm.

## Public training signature

The owner supplied the following training-audit aggregates, separate from
protected scoring episodes:

| BC+PPO arm | Joint optimizer calls over 100 rollouts | Logged approximate KL range |
|---|---:|---:|
| Original std .60653 in each coordinate | 22,782 | .01999–.13188 |
| Residual-initialized std | Exactly 100, one per rollout | .21824–6.01261 |

The target KL was .1. Production residual stds were approximately
`[.0716112,.0785102,.0793124,.0839097,.1164491,.0979666,.1145714,.0918172]`.
This is a specific sampler/step-size/trust-region interaction signature. The
original implementation also stops critic optimization when actor KL stops.
However, both RL arms finished 0/128 development success, versus 49/128 for BC-only;
these counts do not establish why deterministic BC performance was lost.
The failed residual version remains stopped without extension or tuning.

## Mathematical mechanism and primary context

For equal diagonal std before and after a mean change,
`KL(old||new) = .5*sum_i[(delta_mu_i/sigma_i)^2]`.
An equal action-mean movement therefore has much larger latent KL at smaller
sigma. Fresh Adam steps are approximately gradient-scale invariant when epsilon
is small; reducing sigma does not automatically shrink the mean parameter step.
PPO's clipped objective evaluated at the old parameters is not a hard bound on
the KL produced by the subsequent optimizer step. The next minibatch check can
then halt training.

These are established optimizer/trust-region issues. Primary sources are
[Adam](https://arxiv.org/abs/1412.6980),
[PPO](https://arxiv.org/abs/1707.06347) and
[TRPO](https://proceedings.mlr.press/v37/schulman15.html). No novelty claim is made.

## Exact bounded diagnostic

Source: `research/discovery/gaussian_mean_kl_probe_v1.py`.
Result: `research/discovery/gaussian-mean-kl-probe-v1.json`.

The inputs were the existing public Mac CPU demo observations and local BC
student, not production/protected trajectories. Two disjoint sets of 32 training
episodes supplied 1,600 update observations and 1,600 next-minibatch probe states.
All BC validation episodes were excluded. The student mean and observation inputs
were identical in every case.

Each of six synthetic sampling seeds drew its own old-policy Gaussian actions
conditional on these fixed states. A shared, fixed synthetic advantage vector
was used across all four cases. It was a function of the standardized sampling
noise, not GAE, teacher reward, task return or an action label treated as RL reward.
States themselves remain an offline demonstration distribution; no on-policy
robot trajectory is implied.

Each case performed **one** fresh-Adam update of the exact clipped PPO actor plus
entropy objective, using LR .0003, epsilon 1e-5, clip coefficient .2, entropy .003
and global actor-gradient norm .5. Critic gradients were excluded to isolate actor
geometry; this is not a full joint PPO training update. Logstd LR stayed .0003.
Only mean-network LR was scaled in the two controls.

Residual sigma was recomputed from the local BC training split, not copied from
the production audit or teacher. Its geometric ratio to default sigma was
`r=.14051531477915882`. The four predeclared cases used mean LR multipliers
`1,1,r,r^2`. No scale search or rerun selected a winner.

| Case | Median mean-parameter step L2 | Median exact next-batch latent KL | Median sampled next-batch KL proxy |
|---|---:|---:|---:|
| Default sigma, original LR | .10191 | .22389 | .21136 |
| Residual sigma, original LR | .10183 | 16.79096 | 15.82227 |
| Residual sigma, mean LR×r | .01431 | .33578 | .32111 |
| Residual sigma, mean LR×r² | .00201 | .00664 | .00634 |

Mean movement dominated: the variance-only KL component was about 5–7e-7. The
default case also exceeded .1 in all six synthetic seeds. Thus these values are
**not a numerical reconstruction of the production KL range**. The sigma-ratio
control reduced KL substantially but did not bring it below .1; the squared-ratio
control did so in this constructed test. Lower KL by itself is not improved
learning—an arbitrarily small LR can suppress both KL and useful progress.

The diagnostic supports one narrow mechanism: smaller sampling sigma can coexist
with an almost unchanged Adam mean step and a much larger policy-space movement.
It does not prove that this mechanism caused either final 0 result, that it explains
the entire 100-rollout trace, or that any tested LR scale is optimal. The KL is for
the latent Gaussian used by the trainer, not the clipped executed-action density
or physical-state distribution. Gaussian likelihoods/CAPG were not modified.

Four CPU tests pass: analytic Gaussian KL/decomposition; inverse-variance scaling;
scalar Adam mean-step invariance; and mean-only LR scaling with the same std
update and unchanged base actor. The experiment took 1.019 CPU seconds, 24 optimizer
calls and zero simulator transitions/GPU dollars. All cases, digests, episode
indices and forward/backward row counts are in the result JSON.

## Future hypotheses, kept separate from the current from-scratch pair

The reviewed from-scratch control/critic-completion pair remains the priority and
unchanged. The following are future assisted-track proposals only:

1. The previously requested two-factor experiment can cross **initial sigma**
   (original/residual) with **critic schedule** (coupled stop/critic completion).
   It tests whether allowing value fitting after actor stop changes the outcome;
   it does not itself repair large actor steps.
2. A separate narrow LR experiment can hold residual sigma and the critic schedule
   fixed, comparing original mean LR with one explicitly declared scaled mean LR,
   while std and critic LR remain unchanged. This tests the step-size interaction
   without conflating it with critic completion. The squared-ratio control is a
   conservative candidate from this diagnostic, not an established solution.

An implementation trap must be addressed in any later reviewed version: the
current precision scheduler resets every optimizer group to the same LR each
rollout. Distinct LRs set only at optimizer construction would be overwritten.
Any new proposal must preserve group multipliers through that scheduler and log
the actual mean/std/critic LRs on each update. No such training-code change is
made here.

A useful future result requires more than KL compliance or restored optimizer
counts: paired, matched-data learning curves/final quality must improve with all
teacher/demo/BC and additional compute counted. If KL falls but learning remains
poor, the step-size hypothesis is insufficient. If scaled mean steps suppress
correction learning or only make progress slower, reject that version. Independent
training seeds and owner-controlled confirmation remain necessary for any gain.
No additional BC runs, source rescue or current-pilot changes follow from this
diagnostic.

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_gaussian_mean_kl_v1.py' -v
.venv311/bin/python -m research.discovery.gaussian_mean_kl_probe_v1 \
  --dataset research/discovery/local-cpu-demos-smoke-v1 \
  --bc-actor research/discovery/local-cpu-bc-smoke-v1/final-actor \
  --output /tmp/new-gaussian-kl-probe.json
```
