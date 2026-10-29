# Active reliability research: two rejected mechanisms and an episodic risk pilot

Target: reliable 99–100% final StackCube success across training seeds, with
protected scoring unchanged. No such robot result is claimed here. This phase
implemented and tested three mechanisms rather than continuing PPO parameter
sweeps. All experiments were CPU-only and each completed within five minutes.
No GPU, infrastructure or paid API was launched by the candidate agent.

Public SAC feedback is preserved: 262,432 transitions, 129,216 optimizer rounds,
0/128 final stacks; grasp 128/128, lift 115/128, transport 112/128, placement
55/128 and release 0/128. This one bounded adapted recipe does not falsify SAC or
off-policy learning generally. The strong PPO reference remains the owner-declared
confirmed reference. CAPG and the earlier recursive-target branch stay stopped.

## 1. Learned local contraction: gate failed

`research/discovery/contraction_v1.py` fits a dynamics model and a direct scalar
reward critic to the same ordinary transitions. The candidate augments actor
learning with a penalty on the largest singular value of the learned closed-loop
Jacobian. Controls optimize the direct reward critic, with and without a spatial
smoothness penalty. No analytic dynamics Jacobian or optimal action is a label.

Six paired seeds in each of a smooth and a synthetic hybrid actuation-switch
plant, equal actor update counts:

| Plant | Direct reward Q | Q + spatial smoothness | Q + contraction |
|---|---:|---:|---:|
| Smooth | 83.33% | 66.67% | 66.67% |
| Hybrid switch | 0.065% | 0.586% | 24.09% |

The gate required a gain of at least 13/256 successes in at least four of six
paired seeds in **both** plants. The candidate had zero qualifying wins in the
smooth plant and six in the hybrid plant, so the gate failed. Stop this fixed
Euclidean-contraction version; no robot pilot is warranted. These are local,
offline one-step controls, not competent full robotic RL algorithms.

A separate chain-rule test passes. A counterexample also shows why one-step
Euclidean contraction is stronger than ordinary stability: a matrix with spectral
radius .7 can have largest singular value greater than 1. Fixed coordinate units
and transient amplification can therefore conflict with the auxiliary penalty.
This is a limitation of this version, not a refutation of contraction control.

Closest prior art: [learned contraction metrics](https://proceedings.mlr.press/v155/sun21b.html),
[contraction actor-critic](https://arxiv.org/abs/2506.15700), and
[ContractionPPO](https://arxiv.org/abs/2603.19632). Ordinary policy smoothness is
already [CAPS](https://arxiv.org/abs/2012.06644). Our prototype supplies no learned
metric, certificate or novelty claim. Full result:
`research/discovery/contraction-v1-results.json`; experimental loop 10.50 seconds,
49,152 toy data transitions, 7,200 dynamics and reward-critic updates each, 14,400
actor updates. Contraction adds second-derivative compute.

## 2. State-conditioned exploration: gate failed

`state_variance_v1.py` implements an independent, per-step Gaussian variance head
while leaving deterministic actor weights and the observation/action interface
unchanged. Its zero-output initialization preserves the pinned public PPO mean,
critic, Gaussian density and sampling RNG exactly in CPU tests. The new head is
training-only and is not part of the exported deterministic policy.

The two-stage toy requires a broad action to reach a precision stage, followed by
a narrow terminal action. All learners receive ordinary observations/actions and
scalar rewards, without optimal-action targets. Six paired seeds, identical mean
architecture and update counts, with global-variance and entropy-decay controls:

| Policy family | Final deterministic toy success |
|---|---:|
| Global variance | 100.00% |
| Global variance, entropy decay | 99.90% |
| State-conditioned variance | 88.67% |

There were zero wins over both controls. Five candidate seeds reached 512/512;
one reached 164/512. The candidate reduced precision-stage std to its declared
floor near .00674 while retaining larger early-stage std. That allocation is
measured; it does not establish the reason for the failed seed. Extra parameters,
compute and the bounded logstd are explicit adaptations, so this is not a pure
variance-allocation causal proof. Stop this version without a GPU attempt.

Closest primary work: [gSDE](https://arxiv.org/abs/2005.05719) and
[adaptive Gaussian exploration](https://proceedings.mlr.press/v108/papini20a.html).
Our independent per-step head is not temporally correlated gSDE. Full result:
`research/discovery/state-variance-toy-v1-results.json`; 58.91 experimental seconds,
9,216,000 toy transitions, 72,000 actor/critic update rounds, equal collected
transitions but additional head computation. This result does not refute those
published methods.

## 3. Lower-tail return learning: illustrative gate passed

`quantile_tail_v1.py` implements exact pinball regression, fixed equal-mass
quantiles and fractional-bin lower-tail integration. A contextual terminal toy
deliberately has a reward/reliability tradeoff: faster choices have higher payout
but greater stochastic failure probability. All critics fit the same ordinary
context/action/scalar-reward transitions, including failures. Analytic success
probabilities are diagnostic only.

| Actor objective | Analytic mean success | Expected reward |
|---|---:|---:|
| Same quantile critic, mean | 94.30% | .93149 |
| Conventional mean minus .5 std | 98.09% | .91523 |
| Lower 10% quantile mean | 99.27% | .90943 |

All six paired gains over the quantile-mean control exceeded two percentage
points. The prespecified **average** gate passed; the worst candidate seed was
98.87%, so this does not achieve the across-seed 99% goal even in this toy.
The lower expected reward is an explicit objective tradeoff, not a free gain.
Full result: `research/discovery/quantile-tail-toy-v1-results.json`; 10.35 seconds,
98,304 terminal toy transitions, 12,000 quantile and moment-critic updates each,
9,000 actor updates. Critic parameter counts differ and are not compute matched.

This toy constructs the tradeoff; it does not show that StackCube dense returns
rank late stability failures correctly. Learned quantile outputs also cross in
the recorded fit diagnostics. Those diagnostics do not certify useful action
derivatives or robustness to model errors. Therefore **do not promote the
learned-quantile-gradient toy directly to a robot learner**. The next implementation
instead estimates whole-episode risk gradients from actual trajectories, avoiding
learned dynamics/return-model action gradients. It is a distinct, conventional
score estimator for the same risk objective, not a tuned rescue of either failed
mechanism above.

The additional model-error check is an analytic counterexample: adding
`.01*sin(100*a)` to the exact toy CVaR changes every value by at most .01, but at
action 0 reverses its action derivative from -.786 to +.214. The value-error-only
robustness premise therefore fails. This does not establish that the fitted toy
or a robot critic has that error function; it establishes why the illustrative
performance gate cannot certify learned action gradients. Record:
`research/discovery/quantile-action-gradient-counterexample-v1.json`. The proposed
episodic score runtime has no learned-model action derivative to corrupt this way.

Closest primary work: [quantile distributional RL](https://ojs.aaai.org/index.php/AAAI/article/view/11791),
[IQN risk-sensitive policies](https://proceedings.mlr.press/v80/dabney18a.html),
and [policy gradients for coherent risk](https://papers.neurips.cc/paper_files/paper/2015/hash/024d7f84fff11dd7e8d9c510137a2381-Abstract.html).
Risk-sensitive objectives and their combinations with PPO are established;
[CPPO](https://ijcai.org/proceedings/2022/510) uses a different constrained
formulation, and [predictive CVaR policy gradients](https://proceedings.mlr.press/v235/kim24x.html)
learn component weights. Our ordinary episodic score adapter claims neither
algorithmic novelty nor equivalence to those complete methods.

## Concrete robot handoff

The self-contained, pinned three-arm payloads and exact CLI contract are in
[the pilot handoff](research-discovery-risk-pilot-20261027.md). They preserve the
ordinary actor/critic state_dict, frozen public interface and stock reward.
The risk score is `(G-q_alpha)*I[G<=q_alpha]/alpha`, with the VaR baseline and no
ordinary state-value baseline inside the selected tail. Alpha 1 supplies a matched
mean-return score control. The ordinary strong PPO arm retains GAE.

Nine exact two-step stochastic gradient checks cover policy probabilities
.2/.4/.7 and alpha .2/.5/1; maximum finite-difference error is 1.24e-11. A test of
the actual clipped surrogate at the behavior policy also verifies its explicit
horizon and RMS scaling. Six risk-specific tests pass. Four contraction/variance
tests and four quantile-kernel tests also pass. These validate mathematics and
interfaces, not robotic policy quality. PPO clipping, repeated epochs, estimated
VaR and batch RMS scaling remain declared approximations.

The owner is preparing a separately accounted continuation from OUR confirmed
24,985,600-transition reference, with optimizer restart in every arm. No protected
reference bytes were inspected by this agent. Pilot outcome and policy quality
remain pending. No claim of a fresh-from-scratch or equal-total-budget gain is
permitted. No earlier failed experiment or running source was changed.
