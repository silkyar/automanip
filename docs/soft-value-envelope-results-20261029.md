# Frozen SAC value audit: late positive inconsistency on stored transitions

A single CPU-only audit found values incompatible with the frozen-policy reference
bounds in the late ordinary-SAC snapshot. Earlier ordinary SAC and both coordinate
snapshots had no such breaches on the four declared pools. This is a consistency
finding, not proof of a simulator bug, historical spike reconstruction, true value
error under the adaptive training process, or the cause of behavior regression.

Production actor/action/log-density reconstruction matched exactly on the fixed
64-row gates. Reconstructed production targets differed by at most7.63e-6 in CPU
FP32, within the predeclared tolerance. Snapshot parameter digests and the parent
CPU RNG were unchanged/restored. No optimizer, simulator or scorer was invoked.

## Conditional reference

For this actor, final hidden features are bounded by tanh. For action coordinate j,
`B_j = ||W_mean,j||_1 + |b_mean,j|` bounds the latent mean. With log-standard-deviation
ell in[-5,2] and c=sqrt(2/pi):

`H_min = sum_j min_{ell in {-5,2}} [0.5 log(2 pi e) + ell - 2 c exp(ell) - 2 B_j]`.

This follows from entropy change under tanh, logcosh(u)<=|u|, and
E|mu+sigma Z|<=B_j+sigma c. The expression is concave in ell, so its minimum is
at an endpoint. The action-box maximum entropy is `H_max=8 log2`.

For a **frozen policy, constant nonnegative alpha, discount gamma and rewards in
[0,1]**, the expected soft-Q reference interval is:

`[gamma alpha H_min/(1-gamma), (1+gamma alpha H_max)/(1-gamma)]`.

It does not bound individual Monte Carlo targets. Nor does it establish that a
lagged critic trained with changing policies/temperatures should already equal
that stationary reference. Positive breaches are conditional incompatibility with
exact stationary Q, not an unconditional implementation certificate.

The stock normalized reward is not explicitly clipped: its gripper-width branch
relies on physical joint states. Under legal Panda finger positions, the reward
branches are bounded as required. All observed audit rewards were in[0,1] and
post-action finger positions in[0,.04]m, but that is not a proof about arbitrary
corrupted or unobserved future simulator states. This assumption remains explicit.

## Observed numerical result

| Frozen snapshot | Alpha | Expected-Q reference interval | Breaches on declared pools |
|---|---:|---|---|
| Ordinary SAC512k |.008611 |[-487.243,104.727] |None |
| Ordinary SAC terminal |.019180 |[-1145.157,110.529] |Upper only |
| Coordinate SAC512k |.006853 |[-404.463,103.762] |None |
| Coordinate SAC terminal |.006327 |[-400.036,103.474] |None |

Both late ordinary critics exceeded the upper bound together on57/3,328 identical
stored behavior state/actions. Taking their minimum therefore does not eliminate
these breaches. Per-pool counts were13/832,29/832,9/832 and6/832, with the declared
numerical margin. All other snapshots stayed within their intervals on those same
inputs. This control holds the stored observations/actions fixed across snapshots;
alpha and the stationary reference still differ.

Across eight shared Gaussian draws, the largest reconstructed target was+207.816:
critic contribution+207.885, entropy contribution−.273, reward+.204. Other late
ordinary maxima were190.234,204.843 and133.418. Their large reconstructed tails
were positive and critic-dominated on these rows. Historical absolute targets
above1,000 were **not recovered**; the original replay/minibatches were not saved.
There was no pool expansion or search to reproduce them.

The same-input control concerns stored behavior actions, not actual actor-update
minibatches. Policy actions also differ across frozen snapshots in the sampled
backup audit. These observations do not uniquely identify critic-parameter drift,
entropy feedback, data distribution, or the cause of acquisition/regression.
Remaining within a broad bound does not establish an accurate critic.

## Work and next use

Four fixed observed pools,832 rows each, evaluated against four snapshots, with
eight shared standardized Gaussian draws. Counted13,824 actor-forward rows,
267,776 critic-head rows,107,008 Gaussian sample rows, zero optimizer calls.
Actual child-process wall2.968 seconds; internal audit clock1.254 seconds excludes
imports/final serialization. Zero new environment transitions, GPU seconds or
benchmark queries. Original data/training costs remain in historical records.

Independent saved-array review reproduced all16 cells' summaries and target
components exactly, recomputed entropy/value bounds from numeric weights, and
checked observed rewards/finger ranges and joint-twin breach counts. Production
reference tensors were not saved; its forward-parity gate is supported by the
reviewed source and runtime receipt rather than a second forward execution.

[Protocol](../research/discovery/soft-value-envelope-plan-20261029.json),
[results](../research/discovery/soft-value-envelope-audit-20261029.json), and
[saved numerical arrays](../research/discovery/soft-value-envelope-audit-20261029.npz).

The new cumulative-learning milestone supersedes further algorithm experiments.
This finding remains context for explicit critic initialization, numerical checks,
and behavioral retention measurement. It does not show that critic resets help
transfer, authorize a value-clipping intervention, or require another GPU run.
