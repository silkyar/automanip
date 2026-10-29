# Public continuation-noise probe: setup failure, no variance result

2026-10-28. Both attempts are preserved. No successful rollout, variance
estimate, policy training or protected evaluation occurred. No third setup
attempt will run in this autonomous window.

The [original protocol](continuation-noise-probe-plan-20261028.md) passed
analytic/estimator CPU checks and independent method/source review. It assumed
that identical reset seeds would replicate1024 environments. Its guard stopped
before the first env.step: maximum initial-observation difference1.8388419.
Pinned task/reset source explains the issue: one Torch RNG supplies different
rows of a batch of object placements/quaternions. Equal seed entries reproduce
the vector batch, not identical rows. Child16.3884s; wrapper21.5082s.

The [separately reviewed v2 protocol](continuation-noise-v2-plan-20261028.md)
explicitly copied registered actor/articulation state through public APIs and
reset controller buffers. Its physical-state parity residual was1.19209e−7;
derived policy-observation residual was1.90735e−5, exceeding the predefined1e−5
limit. It stopped before actions. A first-successor parity result was therefore
never measured. Child14.2855s; wrapper19.7402s. This is not evidence about either
policy's gradient variance. Derived observation precision is a plausible issue;
its cause was not proven and the tolerance was not loosened after the result.

Total actual/attempted rollout transitions:0. Setup performed2 explicit reset
calls across the attempts,2 physical-state reads,1 batched state-copy write,
1 controller reset and1 extra observation read. There were0 optimizer calls,
teacher demonstrations or protected evaluator queries. Simulator initialization
and setup still consumed GPU/CPU time and provider cost.

Both apps were verified stopped with zero tasks. Archives contain hash-verified
inventories, receipts/logs, exact source, policy inputs and protocol versions:

-`runs/continuation-noise-reset-failure-20261028/`
-`runs/continuation-noise-copy-failure-20261028/`

Their guards prevented unsupported conditional-variance conclusions. The cheap
mathematical findings survive independently: continuation noise cannot be
removed by an action-only control variate, and same-sample adaptive gating can
introduce bias. They do not supply a measured noise decomposition for StackCube.
Related [literature review](literature/gradient-variance-followup-20261028.md).
