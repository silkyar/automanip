# BC exploration pilot results

Neither PPO variant retained the cloned student's development performance after
5,120,000 new environment transitions. The residual-noise initialization failed
the predeclared follow-up gate and is stopped. No confirmation was queried.

| Policy | Training process time | New RL transitions | Actual joint optimizer steps | Frozen development |
|---|---:|---:|---:|---:|
| BC-only random student | 10.87 s | 0 | 5,000 BC updates | 49/128 = 38.28% |
| BC + standard PPO noise | 307.65 s | 5,120,000 | 22,782 | 0/128 |
| BC + residual-calibrated noise | 209.45 s | 5,120,000 | 100 | 0/128 |

Both PPO arms started from the exact same BC actor, dataset and BC report, with
the same random critic, optimizer configuration, training seed, reward, task,
physics, state/action interface and frozen evaluator. Only the initial existing
eight Gaussian standard deviations changed; they remained trainable. The
observations establish loss of the starting deterministic policy's measured
success at the final checkpoint, not its cause or its eventual behavior after
a longer budget. No extra training is used to rescue this fixed screen.

## Exploration and optimization interaction

The standard arm's declared initial standard deviation was .60653 in every
coordinate. The residual arm recomputed its initial values from successful
**training** demonstrations, excluding all 49 held-out demonstration episodes:

`[.07161, .07851, .07931, .08391, .11645, .09797, .11457, .09182]`.

Its cumulative audit shows exactly one joint update in each of 100 rollouts.
Logged approximate KL ranges from .21824 to 6.01261, exceeding the unchanged
.1 stopping threshold. The control recorded between one and 256 updates per
rollout and 22,782 total. PPO's KL condition stops the joint actor/critic loop,
so reducing exploration also sharply reduced value fitting in this experiment.
These are measured counts and code behavior; they do not establish that
additional critic fitting will restore task success. Both arms failed despite
their different counts. The residual arm's lower wall time is not a useful
performance improvement.

The next separate from-scratch proposal tests that coupling directly: retain
the actor's KL stop and continue only value fitting through the remainder of
the existing eight configured epochs. It uses no teacher data and is related
to established policy/value optimization separation, without a novelty claim.
It was proposed before these completed development outcomes were inspected.

## Assistance and costs

This was a demonstration-assisted track. Its own-project teacher had already
used 49,971,200 transitions and 2,295.41 trainer seconds. Fresh GPU collection
added 25,600 transitions in 29.37 subprocess seconds: 512 complete episodes,
497 successful and 15 failed, all retained. The shared BC fit selected 448
successful training episodes and held out 49 entire episodes; it performed
5,000 updates / 5,120,000 sampled demo rows. Held-out action MSE was .009289.
Teacher weights were not a student input; the random student learned examples.
Teacher training, demo generation and BC are additional costs, not free
initialization or a from-scratch sample-efficiency result.

The verified archive `runs/stackcube-bc-exploration-pilot-20261028/` contains
59 files and 47,411,054 bytes, including all stage inventories, raw complete
failures, native checkpoints/events/logs, safe actors, calibration receipts and
three frozen development reports. Summary SHA256:
`dfc60d3f4843bd2448a6ed485930f523aa1a8b95cc2d12ea223ae24427f56ba0`.
No native checkpoint or NPZ was deserialized in the trusted scorer or local
archive verification. The independent scorer received safe numeric policies.

App `ap-bOBQJKXj8qg3MiLITcrYGF` stopped automatically with zero tasks. Reported
pilot cost was $0.43797593, below the $3 envelope, subject to billing lag.
Session reported spend after close-out was $11.23887100 under the same $100
cap. AMP stayed disabled and no GPU motion-prior experiment was launched.

See [the fixed plan](stackcube-bc-exploration-plan-20261028.md) and
[the experiment ledger](experiment-ledger.md).
