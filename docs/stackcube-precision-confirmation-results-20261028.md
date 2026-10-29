# StackCube 50M confirmation results

The fixed full-horizon constant PPO recipe achieved **95.90% mean confirmation
success** across three training seeds. It missed the preregistered target of
at least 99% mean and at least 98% for every seed. The user accepted about 96%
as the reference performance for the next sample-efficiency experiments.

| Training seed | Development | Independent confirmation | Evaluator seconds |
|---|---:|---:|---:|
| 1 | 127/128 = 99.22% | 497/512 = 97.07% | 106.30 |
| 2 | 126/128 = 98.44% | 489/512 = 95.51% | 100.40 |
| 3 | 126/128 = 98.44% | 487/512 = 95.12% | 100.83 |
| Mean | 98.70% | 95.90% | — |

Each final policy was fixed at 49,971,200 transitions before confirmation.
There were no additional training runs, replacement checkpoints, repeated
policies or repeated confirmation attempts. The frozen scorer, task and protocol
were unchanged. These are historical fixed confirmation scenarios independent
of development, not a private final holdout. Episode outputs and scores are
excluded from candidate training and pilot tuning inputs.

The previous 25M recipe confirmed at 94.08%. Roughly doubling the transition
budget therefore improved the measured confirmation mean by 1.82 percentage
points. This does not establish that doubling compute is an efficient way to
improve reliability. The development/confirmation gap does not identify a cause.

Exact digests and the predeclared target are in
`deploy/modal/precision-confirmation-plan.json` and
`docs/stackcube-precision-confirmation-plan-20261028.md`. The trusted driver
registers the exact hash-bound frozen scorer body against a separate static
policy-digest admission image. Scorer bytecode/constants parity and two admission
tests passed; the original evaluator source and historical allowlist were
unchanged. An initial SDK registration failure happened before any coordinator
or evaluator submission; its setup log is preserved. No evaluation claim was
deleted or refreshed.

Protected reports and their verified inventory are under
`runs/private-confirmation/stackcube-precision-20261028/`. App
`ap-Kx4o8UQNaWMNnC99g9UNKc` stopped automatically with zero tasks.
Provider-reported cost was $0.23332623 at the latest read, subject to billing lag,
within the $1 envelope inside the shared $100 cap. No private final holdout
result is claimed.

See the [experiment ledger](experiment-ledger.md) for all training times,
budgets, unsuccessful experiments and current work.
