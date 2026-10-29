# Neither isolated adaptive entropy route recovered early grasping

Both registered crossed runs completed exactly 102,400 transitions with the
same seed-1 random initialization and 6,088 critic /2,029 actor updates. Each
received exactly one frozen 128-episode development score at that point.
Neither met the preregistered grasp-recovery threshold. No policy stacked.

| Actor/temperature initialization | Soft-backup multiplier | Grasp /128 | Lift /128 | Stack /128 | Time |
|---|---:|---:|---:|---:|---|
| High, cached original |1 |0 |0 |0 |70.83 s checkpoint |
| Low, cached alpha-only change |1 |64 |1 |0 |81.12 s checkpoint |
| High, new route test |.005 |2 |0 |0 |75.29 s complete process |
| Low, new route test |200 |0 |0 |0 |58.75 s complete process |

Cached diagonals were reused without retraining or rescoring. Their displayed
times are the saved checkpoint timestamps; the new rows use complete child
process time including imports and final export. New checkpoint timestamps were
72.187 and56.029 seconds respectively. Hardware throughput differences are not
controlled causal effects of these changes.

The narrow result: neither lowering only the soft-target coefficient in the
original high-alpha loop, nor retaining the low-alpha actor/temperature loop
while multiplying its soft-target coefficient, sufficed for the observed grasp
recovery. This weakens either isolated route being sufficient under these
declared interventions. It does not prove both are universally necessary or
uniquely explain the earlier result. All other measured stages in the new runs
(transport, placement, release, contact-free placement and ten-step stability)
were zero. These single-training-seed adaptive-development observations do not
establish stacking gains,10x efficiency or a generally poor algorithm.

## Adaptive feedback limits a simple factorial interpretation

At the last102.4k minibatch, actor/effective-backup alpha were .233682/.001168
in the high-actor route, and .097958/19.591665 in the low-actor route. Thus the
latter route did not maintain a fixed backup alpha near1: its temperature grew
through feedback. Its Q mean was−4597.08 and target mean−4623.12, versus2.37046
and2.38090 in the high-actor route. These are last-minibatch estimates, not true
value error or proof of divergence. Differential-entropy returns can be negative,
and neither scale nor TD loss identifies correct action preferences.

The experiment is a closed-loop route-sufficiency test, as preregistered, not
a fixed-temperature factorial/main-effect estimate. Replaying reference alpha
schedules could answer a narrower controlled-dose question, but would require
its own matched controls; no further training follows automatically.

## Verification, accounting and preserved execution failure

The fixed-batch CPU gate took .614s with no optimizer or simulator calls. Both
routes matched all32 initial actor/critic/target tensors, RNG position and
original actor/temperature losses and gradients. The target-delta identity held
within2.39e−7. Two existing terminal/timeout tests passed. Independent source
review and pinned-SDK registration checks preceded admission. Candidate code
received no evaluator package, protected seeds, prior policy or diagnostic data.

The original coordinator could not see the first worker's receipt on its mounted
read-only volume, although that worker had returned complete. Separate Volume
API reads found it; every candidate file was downloaded and hash-verified.
The original failed summary remains `trusted-summary.json`. A reviewed score-only
handoff validated the archived initial/final safe artifacts and submitted the
previously unqueried high-route primary to the unchanged frozen evaluator.
Global GPU claims were retained; neither learner nor any evaluation was repeated.
`reconciled-summary.json` records the repair and original-summary digest.

Additional training:204,800 transitions,12,176 critic and4,058 actor/temperature
updates,134.040 combined child-process seconds. Frozen evaluation:12,800
transitions,256 episodes,61.034 subprocess seconds; zero confirmation queries.
Training-row reuse and detailed compute counters remain in the raw reports.
The separate earlier failed physical probe added512 public transitions and no
fitting. None of this establishes a new time-to-success result.

Archive: `runs/entropy-route-20261029/`,44 hash-verified candidate/score files
(plus inventories, exact source/plan/CPU evidence and repair records). Both the
training app and score-only app were independently read back STOPPED with zero
tasks. The original$2 reservation covered both apps and was closed only after
archive/readback. Reported campaign increment was$.14491495; session total
$14.06657359, remaining$85.93342641, no active reservations. Metering may lag.
No Runpod spending. The scientific training ledger now contains68 completed
runs across36 recorded variant families; historical failed runs are retained.

The next research work should refine the entropy/coverage hypothesis with
cheap diagnostics, rather than call this a new algorithm or immediately repeat
the failed interventions. In particular, global entropy feedback can be examined
per action coordinate before considering any allocation change. Such a check
must also show where stronger entropy would harm precise grasping; avoiding
saturation alone is not the task objective.
