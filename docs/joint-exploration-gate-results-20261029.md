# Matched-coordinate-marginal coverage gate passed; learning remains untested

The [registered gate](joint-exploration-gate-plan-20261029.md) compares shared
vector versus independent coordinate uniform-mixture selectors, epsilon.1.
At the same frozen actor/state their full one-coordinate marginals match.
The difference is cross-coordinate dependence, not a temporal policy or extra
information about the task. Separate learned actors/states would diverge.

For the fixed10mm/up>=5mm target-kinematic proxy across all78 states:

| Collector distribution | Mean joint proxy | Median joint proxy |
|---|---:|---:|
| Shared vector selector |.0224673 |.0187891 |
| Coordinate selectors |.0168458 |.00195313 |

The point median ratio9.62 and nondecreasing mean pass the declared cheap gate.
An independent review identified a zero-count denominator risk before the root
read this result. The [validity note](joint-exploration-gate-validity-note-20261029.md)
retains the original point receipt and adds only a conservative existing-count
check: working-MC lower-vector median.0170753 versus upper-coordinate median
.00284346, ratio6.005. Coordinate median was positive. No sample, state, endpoint,
inference or physical query was repeated to obtain that check. These working
bounds are not simultaneous guaranteed coverage over all78 states.

Analytic controls also passed. Two rare regions with p=.001,u=.25 give vector
.0062509 versus coordinate.00067081, ratio9.3184 (about9.32; the initial plan's
rough9.26 approximation was imprecise). The exact identity and >9 gate are
unchanged. A single-coordinate event has equal probability. A rare/common
negative control favors coordinate exploration (.0233126 versus.0011491), so
there is no universal dominance or novelty claim.

Additional cost:.134s CPU,78 actor rows,40,014 analytic FK poses,39,936 new
coordinate-mixture arm vectors plus explicitly regenerated existing Gaussian/
uniform values and new masks. Original component FK estimates are reused, not
independent replications. The later bounds calculation took.0843s and used no
additional inference/FK. Zero fitting/robot/scorer calls. Full receipts are
`research/discovery/joint-exploration-gate-20261029.json` and its validity JSON.

This is support for a small **behavior-collector learning comparison**, not a
StackCube success result or9.6x learning gain. The proposed learner comparison
has its own [protocol](joint-exploration-plan-20261029.md), exact source and
independent admission review. It must beat the untouched cached low-alpha
reference as well as its matched-marginal control; no promotion follows from
this coverage gate alone.
