# Randomized release: brief success, poor retention

The tested five-command opening repair failed its registered20-percentage-point final-success screen. It nevertheless revealed a specific failure sequence: many policies briefly stack successfully and then lose the stack.

| Intervention | Seed2 final /512 | Seed3 final /512 | Ever success /1,024 |
|---|---:|---:|---:|
| Unchanged | 2 | 0 | 6 |
| Fixed-time opening | 6 | 6 | 551 |
| Geometry-triggered opening | 16 | 11 | 630 |
| Targeted opening plus zero arm delta | 25 | 15 | 564 |

Targeted opening increased final success by2.44percentage points,98.75%block-bootstrap interval[1.27,3.81]. Adding zero arm delta increased it by3.71points over unchanged,[2.05,5.57]. Both are below the registered20-point material threshold. Timing and additional-arm contrasts were unresolved at the registered interval level. These are manual intervention effects, **not trained-policy improvements**.

The design randomized16lanes per intervention within each64lane rollout,32blocks for each of two frozen failed8.192MGAE policies. It avoided exact reset-and-replay assumptions. Every intervention delivered five opening commands; a late fallback included episodes never reaching the geometric trigger. All assigned episodes counted. The unchanged official50stepphysical metric and48stateobservations were retained. No task, reward or evaluator rules were changed.

Exploratory trajectory inspection, conducted after the primary result, found360and270episodes with any success under targeted opening, but only16and11successful at the end. Among these episodes,349and263subsequently lost success at least once. At the first loss, **all failed the measured static criterion**, while only3and2had already lost stacking geometry. Finger contact had returned in177and124cases. The median first loss occurred at the first resumed actor command. The first resumed gripper command was negative in433/474and400/470episodes with a post-window action. These associations do not distinguish gripper closing from arm movement; failures can occur during the opening window too.

The decisive follow-up is therefore maintenance, not another opening-timing sweep: give every group the same initial five-command opening sequence, then independently randomize continued opening and zero arm delta in a2×2factorial. This tests whether preserving the achieved configuration repairs the final-success loss. No learned corrective skill or sample-efficiency claim follows from a successful manual controller.

Actual work:204,800physics transitions, zero optimizer calls, zero protected/confirmation/final queries. Workers73.78and84.69seconds. The archive contains74remote files /42,713,119bytes plus source/plan provenance. A separate CPU audit reconstructed **every one of204,800executed action rows exactly**, including randomization and intervention masks, in13.22seconds without physics. Six policy/analysis/noise checks passed before launch. [Registered protocol](../cumulative/experiments/randomized-release-probe-v1.json); [analysis](../cumulative/results/randomized-release-20261030.json).

Exact raw data and exploratory post-window records are retained under `artifacts/cumulative/release-randomized-20261030/`. The preceding invalid replay comparison is separately preserved in the [design-correction report](release-control-design-correction-20261030.md). Neither result changes historical benchmarks or the84-run training ledger.
