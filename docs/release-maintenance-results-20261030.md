# Maintaining a released stack: a causal diagnostic

After the same five-command opening sequence, maintaining opening and commanding zero arm delta raised final physical success from **32/1,024 (3.13%) to596/1,024 (58.20%)**. This is a randomized **hand-specified intervention**, not a new learned policy, training result or sample-efficiency improvement.

| Post-opening behavior | Seed2 /512 | Seed3 /512 | Pooled final success |
|---|---:|---:|---:|
| Resume the actor | 24 | 8 | 3.13% |
| Keep fingers open | 252 | 84 | 32.81% |
| Zero arm delta | 94 | 68 | 15.82% |
| Keep open and zero arm delta | 310 | 286 | 58.20% |

All groups first executed the original actor, then the same geometric-triggered five opening commands with original arm actions. Only after this window did the randomized two factors differ. Fallback episodes also counted, even when no post-window time remained. In this pinned controller, zero arm delta sets each new position target to the currently measured joint position; it does not preserve an earlier desired target or physically immobilize the robot or cube.

The combined gain was55.08percentage points, registered98.75%whole-vector-block bootstrap interval **[51.27,58.79]points**. Opening alone gained29.69points[25.88,33.30], passing the20-point material screen. Zero arm alone gained12.70points[9.86,15.63], below that material threshold. The factorial interaction was+12.70points[7.03,18.55], supporting a benefit of the combination beyond the sum of these individual group differences. Intervals condition on the two fixed failed policies; they do not describe a population of training seeds.

This identifies a repairable maintenance bottleneck in these policies. It does not show that opening initiation is solved, that every remaining failure has the same cause, or that a learner will discover this behavior efficiently. The previous short-window probe reached transient success frequently but lost staticity, often at the first resumed actor action. The factorial now supplies intervention evidence that continued finger and arm behavior contribute to the final failure. No evaluator, reward, horizon or physical success rule was altered.

![Physical success over time under manual interventions](../artifacts/cumulative/release-maintenance-20261030/success-trajectories.png)

The bands above show the range of two fixed source policies, not confidence intervals. Every curve includes1,024assigned episodes. The left panel is the preceding randomized short-opening probe; the right panel is this maintenance factorial. Figures are available as PNG and SVG in the local archive.

Actual work:204,800physics transitions, zero optimizer calls, zero protected/confirmation/final queries. Workers took88.08and75.54seconds; wrapper times were91.99and78.52seconds. The source actors were acquired after8.192Mtransitions each in already counted from-scratch GAE runs. Their acquisition is not free, and no new training happened here.

Seventy-four remote files /38,968,356bytes were downloaded with hashes. A separate CPU audit reconstructed every one of204,800executed action rows exactly, including assignment, opening and arm masks, in12.00seconds without physics. Four maintenance policy/analysis tests passed before launch. [Protocol](../cumulative/experiments/release-maintenance-probe-v1.json); [complete analysis](../cumulative/results/release-maintenance-20261030.json). Raw trajectories, audit and source provenance are under `artifacts/cumulative/release-maintenance-20261030/`.

The next research question is how to **learn** persistent corrective behavior without supplying the geometric trigger or manual actions as an uncounted teacher. A narrow candidate is state-dependent action uncertainty or a learned corrective option; an entire new chunking algorithm is not yet justified. A learning experiment must use separate training data, preserve the frozen artifact/evaluation contract, report source-policy acquisition, and compare with equal-budget ordinary PPO continuation. Any later from-scratch claim also requires fresh initialization and matched baselines. These diagnostic episodes remain excluded from the learner.
