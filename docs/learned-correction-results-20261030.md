# Supported selection helped, but did not beat random timing

The learned-selector prototype **failed the registered overall comparison**. Its support filter prevented premature interventions and preserved grasping, but final success did not exceed a simple random-time correction. No policy was promoted and historical benchmarks remain unchanged.

| Prototype | Source2 /128 | Source3 /128 | Pooled success | Ever grasp /256 |
|---|---:|---:|---:|---:|
| Original Gaussian actor | 0 | 0 | 0% | 255 |
| Random-time persistent correction | 53 | 55 | 42.19% | 247 |
| Raw learned trigger | 26 | 24 | 19.53% | 113 |
| Learned trigger with support check | 61 | 39 | 39.06% | 255 |

Every correction used the same supplied zero-arm/open-gripper prior. Each fixed parent was an8.192Mcheckpoint from an already completed10.24Mrun. Outcome heads were fitted on the new training-only randomized corpus; thresholds and support cutoffs were frozen before fresh public episodes. Learned triggers used the original48observations, with no geometry rule, success flag or privileged object velocity. This is an assisted composite-selector experiment outside the frozen MLP artifact protocol, not a new from-scratch benchmark score.

Support minus raw gained19.53percentage points,98.75%block-bootstrap interval[11.72,26.95]. It also gained39.06points over the original actor,[33.98,44.14], and passed the coarse grasp-preservation screen. But support minus random was **−3.13points, interval[−13.28,6.64]**, missing the registered+10-point requirement. Raw selection was worse than random by22.66points,[−29.69,−15.23]. All assigned episodes counted.

Coverage matters. The raw head switched in224/256episodes, including78immediate reset interventions, and often prevented grasping. The supported head switched in158/256, with no immediate reset interventions. Of its triggered episodes,100finished successful;98episodes never triggered. That conditional rate describes a selected subgroup and is not a randomized causal comparison with random timing. It does not rescue the failed overall screen.

The earlier held-batch prediction improvement therefore did not establish useful closed-loop timing. Action-conditional support addressed one failure—extrapolation into untreated reset states—but was insufficient. Another plausible issue is the stopping decision: a correction's estimated value now must be compared with the opportunity to wait and correct later. Comparing only with the original policy's eventual outcome, or using a fixed confidence threshold, does not estimate that continuation value. This is a new hypothesis, not an established explanation of the failure.

The next essential check uses finite-horizon optimal stopping and regression of continuation value, informed by least-squares Monte Carlo methods from option pricing. It first tests a small problem with an analytic optimum and a deliberately invalid future-peeking comparator. Existing training trajectories can later supply continuation experience; public prototype episodes will not enter fitting. There is no justification for a threshold sweep or another broad PPO/SAC collection.

Actual work:51,200public physics transitions, zero parameter updates during assessment, zero protected/confirmation/final queries. Workers33.54and25.92seconds; wrappers37.91and29.13seconds. Twenty-six remote files /9,369,149bytes were archived with exact numeric heads, sources and input hashes. A separate CPU audit reconstructed all51,200executed action rows exactly in3.13seconds without physics. Five prototype tests passed before launch; a later combined48-test check passed in0.868seconds. The Modal app was read back stopped with zero tasks.

[Protocol](../cumulative/experiments/learned-correction-probe-v1.json); [complete analysis](../cumulative/results/learned-correction-20261030.json). Raw traces and audit are under `artifacts/cumulative/learned-correction-20261030/`. The source policy's acquisition, fixed corrective prior, data collection, CPU fits, scoring and failed attempts remain separately accounted for.
