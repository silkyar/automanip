# Fixed-offset neural diagnostic

Design frozen before fitting, 2026-10-28. This is one CPU-only synthetic fixed-policy experiment, not a robot run or algorithm admission.

Use the actual `replay_learner.Critic(width=64, normalization=None)` with the scalar LQR embedded in 48 observations/8 actions: `s'=.8s+.5a`, `r=−s²−.1a²`, policy `a=−.6s`, gamma .95. Generate 16,384 shared replay rows and 4,096 heldout rows with existing RNG seeds/distributions; cast the same float32 random draws and common initialized linear weights to float64, with dynamics/learning in float64. Three initialization seeds, identical minibatch indices across arms, 1,000 updates each, batch 256, Adam .001/(.5,.999), tau .005. No actor, entropy or analytic fitting labels.

Per seed: A is the original reward/initial critic; B adds +1 reward with the same initialization; C uses B's reward and fixed coordinate C=20, subtracting 20 from both internal online/target output biases so represented initialization equals B; D uses shifted reward and C=20 but A's unchanged internal initialization. D provides an informative oracle value offset at initialization. It is not PopArt or a learned method.

The exact internal target is `r_shift+.95*(C+q_target)−C`. Heldout analytic truth uses `k=1.036/(1−.95*.25)`, `Q=r−.95*k*(s')²`, plus 20 for B/C/D; `dQ/da=−.2a−.95*k*s'`. Measure Q RMSE/mean bias and scalar action-gradient RMSE/cosine. Oracle quantities never enter fitting.

Check actual coordinate-adjusted online/target parameters, parameter gradients and Adam states across every update, plus final represented predictions/action gradients: C should match B; D's residual parameters/predictions should match A and represented values should differ by 20. Report actual maxima against 1e−7 without retuning. Parameter parity after initial coordinate adjustment is checked before fitting.

Primary gate: B's gradient RMSE >2× A in all three seeds supports constant-mode interference in this toy. Ratios ≤1.25 in all three, despite high B value error, weaken it; otherwise partial/inconclusive. Coordinate C cannot mathematically improve over B; a D improvement is informative initialization. Numerical parity failure must be reported, not rescued.

Planned fitting cost: 12,000 optimizer calls, 3,072,000 sampled rows, 6,144,000 fitting critic-forward rows. Shared unique data: 16,384 synthetic training transitions plus 4,096 heldout; measurements/gradient rows and full CPU wall/process time reported separately. Hard 300-second CPU wall cap, no repeat/retune. Robot transitions, training policies, teacher data and evaluator queries: zero.

Implementation: `research/discovery/value_offset_probe.py`. Exclusive receipt: `research/discovery/value-offset-probe-20261028.json`. Results will be appended after this single execution.

## Single execution results

**Primary verdict: partial/inconclusive.** B/A action-gradient RMSE ratios were **1.176854, 1.324448, 1.218520**. None exceeded the 2× support threshold. Seed 2 exceeded 1.25×, so the all-seed weakening threshold also failed. Do not change these thresholds after observing the results.

All 12 fits completed their 1,000 updates. A/D and B/C are grouped below because the coordinate parity checks passed; every individual measurement remains in the receipt.

| Seed | Arms | Q RMSE | Q mean bias | Action-gradient RMSE | Gradient cosine |
|---|---|---:|---:|---:|---:|
| 1 | A / D | 0.044619 | −0.032989 | 0.125563 | 0.988401 |
| 1 | B / C | 15.722956 | −15.722856 | 0.147770 | 0.986356 |
| 2 | A / D | 0.123648 | −0.118674 | 0.106995 | 0.992919 |
| 2 | B / C | 15.879817 | −15.879739 | 0.141709 | 0.986098 |
| 3 | A / D | 0.104316 | −0.097477 | 0.114504 | 0.992898 |
| 3 | B / C | 15.756605 | −15.756507 | 0.139525 | 0.987080 |

The largest C/B discrepancy across all measured online/target parameters, parameter gradients and Adam states was **2.3865e−11** (a parameter-gradient discrepancy). Final heldout represented Q/action-gradient differences were at most **3.0287e−13 / 1.9629e−13**. D/A residual parameter/state discrepancies were at most **7.4239e−14**; represented Q minus the oracle 20 differed by at most **7.1054e−15**, and action gradients by at most **4.8850e−15**. All were below the declared 1e−7 threshold. Initial matching checks also passed. Measurement preserved fitted parameters; underlying learner/actor sources were unchanged.

Thus a large constant value error coexisted with fairly accurate action-gradient direction in this fixed-policy example. Some gradient degradation occurred, but the declared strong interference gate failed. C's lack of improvement is the predicted coordinate invariance; D shows the effect of supplying informative initialization, not learning how to remove an offset. These results give no basis to implement a fixed-coordinate trick as an efficiency method. They do not test adaptive reward-rate estimation, PopArt, policy improvement, entropy feedback, robot exploration, or a tenfold gain.

Actual fitting work: **12,000 optimizer calls; 3,072,000 sampled fitting rows; 6,144,000 fitting critic-forward rows**. Measurement added **393,216 critic-forward rows**, including **196,608 action-gradient rows**; analytic labels were computed for **98,304 heldout row presentations** only. The shared dataset remained 16,384 unique synthetic training transitions and 4,096 heldout transitions, with no new data collected per arm.

Diagnostic wall/process CPU time: **6.6890 / 6.2194 seconds**. External `/usr/bin/time -p` for the complete Python process, including imports/teardown: **8.14 seconds real, 7.87 user, 1.04 system**. PyTorch used one CPU compute thread, float64, version recorded in the receipt. No retune, second fit, policy training, simulator transitions, protected episode access, evaluator queries, or cloud calls occurred.
