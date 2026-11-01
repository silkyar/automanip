# First-update probe: large updates damage behavior, useful direction remains unproven

The approved experiment completed in **46.66 seconds of probe process time** on one NVIDIA L40S, using exactly **201,600 new simulator transitions**, including diagnostics. Production reconstruction passed bitwise on all three real batches. **The full H1 gate did not pass:** limiting movement recovered most of the lost behavior, but the small forward update did not reliably beat its reverse-direction control. No follow-up training was launched.

## Results on paired public diagnostic episodes

Each column is a new collection from the same historical assisted actor/critic, not an independently trained policy. Each branch uses the same 64 fresh diagnostic reset episodes within that column. These are training diagnostics, not protected benchmark results.

| Actor movement | Collection 90282001 | Collection 90282002 | Collection 90282003 | Total / 192 | Success |
|---|---:|---:|---:|---:|---:|
| No update | 27 | 23 | 25 | 75 | 39.06% |
| Full first mean update | 9 | 11 | 18 | 38 | 19.79% |
| Bounded forward, 1/32 of mean displacement | 23 | 18 | 31 | 72 | 37.50% |
| Tiny forward, 1/128 | 25 | 21 | 26 | 72 | 37.50% |
| Tiny reverse, −1/128 | 26 | 23 | 24 | 73 | 38.02% |

The full mean step reduced aggregate success by **19.27 percentage points**. The bounded step recovered **17.71 points**, about 92% of that loss, but finished below the unchanged actor. Fractions were chosen from analytic Gaussian KL on 64 disjoint collection episodes, without task outcomes. All 3,200 held-out states per collection were used. No fractions or thresholds were selected after seeing diagnostic performance.

| Collection | Exact joint-update KL | Mean-only full-step KL | Bounded-step KL | Tiny-forward minus tiny-reverse mean dense return | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| 90282001 | 9.8174 | 9.8186 | .00962 | +.00051 | [−1.586, +1.543] |
| 90282002 | 7.9912 | 7.9916 | .00783 | −.49833 | [−1.978, +.687] |
| 90282003 | 5.6845 | 5.6847 | .00555 | +.32113 | [−.0235, +.721] |

The initial production approximate-KL checks were near zero, as expected before any update. The resulting joint step nevertheless reached held-out KL 5.68–9.82, far above the configured .1 target. This is consistent with a check that stops subsequent updates without undoing the first accepted one. The mean movement dominates this displacement; changes in standard deviation are logged separately.

Equal-seed tiny-forward minus tiny-reverse mean dense return is approximately **−.0589**, with exploratory cluster/paired bootstrap 95% interval **[−.923, +.638]**. All individual intervals also include zero. With just three data draws and 64 diagnostic episodes each, this is limited evidence; it cannot establish a population-scale null or prove the direction is useless.

## What was learned

**Actor movement caused substantial immediate damage in this controlled setting.** The branches share the starting critic, noise, data-derived update direction and paired reset episodes; only the mean parameters move. This is stronger evidence than observing a high KL beside a later collapsed policy. The first mean step alone can hurt the starting behavior.

**Shrinking that displacement largely prevents the damage.** That supports a retention/optimization concern. It does not meet our stronger claim that a useful direction was merely taken too far: the required positive forward-versus-reverse return evidence is absent. We must not select the third collection's bounded 31/64 result and call that an improvement.

**This does not establish the cause of the entire historical collapse to zero.** It examines one initial update of an assisted actor. It does not diagnose later value errors, coverage changes, cumulative movement, from-scratch delayed learning or training-seed variability. Holding critic/noise fixed is appropriate for isolating actor movement; these branches are not alternative full PPO procedures.

Stochastic collection success was 248/1024, 249/1024 and 294/1024, with about 10.8% of sampled action coordinates clipped. Those are different episodes from the deterministic diagnostic resets. Their lower success cannot be treated as a paired causal estimate of exploration-noise damage.

## Reassessment against the original objective

Trust-region behavior deserves attention as **baseline correctness and retention**, especially for low-noise assisted policies. This probe does not yet justify making a trust-region variant the next large research campaign, and does not justify a learning-rate sweep. Any later intervention must demonstrate useful improvement beyond the no-update control, rather than merely preserve a clone.

The original question remains how to learn from scratch with at least 10× fewer interactions. The broader evidence still favors examining whether scarce useful experience can produce **reliable, reusable learning targets**, with a competent modern replay baseline as a comparison. Our failed short SAC adaptation is not that baseline. The next research decision should connect target quality/data reuse to measured early from-scratch behavior, with matched compute and T80/T90 curves. This is a direction for reassessment, not a queued experiment or a claim that replay will deliver tenfold gains.

No additional GPU experiment, benchmark query, teacher collection or autonomous variation is authorized by this result. The probe is stopped.

## Verification, accounting and preservation

- Reconstruction compared the literal hash-bound production code with an independent step on identical real data/RNG. All three batches matched parameters, gradients, Adam moments/steps, advantages, returns and NumPy/CPU/CUDA RNG exactly. The original logged hyperparameters and complete saved pre-update Agent were reused.
- One ordinary joint step per collection supplied the mean displacement. Three production-reference plus three reconstruction calls mean **six optimizer calls / 9,600 fitting rows**; these are accounted, not six independently trained agents. Forward hooks counted **595,200 actor rows / 1,026 calls** and **172,416 critic rows / 165 calls**, including KL-grid and diagnostic work.
- The assisted prefix remains **49,971,200 teacher transitions / 2,295.41 trainer seconds**, plus **25,600 demo transitions** and **5,000 BC updates**. It is not free initialization or a from-scratch result. No new teacher data was collected.
- Probe process: **46.663 seconds**. Worker function: **50.345 seconds**. Queue, image setup and billing are distinct. Reported charges are **$0.03206557** for the scientific app plus **$0.00014793** for the token-interrupted submission: **$0.03221350** combined, subject to metering lag. Total session reported spend is **$12.04770834**, with no reservations remaining under the shared $100 cap.
- Both apps are verified **stopped with zero active containers**. Authentication expired during the first submission; renewal and a stopped-app/no-attempt audit preceded the only scientific submission. No simulator work was repeated.
- The local [archive](../runs/stackcube-first-update-probe-20261028/archive-manifest.json) preserves **32 remote files / 54,824,504 bytes**, plus local source/plan/preflight provenance. Complete collection arrays, actual actions, masks, bootstraps, RNG states, first-step states, advantages, returns, phase trajectories, logs and per-episode diagnostics remain intact. Every remote file was verified by size and SHA256; native initial weights were only read inside the restricted candidate worker using `weights_only=True`.
- There were **zero development or confirmation queries**. The official task source, software pins, state/action interface, controller, horizon, physics and trusted evaluator are unchanged.

[Machine analysis](../runs/stackcube-first-update-probe-20261028/analysis.json), [raw report](../runs/stackcube-first-update-probe-20261028/run/report.json), [execution record](first-update-probe-execution-20261028.md), [approved plan](../deploy/modal/first-update-probe-plan.json).
