# Learning where a correction should persist

The randomized robot evidence identifies a maintenance bottleneck in two fixed failed policies: opening and zero arm delta together preserve many stacks that the original actor subsequently destroys. The next objective is to learn where a corrective behavior works from experience, without supplying the geometric trigger. **No new robot policy has yet been learned or promoted.**

The fixed base actors each cost8.192Mtraining transitions. The candidate zero-arm/open-gripper behavior is a supplied action prior. A later learned selector must disclose both; it cannot be reported as learning StackCube from scratch. Historical95.90%confirmation remains unchanged, and the10×objective remains unmet.

## What the local tests found

In a scalar50stepstopping problem, a fixed progressing controller often passes through a successful region and then leaves it. Randomly choosing continuation or a persistent hold at a random time produces training outcomes. The learner receives the state, chosen action and terminal outcome, without the target boundary.

The first two logistic outcome models failed to activate the gate: their maximum predicted hold success was only.523–.599. All three rollout-calibration gates rejected them. A one-factor check removed coefficient regularization while retaining data, features, optimizer budget and trigger thresholds. Two of three candidates then passed rollout calibration; one reached87.18%versus12.24%base success. But another lost4.35%of previously successful base episodes, failing the99%retention requirement, and the third was withheld by calibration. **The registered three-seed screen still failed.** Removing underconfidence did not establish reliable improvement.

A simpler support construction then passed a deliberately limited test: use the interval spanned by successful corrective training outcomes. All three toy seeds matched the oracle, approximately89.3–89.6%on the original distribution and82.0–82.5%on wider initial conditions, with100%observed base-success retention. Shuffled outcomes were rejected. This is an assisted toy result, not a robot learning curve.

The reason is analytical. Assume the corrective action leaves the state fixed and the true success region is convex. Successful correction examples lie inside that region, so their convex hull does too. A switch inside this hull enters an invariant successful state; outside it the old policy continues. A trajectory that previously succeeded therefore either remains unchanged or switches to an already successful invariant state. The gate can be stateless because corrected states remain inside its region. Our latched and stateless toy executions agreed exactly.

Those assumptions matter more than the high toy score. A disconnected-goal counterexample made the same hull switch fail0/128where the base succeeded128/128. Robot contact dynamics are nonlinear, zero joint delta does not freeze a cube, object velocity is omitted from the48observations, and a soft neural approximation need not preserve the same region. **A convex hull of48dimensional robot states is not an implemented or justified repair.**

Exact registered plans and source/data/result archives are retained:

- `cumulative/experiments/persistent-switch-toy-v1.json`: first model failure, three unit tests,0.592seconds.
- `cumulative/experiments/persistent-switch-regularization-v1.json`: one-factor partial improvement with failed retention screen,0.709seconds.
- `cumulative/experiments/invariant-support-toy-v1.json`: narrow invariant-set property and counterexample, two unit tests,0.031seconds. Reconstructed original training data are now stored in non-pickle NPZ form.

All were CPU-only. The plans separately count generated and replayed scalar transitions, optimizer/fitting work and calibration/test rollouts; none is robot experience. They supply mechanisms and failure cases, not sample-efficiency claims.

## The next robot experiment

The [random-switch training-corpus protocol](../cumulative/experiments/random-switch-training-v1.json) is one bounded data collection, not a training sweep. Two frozen sources,2,048episodes each,204,800total transitions, zero policy updates. Each64lane batch randomly assigns32episodes to continue the original policy and32to the correction, with decision times uniformly selected from10..44before rollout. No geometry, outcome or privileged signal triggers correction.

The isolated trainer uses only the pinned task, safe parent actor files and the collector. It contains no evaluator or benchmark seeds. Full51observation/50action episodes, failed outcomes, behavior parameters, random assignments and source costs are preserved. The official task already rewards ungrasping and staticity; this experiment does not add another reward for them.

Existing257dimensional actor features feed small outcome models on CPU. Whole vector batches are split into fitting, calibration and testing. Held-batch Brier scores are compared with constant and decision-time baselines, plus shuffled-label controls. Privileged object velocities are recorded in separately named diagnostic columns and explicitly counted, but are excluded from model features. This can test a particular predictor's usefulness; a failure would not prove that current observations contain no usable information.

A predictive result would still be insufficient to deploy a gate. Random-time outcome prediction differs from first-trigger closed-loop execution. A later candidate must pass fresh rollout calibration, preserve earlier behavior, and fit the existing safe MLP artifact contract. No arbitrary gate/controller Python can enter the frozen evaluator. Direct latched-gate execution is currently unsupported by that artifact contract; compilation/distillation and its closed-loop fidelity remain missing work.

The first learning comparison should share the same base and fixed action prior across a learned outcome selector and an option-policy-gradient comparator. Ordinary actor warm-start PPO is a separate contextual baseline; exact learner continuation is unavailable because these source checkpoints lack full critic/optimizer state. Prefix acquisition, exploration, model fitting, policy compilation, calibration, scoring and failures must all be counted. New held-task transfer and retention must be measured before claiming a reusable robot skill.

## Literature context

[SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) constrains departures from an available baseline where state-action data are insufficient. Its finite-MDP guarantees do not directly certify our continuous, partially observed robot setting. [Learning MPC for linear systems](https://arxiv.org/pdf/1702.07064) builds invariant sets and terminal costs from previous successful trajectories under explicit linear dynamics and convex constraints. This provides relevant precedent for reusing successful experience, not permission to assume contact-state interpolation is safe.

The [focused temporal-abstraction reading](literature/temporal-maintenance-followup-20261030.md) covers Option-Critic, Q-chunking and SEAR. Their assistance, compute and execution assumptions differ. We are using these ideas to formulate a minimal causal learning test, without claiming a new algorithm or importing an uncounted prior dataset.
