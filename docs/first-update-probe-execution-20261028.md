# First-update probe: approval, reconstruction and execution record

The user approved exactly the [proposed first-update probe](research-strategy-diagnosis-20261028.md), with three conditions: stop if production reconstruction fails, limit conclusions to an assisted policy's first update, and stop/reassess afterward. Critic/noise-fixed response branches are diagnostic policies, not complete alternative PPO procedures.

## Reconstruction gate

The source is the archived production residual-noise PPO payload, SHA256 `c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200`. Its exact Agent, GAE block and optimization block are extracted without importing the simulator or executing its main program. An independent implementation is compared against the literal production calculation, stopping the reference optimizer immediately after its first accepted step.

The actual historical hyperparameter table was recovered from the original TensorBoard event's text-summary bytes. It confirms gamma .9, lambda .9, normalized minibatch advantages, unclipped value loss, finite-horizon GAE **false**, entropy .003, learning rate .0003, target KL .1, value coefficient .5, global gradient cap .5, reward scale 1, 1,024 environments, 50 steps and 32 minibatches. All reconstructed fields match this receipt. This distinction matters: a full 50-step episode does not imply the finite-horizon GAE option is enabled.

CPU gate passed in 1.40 seconds on fixed finite 50×1,024 arrays, including nonzero final-observation bootstrap values, an interior boundary and explicitly saturated action fixtures. Exact equality was checked for advantages, returns, first fitting indices, clipped gradients, parameters, Adam moments/steps and NumPy/CPU RNG. Both ordinary and finite-horizon GAE branches were checked; the actual experiment uses ordinary GAE. No CUDA result is inferred from these CPU checks. The same reference-versus-reconstruction comparison is required on each real CUDA batch before any response branch runs.

Fourteen targeted tests passed in 3.42 seconds, covering the new probe plus existing critic-completion and residual-noise checks. A deliberate wrong-update injection is rejected; it is a negative-control test, not an actual reconstruction mismatch. Source drift and insufficient disjoint holdout episodes also fail closed. No simulator interactions were used in these CPU checks.

## Fixed initialization and fresh data

Reuse the original pre-update full-Agent `ckpt_1.pt`, SHA256 `f41f449b6b8c9f655698c8d36fd8e808f0dc1dc1b31eef5aaea96e0b51453d58`, verified against its retained archive inventory. This preserves the historical random critic, BC actor and calibrated Gaussian noise exactly. Read it with `weights_only=True` only inside the isolated candidate-side artifact reader; no native checkpoint enters a trusted evaluator. The actor is separately checked against the numeric BC artifact SHA256 `4bab8b71061341ea985a6bb0384170c66a1b9a963d283942b84a99c2ad42c0a2`.

Collection reset/sampling seeds are 90282001–90282003; fresh paired diagnostic reset seeds are 90283001–90283003. These are public training probes, not benchmark episodes. They are three data draws from one existing actor, not three independent training seeds. The original built-in 400-step evaluation is omitted; this changes startup RNG history intentionally. Saved RNG states and fresh seeds make the new data explicit. Exactness concerns the production operator conditional on saved initialization and new inputs, not a replay of historical trajectories.

Training collection retains production auto-reset/mask behavior: raw Gaussian actions/logprobs are stored; only executed actions are clipped; truncation bootstraps use pre-reset final observations. Diagnostic rollouts disable auto-reset and retain the full 50-step horizon. The official task source and Linux package/GPU pins must match. No task, reward, evaluator, protected seeds or physics files are edited.

## One bounded execution

[Immutable launch plan](../deploy/modal/first-update-probe-plan.json), [reference/reconstruction helper](../research/discovery/first_update_probe.py), [GPU probe](../research/discovery/run_first_update_probe.py), [Modal adapter](../deploy/modal/first_update_probe.py), [offline analysis](../scripts/analyze_first_update_probe.py).

One restricted, network-blocked L40S worker; at most 201,600 new transitions including diagnostic rollouts; one $1 reservation within the existing shared $100 total. Package versions, task/source/input digests and all image inputs are bound by the plan. Submission lease is 580 seconds, function timeout 550 seconds, startup timeout 30 seconds and subprocess cap 510 seconds, reserving time to preserve output. No retry gets a fresh scientific budget. A durable exclusive attempt marker rejects replays.

Both production-reference and reconstruction updates count: at completion, six optimizer calls / 9,600 fitting rows total. Actor/critic forward hooks count actual rows/calls, including grid searches and all response branches. Identical fraction branches reuse exactly the same diagnostic report rather than spending extra simulator queries. Original/updated noise, joint Gaussian KL, mean-only branch KL, clipping, dense returns and full phase trajectories are recorded separately. Actor movement branches retain the original critic/noise.

Reported resource rates were checked on [Modal's pricing page](https://modal.com/pricing): the 600-second L40S/four-core/16-GiB compute envelope is approximately $0.378 at list rates, before storage/egress. The full reserved cap remains $1. This is an estimate; final provider metering is authoritative. Last refreshed existing campaign spend was $12.01549484 before this attempt, with no other reservations.

## Submission interruption

The first submission created app `ap-aOgs5ZVpEL4grVdeeVjNIi` and built the explicitly allowed source/artifact image layers. Modal then returned **AuthError: Token has expired**. A separate read-only lifecycle query received the same authentication error. There is no local scientific FunctionCall launch receipt. No scientific outcome or completed robot experiment is claimed.

The normal Modal browser token-renewal flow was opened for the user. Before resubmission, verify the original app is stopped and the output Volume has no scientific attempt marker; otherwise inspect the existing attempt instead of starting another. Keep the reservation until this is resolved. No broad experiment queue is enabled.

Resolved: renewal succeeded. The original app was verified stopped and all scientific attempt/result markers absent before submission. The single scientific call in app `ap-ciDCAvdT0ebDZPO4YRDRk7` completed successfully. Both apps are now verified stopped with zero active containers; the reservation is closed. See the [results and scientific reassessment](first-update-probe-results-20261028.md). No scientific run was repeated.

After the approved probe completes—or if its reconstruction fails—stop and reassess. A lower KL, preserved baseline behavior or assisted success is not a from-scratch sample-efficiency improvement.
