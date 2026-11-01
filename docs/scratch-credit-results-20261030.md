# Finite Monte Carlo targets did not improve from-scratch StackCube learning

October30,2026. [Registration](../cumulative/experiments/scratch-credit-v1.json); [machine summary](../cumulative/results/scratch-credit-20261030.json). This is a completed negative experiment, not a promoted recipe. Historical benchmarks and the50M reference remain unchanged.

The target choice that modestly helped a fixed assisted actor did **not** produce faster complete learning from scratch. Finite-MC PPO scored0/128 at every independently assessed checkpoint in all three training seeds. The tuned GAE control reached116/128 in seed1, while seeds2–3 remained0/128 within this budget. This variability is itself important: the best ten-minute seed must not stand in for a representative result.

| Method | Seed | Trainer seconds | Training transitions | Optimizer calls | Learner seconds | Final public successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tuned GAE PPO | 1 | 591.08 | 10,240,000 | 30,090 | 130.28 | 116/128 |
| Finite-MC PPO | 1 | 596.62 | 8,960,000 | 44,800 | 186.45 | 0/128 |
| Tuned GAE PPO | 2 | 553.24 | 10,240,000 | 27,875 | 114.11 | 0/128 |
| Finite-MC PPO | 2 | 596.73 | 9,625,600 | 48,128 | 184.23 | 0/128 |
| Tuned GAE PPO | 3 | 546.76 | 10,240,000 | 28,076 | 110.72 | 0/128 |
| Finite-MC PPO | 3 | 596.69 | 9,574,400 | 47,872 | 182.56 | 0/128 |

Both methods had the same600-second trainer-process and10.24M-interaction ceilings. MC used more learner time and optimizer calls, reaching fewer transitions. Thus equal configured ceilings did not mean equal consumed learner work. There is neither a sample-efficiency gain nor a compute-matched improvement to claim.

![All three learning curves](../artifacts/cumulative/scratch-credit-20261030/learning-curves.png)

The common prespecified comparison grid ends at8.192M because all three MC runs hit their wall limit before10.24M. GAE seed1 had5/128 there and116/128 at10.24M. Mean paired normalized-AUC difference over the common grid was−0.001628. No pair met the registered twofold T80 screen. T80/T90 require two consecutive observed crossings; one final high checkpoint does not establish a sustained crossing. Later MC learning beyond the cap is unknown; this result rejects this bounded intervention, not Monte Carlo estimation in general.

## What was held fixed

The control is the pinned strong precision-constant source (`cd7ee570…`), with gamma.9, entropy.003,8 PPO epochs,32 minibatches,1024 environments and full50-step episodes. The two generated programs differ only in their target block. MC replaces the whole bootstrapped GAE target with actual50-step reward-to-go and zero terminal tail. Trace choice and boundary treatment change together; this is not a lambda-only attribution. Actor/critic architecture, initialization, optimizer, PPO update body and task/reward are otherwise shared.

All six runs used fresh random initialization, with no teacher weights, demonstrations or previous experience. Actual environments reported unchanged normalized dense reward,20Hz control,100Hz simulation and50-step horizon. Runtime task-source, PhysX and dependency hashes/pins matched. Each trial used one L40S. Candidate training images excluded trusted evaluator/benchmark files; scoring used a separate image and read-only candidate mount, loading safe actor JSON only. The independent scorer inherited unchanged physical success and the frozen kernel.

All48 checkpoint assessments completed without execution errors. Each point used128 paired fresh **public-development** episodes; these are explicitly different from historical fixed benchmark-development episodes. No confirmation or final holdout was queried. Final public GAE mean was30.21%, rather than the best-seed90.63%. This does not replace the historical95.90% confirmation result at50M.

## Evidence, limitations and next diagnostic

The unchanged production GAE/update source was checked mechanically. CPU tests verified the target-only diff, exact finite-return toy values, rejection of incomplete episodes, paired analysis, nonuniform AUC and sustained-crossing/censoring rules. **18 selected CPU tests passed in0.100s.** These are separate from the six actual cloud training runs and independent GPU assessments.

Training logs agree with the scores: final stochastic success-at-end was80.96% in control seed1, approximately0.1–0.2% in controls2–3, and0% in all MC runs. Thus this is not merely an apparent collapse introduced by actor conversion. The assisted one-step finding remains valid within its original scope; it did not transfer to this learning procedure.

The next question is whether finite-horizon value prediction loses useful time information, or whether future-return variance dominates even with an adequate baseline. The existing critic observes state without remaining time. We will first test a constant-reward analytical example and cross-fitted clock/state baselines on existing training trajectories. Lower prediction error alone will not establish better policy learning. No larger-budget MC rescue or new broad sweep follows from this result.

Total new work:58,880,000 training transitions and307,200 independent scoring transitions, six scientific training runs, zero prior-acquisition transitions. Local archives contain1,282 training files /1,036,087,713bytes and18 scoring files /1,902,681bytes, with SHA256 manifests, safe actors, journals, source provenance and blind first64-lane samples from every rollout. Scalar timing was extracted with an isolated TensorBoard2.20 reader; its dependencies did not change the GPU environments or project locks.

Missing assets are explicit: these samples do not contain all1024 lanes, full critic checkpoints, every per-rollout actor, or actual post-autoreset terminal observations. They must not be treated as a complete learner resume, exact full-batch gradient reconstruction, or canonical terminal-transition replay store. No rollout videos were recorded in this experiment.
