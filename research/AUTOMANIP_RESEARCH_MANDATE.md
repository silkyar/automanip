# AutoManip: Open-Ended Autonomous Discovery of Faster Robot Learning

**Status:** Research mandate for Codex, October 2026  
**Scope:** Extend the existing AutoManip repository; do not recreate it or overwrite existing experiments.

## Mission

We are **not** primarily trying to tune PPO or obtain a slightly higher final StackCube success rate. We want to test whether an autonomous AI research agent can discover **substantially more sample-efficient ways to learn manipulation**, potentially by combining ideas from mathematics, neuroscience, information theory, control, optimization, representation learning, and fields roboticists might not routinely examine.

Our ambitious target is **10× fewer simulator environment transitions to reach a specified success threshold** than the strongest validated reference method. A 100× improvement would be extraordinary, but is an aspiration, not a forecast. Prioritize genuine new learning mechanisms and decisive experiments over clever-sounding speculation. The agent may conclude that a hypothesis does not work.

**Primary research question:** Can an AI agent discover a reusable learning principle that reduces the experience required for robotic manipulation across multiple tasks?

## Where the project stands

We have a working ManiSkill state-based `StackCube-v1` training/evaluation setup and an autonomous Codex experimentation workflow. The original official PPO recipe had approximately **31.5% mean development success** and **32.2% confirmation success** across three training seeds, at about **25 million transitions per seed**. Subsequent autonomous experiments, with task, environment reward, network architecture, observations, action space, and evaluator held fixed, reported:

| Cumulative intervention | Mean development success | Confirmation success |
|---|---:|---:|
| Original official PPO recipe | 31.5% | 32.2% |
| Increase discount factor γ: 0.8 → 0.9 | 71.1% | Not reported |
| Add PPO entropy regularization: 0 → 0.003 | 83.6% | 82.1% |
| Continue training after first success instead of resetting immediately | **94.0%** | **PENDING — do not claim validated** |

In pre-change diagnostics, **98.4% of episodes achieved success at least once**, but only **83.6% were successful at the final step** of the 50-step evaluation. The no-early-reset candidate reportedly trained in 17.8 minutes versus 21.5 for its paired control, both at approximately 24.99M transitions. Treat the speed difference as a measurement to investigate (reset overhead and trajectory composition can confound it), not as proof of sample efficiency. Preserve exact source logs, checkpoints, seed-level metrics, commits, and failed experiments; the table above is only a summary.

**Immediate prerequisite:** Run the reserved confirmation evaluation for the 94.0% candidate with the already established independent evaluator, and clearly record per-seed results. Do not contaminate the reserved evaluation set through repeated tuning.

These successful modifications demonstrate that the agent can diagnose and improve a training recipe. **They are not yet evidence of a novel algorithm, 10× sample-efficiency gain, or generalizable scientific discovery.**

## New optimization objective

Shift the primary metric from final success after 25M steps to **the number of simulator transitions required to reliably attain a target held-out success rate**. Track thresholds such as 80%, 85%, and **90%**, alongside area under the learning curve and final success.

For each candidate method report:

- Environment transitions to each threshold, across multiple training seeds, with uncertainty; if not reached, report *not reached within budget*, not an extrapolated number.
- Learning curve under a common evaluation cadence and success definition.
- Wall-clock time to threshold, total GPU-hours/dollars, throughput, and time spent in simulation, inference, optimization, evaluation, and resets.
- Number of resets/episodes, environment steps, gradient steps, simulator queries, and any extra data used.
- Mean and lower-tail success across seeds; failure-phase diagnostics.

A method using fewer transitions but far more model computation should be described honestly as **sample-efficient but potentially compute-inefficient**. Prefer comparisons with matched observation/action access, reward information, and computational accounting.

## Research freedom

The agent may replace PPO completely. It may modify the **training-side** policy structure, loss/objectives, representation learning, value estimation, credit assignment, exploration, replay/off-policy updates, optimization, skill decomposition, temporal abstraction, curricula, and experience selection. It may pursue unexpected connections to other scientific areas, but should explain the proposed mechanism and make it falsifiable.

**Do not constrain the search to the interventions above.** Those are the previous baseline-building phase. The goal is to invent, adapt, or combine useful mechanisms and test them. Literature search is encouraged, including outside robotics; distinguish a genuinely novel result from a rediscovery of published work. Cite prior work and identify the closest established methods before claiming novelty.

The agent may create tiny analytical models, toy tasks, unit tests, or diagnostic simulators to disprove ideas cheaply before consuming the main training budget. Such auxiliary tests are not substitutes for the protected StackCube evaluation.

## What must stay protected

- The **evaluation environment, task success definition, physics configuration, fixed observation/action interface for comparable tracks, evaluation horizon, and evaluator code** are immutable to candidate training agents.
- **Development**, **confirmation**, and **final holdout** must remain separated. Use development feedback for iteration; access confirmation only for promotion; reserve final holdout for milestone comparisons.
- Never expose private seeds or privileged evaluator internals to the research agent; run evaluation in a separate trusted process/permission boundary. Loading a candidate should not mean executing unrestricted agent-supplied code in the evaluator.
- No reward hacking by modifying termination, success flags, evaluator metrics, simulator time step, episode horizon, or object physics at evaluation.
- Changing *training* rewards, curriculum, termination rules, or simulation distribution is permissible when explicitly logged, but evaluation must remain unchanged.
- Identify any privileged object state, demonstrations, pretraining, pretrained models, extra rollouts, planning lookahead, teacher policies, or additional simulator access. These can define **separate experimental tracks**, never hidden advantages in the state-only, from-scratch baseline track.
- Record versions, hardware, Git SHA, seeds, manifests, checkpoint hashes, and budgets to make every claimed result reproducible.

The protected evaluator's purpose is to measure autonomous progress, not to constrain the ideas the agent can test in training.

## Scientific method for autonomous research

For each proposed idea, record:

1. **Observation:** What measured limitation or contradiction motivates it?
2. **Mechanism:** Why might this idea change learning dynamics? Provide mathematical reasoning where useful.
3. **Novelty check:** Closest literature; what is actually new in this test?
4. **Prediction:** Which measurable learning-curve or failure metric should change, and in which direction?
5. **Falsification:** What result would disprove or weaken the idea?
6. **Small decisive test:** Cheapest experiment that isolates the mechanism.
7. **Outcome:** Include negative and null results, crashes, resource use, and uncertainty.
8. **Next hypothesis:** Update the research direction based on evidence rather than working mechanically through a fixed hyperparameter list.

Let the agent select its own research ideas and sequence within compute and safety bounds. Avoid requiring human approval for every ordinary experiment, but seek review before expensive campaigns, major infrastructure rewrites, or changes to the scientific comparison protocol.

## Comparison protocol

- First reproduce or preserve the best **independently confirmed** StackCube baseline. Do not use the 94.0% development candidate as the confirmed reference until its confirmation is complete.
- Match training seeds when feasible, then validate promising methods with **multiple independent training seeds**. Distinguish within-policy episode uncertainty from across-training-seed variation.
- Use cheap, prespecified screening tests to eliminate clear failures; promote apparent wins to appropriately powered multi-seed comparisons with fresh evaluation episodes.
- Report **fixed-transition** and **fixed-wall-clock/compute** results separately.
- Compare with competent conventional controls, including longer training, sensible PPO variants and (when appropriate) established off-policy / imitation / model-based baselines. Do not claim a scientific advance against an artificially weak baseline.
- For combinations of ideas, test components alone and together; use ablations to distinguish mechanism gains from incidental implementation differences.
- Audit simulator and evaluator validity whenever a seemingly dramatic improvement appears.
- After success on StackCube, **freeze the method** and try it without task-specific retuning on at least one held-out manipulation task (e.g. a different stacking, placement, or precision manipulation task), followed by broader generalization studies.

## Research-agent learning is a separate, later question

Preserve the *research trajectory*—hypotheses, code diffs, observations, decisions, unsuccessful experiments, and results—so that later we can test full-history versus distilled-lessons versus no-memory agents, or train an open-weight research model. This is distinct from improving the robot policy; **do not add Tinker/LLM fine-tuning to the immediate critical path**.

## Implementation guidance for Codex

1. **Inspect the existing repository** (`AGENTS.md`, `PROJECT_SPEC.md`, trainer, evaluator, reports, checkpoints, experiment logs). Reuse working infrastructure and keep PickCube and StackCube artifacts intact.
2. **Confirm the pending 94.0% candidate**, or report precisely what is required to do so. Never claim confirmation before results exist.
3. Propose the **smallest extensible architecture** that enables experiments beyond PPO without compromising the evaluator; separate candidate training artifacts from trusted evaluation logic.
4. Add transition-indexed evaluation curves, compute accounting, reproducibility metadata, and a structured research-decision log where missing.
5. Before large refactors or GPU-intensive research campaigns, **present the plan, scientific comparison design, risks, and rough compute budget for review**. Begin with cheap CPU-runnable tests and instrumented smoke tests.
6. Once infrastructure is verified, run an autonomous discovery campaign within an agreed budget, using the scientific loop above. Maintain a readable rolling research journal.

## What success looks like

A successful first milestone is **a confirmed, reproducible, sample-efficiency improvement over a strong baseline** with matched information and compute accounting. A more compelling milestone is a **large gain (ideally 10× fewer transitions to 90% success)** that survives ablations, multiple training seeds, and transfer to another task. The strongest result would be a clear new mechanism supported by evidence and situated accurately against prior research.

**The point is to expand what can be discovered—not to make a dashboard with a higher StackCube percentage.**
