# Does a known hold coordinate improve learning from scratch?

The completed [Grow–Plan–Distill pilot](gpd-pilot-closeout-20261031.md) found a large local proposal-coordinate effect but failed to distill it into a reliable policy. This new study tests whether the same observation informs fresh PPO learning. It is a separate bounded experiment, not an extension of the closed pilot.

Panda's arm commands are relative and its gripper command is absolute. In the pinned controller, normalized command `a` requests finger position `q_target = .015 + .025*a`. Therefore a mean command of zero closes an initially open gripper toward .015 m. A mean command of `40*q7 - .6` requests its current position. This mapping uses an existing proprioceptive observation and known controller constants; it does not know the cube pose, task stage, success or future outcome.

The candidate adds that fixed mapping to a random learned residual. It occupies one isolated unit in each of the existing three tanh hidden layers, with a small internal gain .003 and an inverse output gain. Gradient masks preserve the path while PPO learns the ordinary residual. The exported artifact remains the same plain 48-input, 8-output MLP; no custom controller or code is supplied to the scorer.

## Competing explanations and controls

The hold prior might avoid premature closure and ease later release. Alternatively, the original closing bias might be useful for discovering grasps, and removing it could slow learning. A third possibility is that the previous effect was specific to late-stage search around already acquired behaviors and will not transfer to random initialization.

Three conditions share fresh training seed 41, the original critic and Gaussian standard deviation, strong gamma .9 / entropy .003 PPO, the official reward, full 50-step episodes and all instrumentation:

1. Original full 256-unit PPO.
2. Matched masked actor: 255 ordinary units per layer and the same frozen isolated path, with its output gain zero.
3. Hold actor: identical residual weights and masks, with the fixed hold path enabled.

The second control isolates the fixed prior from capacity/masking changes; the first prevents credit for beating only an impaired control. This is an explicitly engineered actuator prior, not a discovered reusable skill or an unassisted unknown-dynamics learner.

Each condition gets at most 4.096M transitions and 300 trainer-process seconds on one L40S. Actors at 0, .512M, 1.024M, 2.048M and 4.096M are scored in a separate image on 128 paired fresh public episodes each. No benchmark-development, confirmation or final seeds are used. Total ceilings are 12.288M training and 96,000 scoring transitions; automatic retries are disabled.

The predeclared single-seed screen requires at least +.10 normalized success AUC versus **both** controls and no reduction in final success over the common completed checkpoint grid. T80/T90 require two consecutive observed crossings; otherwise they remain censored. Stage metrics can explain failure but cannot substitute for the primary screen. Failure ends this bounded intervention; a passing signal needs matched seed replication before any promotion. Actual optimizer work, learner time, process time and scoring costs must be reported separately.

## Cheap mechanism gate and related work

The exact pinned production Agent passed the [CPU gate](../cumulative/results/gripper-hold-CPU-20261031.json): maximum hold-target approximation error was .23 micrometres over the gripper range; other action means were identical between matched actors; fixed weights did not move during 64 synthetic Adam updates with global clipping; critic and initial standard deviation matched. Unit tests also passed existing safe-policy export parity. These tests establish implementation properties, not learning benefit. Gaussian sampling and clipping still move the expected command inward near joint limits: with the original initial standard deviation, the upper-bound mean hold target yields an expected .03395 m command rather than exactly .04 m.

The general use of a fixed controller plus learned correction is established residual RL. Johannink et al. combine hand-designed feedback with an RL residual for contact-rich assembly; their simulator task starts with an object already grasped. That result does not establish that our neutral gripper prior improves learning a complete StackCube task. We read the method and experimental setup, not just the abstract. [Residual Reinforcement Learning for Robot Control](https://arxiv.org/html/1812.03201v2).

Eßer et al. report that action representations can affect initialization and behavior between timesteps, with task-dependent learning consequences. We checked their abstract; the linked PDF fetch failed, so this note does not claim a full-paper review. This motivates the competing explanations rather than guaranteeing a gain. [Action Space Design in Reinforcement Learning for Robot Motor Skills](https://proceedings.mlr.press/v270/esser25a.html).

The exact [registration](../cumulative/experiments/gripper-hold-v1.json) predates GPU launch. Historical results, pinned task and trusted evaluator remain unchanged.
