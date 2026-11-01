# Separate gripper-target bias from cumulative displacement exploration

The fresh-prefix search passed fidelity but yielded only one positive continuation
among 128 source states. Inspection found an implementation-specific coverage
problem in its quiet-action prior: the Panda arm consumes joint deltas, while
the gripper consumes absolute position targets with normalization
q_target = 0.015 + 0.025*a. Most N(0,0.05²) gripper samples requested narrower
positions than the already grasping fingers. Do not describe fixing this sampler
as a general learning breakthrough or PPO improvement.

Competing explanations: (1) the absolute proposal's closing bias starved useful
exploration; (2) useful motion additionally needs sustained displacement rather
than a single target perturbation; (3) adequate action excitation still does not
produce enough terminal-success labels. This is one controlled coordinate test,
not a sampler-scale sweep. Noise, arm commands, states, horizon and scoring stay
paired with the preserved absolute candidates.

Compare the eight existing absolute candidates against two new variants:

- Anchor-relative: gripper target = q_finger_at_step 30 + 0.025*z, fixed thereafter.
- Current-relative (primary): target = current q_finger + 0.025*z each step.

The same constant z~N(0,0.05²) is used in each matched candidate. All seven arm
commands are exactly the original quiet commands. Targets are translated back
to the existing normalized absolute command and clipped to its original bounds.
Only observation coordinate 7 supplies the controlled finger position; verify
it equals native proprioception and the pinned controller controls that joint.
No stage/success/geometry signal chooses actions. Positive and negative finger
increments are sampled symmetrically; there is no supplied release option.

An analytic perfect-tracking actuator toy verifies the inverse mapping, neutral
innovation, equal arm commands, and the accumulation distinction. It contains no
contact model and does not predict robot performance. Four focused CPU tests
passed before registration. Thirty-seven selected tests passed in 0.907 seconds.

Each of two existing parents uses the same 64 training initial states, base
Gaussian prefix and proposal seeds as the absolute test. Eight candidates per
variant and one repeated candidate per variant give 18 fresh subprocesses per
parent: 115,200 new transitions, including all prefixes and repeat checks. Two
L 40 S functions, <=300 seconds each, zero retries, zero optimizer calls. After
completion the full branching investigation costs 344,320 of the one-million
transition ceiling, leaving 655,680. Parent acquisition is additional disclosed
assistance, not from-scratch learning.

Admission requires prefix fidelity against the preserved absolute reference,
exactly equal arm action tapes, repeated full trajectories within unchanged
physical tolerances with identical actions, and matching first post-prefix
actions/states for anchor/current variants. No tolerance relaxation. Report
requested-opening fractions, final success and grasp/stability trajectories.
The current-relative bank is the prespecified candidate for subsequent learning:
at least 64/128 states with a successful continuation and at least 15 percentage-point
coverage gain over the absolute bank, with nonnegative gain in each parent.
Anchor-relative results distinguish centering from accumulation; do not select
the better variant post hoc for a benchmark claim.

If this fails, withhold distillation and reassess the sparse-label mechanism. If
it passes, register student fitting, an equal-total-physics unbranched collection
control, fresh policy-only scoring and grasp retention before claiming learning.
No private confirmation/final queries, controller changes or historical edits.
