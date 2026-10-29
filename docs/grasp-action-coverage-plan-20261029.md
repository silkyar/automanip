# Frozen-policy action coverage at all pretreatment grasp states

CPU-only descriptive diagnostic, registered before inference/sampling. Use all78
eligible **pre-intervention** observations from the randomized probe, without
selecting by group, dose, lift or outcome. The same frozen256k actor/diagonal
Gaussian and public Panda URDF are disclosed assistance. No fitting, simulator
step, protected evaluation or candidate-method admission.

Compute exact Gaussian CDF/log-CDF probabilities of requested gripper setpoints
<=0,10,20mm, corresponding to normalized actions−.6,−.2,+.2 under the existing
`.015+.025*a` controller mapping. These are setpoint proxies, not calibrated jaw
aperture, force or validated secure-grasp thresholds. The successful intervention
used action−1; its exact probability under a continuous policy is zero, so do
not turn that point probability into an exploration argument.

For arms, exactly512 independent seven-dimensional samples per state, fixed
generator seed93314100. Apply production joint targets `q+.1*a` without
silently clipping to the assisted servo's margins. Mark URDF-limit violations.
Primary kinematic-up region: a valid target with predicted TCP rise>=5mm;
no rotation or lateral restriction. Secondary region additionally has lateral
shift<=15mm and rotation<=.05rad, as in the servo caps. These are target-FK
proxies, not actual PD/contact/physical outcomes. Verify current FK parity first.

Reference distributions: reflected gripper action with unchanged conditional
variance/entropy and arm distribution; and independent uniform actions in the
unchanged bounds (arm generator93314200). Reflection changes the mean; uniform
actions have different variance/entropy. Neither is a strong matched learning
baseline or a deployed policy. Do not select a distribution from outcomes.

Given each fixed state, arm and gripper are independent under the diagonal
Gaussian, so multiply their probabilities for this one-step proxy only.
Integrate the gripper analytically and Monte Carlo only the arms. Report every
state, zero-count MC resolution, Wilson MC intervals, log tails, limit violations,
and full probability summaries. Do not multiply across future steps, infer robot
returns, claim10x sample efficiency from density ratios or launch an exploration
method from coverage alone. All analytic FK/inference/sample counts and CPU time
must be logged.
