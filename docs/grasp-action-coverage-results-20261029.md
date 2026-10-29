# Frozen command coverage is highly uneven across grasp states

The [registered CPU coverage diagnostic](grasp-action-coverage-plan-20261029.md)
used all78 pretreatment eligible observations, without group/outcome filtering.
The physical experiment used a deterministic safe mean; this diagnostic examines
the underlying frozen SAC Gaussian from that same checkpoint. It measures
requested-setpoint and joint-target FK probabilities, **not physical lifting**.

| Requested gripper target proxy | Mean probability | Median probability |
|---|---:|---:|
|<=0mm |.143218 |2.205e-8 |
|<=10mm |.252536 |2.128e-5 |
|<=20mm |.324119 |.001373 |

The averages are dominated by a minority of states; most selected states have
little closing-proxy probability. These setpoints do not certify force, aperture
or secure grasp. The successful intervention used action−1; its exact point
probability in a continuous distribution is zero and supplies no rarity claim.

For the deliberately broad primary arm proxy (valid unmodified joint target,
TCP target rise>=5mm), joint<=10mm/up probability averaged.004155, median0 at
512 arm samples/state. Median0 is a finite MC result, not a proven probability0.
Each state's count/resolution/Wilson MC interval is retained. No narrow7D action
box or physical-return surrogate was selected after sampling.

Reflecting only the gripper's distribution preserves its conditional variance
and entropy but changes the mean. It increased the median closing proxy to
.998627; joint proxy median remained0 because arm coverage is also limited.
Independent uniform actions gave mean joint proxy.187280, but have different
marginal variance/entropy. These are diagnostic references, not matched learning
baselines or deployed interventions. Do not interpret the density ratio as10x
learning efficiency or multiply probabilities across future steps.

The geometry uses production `q+.1*a` targets and reports out-of-limit targets
as invalid rather than silently clipping. This is target kinematics, not actual
PD/contact motion. The current-state FK gate and safe actor parity passed.

Cost:.279s CPU diagnostic timer,78 actor plus78 safe policy rows,79,872 arm
vectors/559,104 scalar variates,468 stable Gaussian log-CDF evaluations and
79,950 analytic FK poses; zero fitting/simulator/protected-scoring calls.
Receipt:`research/discovery/grasp-action-coverage-20261029.json` retains every
state, log tail, counts and hashes. No learning method was admitted by this
measurement. Its next controlled question is cross-coordinate exploratory
dependence with identical conditional one-coordinate distributions.
