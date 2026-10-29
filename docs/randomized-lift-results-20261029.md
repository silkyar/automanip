# Randomized intervention identifies arm–gripper coordination failure

The [new unpaired randomized diagnostic](randomized-lift-probe-plan-20261029.md)
passed its registered closure-under-upward-motion gate. Keeping the gripper
closed during the upward servo produced load-bearing rises in20/21 eligible
lanes, versus4/19 when the servo retained the policy's gripper command. Both
servo groups received the declared upward hand-motion dose in every eligible
lane. Neither arm-only motion nor gripper closure alone passed its directional
recovery threshold.

This answers a mechanical question about one frozen policy with disclosed
assistance. It is **not a learned-policy success rate, stacking result, new
algorithm or sample-efficiency gain**. The independent evaluator was not queried.

| Assigned condition | Eligible /32 | Hand rises>=4cm /eligible | Cube rises>=3cm /eligible | Cube raised and grasped at first hand dose | Reset-relative lifts /all32 |
|---|---:|---:|---:|---:|---:|
| A: policy arm and gripper |19 |1/19 |1/19 |1/19 |1/32 |
| B: policy arm, forced closure |19 |5/19 |1/19 |1/19 |1/32 |
| C: upward servo, policy gripper |19 |19/19 |4/19 |4/19 |4/32 |
| D: upward servo, forced closure |21 |21/21 |20/21 |20/21 |20/32 |

All128 assigned lanes, including50 ineligible lanes, remain in the raw analysis.
Eligibility required five consecutive official grasp flags completed by step25,
before the lane received any intervention. Random assignment was fixed before
observations; there was one vector reset, no repeated-state matching, resampling,
replacement, state copy or replay. The earlier failed paired protocol remains
failed and separately counted.

The eligible lift Wilson intervals were C8.5–43.3% and D77.3–99.2%. These are
working-Bernoulli descriptions of these cases, not guaranteed causal coverage or
population generalization from a single policy/vector batch. C's median
eligibility time was19 and D's18; both reached the hand-motion dose in roughly
7–9 steps. Reported differences are unpaired; no individual counterfactual or
universal necessity claim follows.

## The signal survives timing checks

All20 positive D cases already had a raised cube and a grasp flag at the **first**
hand-motion dose. All20 still had a grasp flag and>=3cm height gain at the final
step. C had4 such cases,16 drop transitions and4 final grasp flags; D had1 drop
transition and20 final grasp flags. The two separate maxima therefore do not
create the main positive result. Final mean finger-joint positions were about
35.1mm per finger in C versus19.5mm in D; these are observed joint coordinates,
not measured contact force or calibrated jaw aperture.

The narrow interpretation is that the closed-grip-plus-upward intervention
reliably rescued this stage in many of the selected public states. The policy's
gripper response during the upward trajectory is incompatible with reliable
lifting in most C cases. Grip-only and policy-arm behavior mostly did not lift.
The task's contact-based grasp flag does not establish that the policy will
maintain load-bearing closure through the next motion. The experiment does not
establish why the learned actor/critic chose these commands.

## Evidence and accounting

Independent review recomputed assignment, all first-eligibility times and the
outcomes directly from the archived NPZ. Observations were finite/contiguous;
only the final128 steps truncated. Live FK errors stayed below6.84micrometres
and2.33microradians. All source pins and thresholds were retained.

New public interactions:6,400 actual/attempted,50 vector steps/one reset;
zero optimizer, training or protected evaluation calls. CPU analytic work:
14,083 FK pose evaluations,1,283 Jacobian and1,283 inverse solves;6,400 frozen
safe policy rows. Diagnostic timer13.172s; wrapper17.366s on oneL40S. Assistance:
the existing own256k-transition/195.43s policy prefix plus public Panda URDF.
None of these diagnostic transitions enters learner training.

Archive:`runs/randomized-lift-20261029/`, six remote files/2,561,756 bytes,
hash-verified with exact source/policy/assignment, `analysis.json` and all lanes.
App`ap-lZnaO8xNnqa5kbSP1UVNiB` was independently verified STOPPED/zero tasks.
Its$1 reservation is closed. Reported increment$.01241026; shared session total
$14.07898385, remaining$85.92101615, no active reservations. Metering may lag;
no Runpod spending. Scientific training count remains68.

Next cheapest check: measure the frozen Gaussian's one-step closing-setpoint
tail probabilities on **all78 pretreatment eligible states**, then a declared
arm-command kinematic coverage estimate. These are coverage proxies, not
physical lifting probabilities. They may distinguish rare exploration from
simple deterministic preference, but do not alone admit a binary-gripper,
temporally correlated or assisted learning method.
