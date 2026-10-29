# Lift-control probe: paired-state validity failed before intervention

The experiment did not answer whether the policy's grasp could support an
upward motion. It stopped exactly as registered when the replayed prefix failed
the measured-state matching gate. Tolerances were retained and the run was not
repeated. No gripper-close or upward-controller intervention was applied.

The baseline completed eight public 50-step episodes. Six lanes met the fixed
five-consecutive-grasp criterion by step 25. The next condition reset the same
seed batch and replayed the same commands. All 48 initial observation components
were bit-identical, and all 14 executed float32 action vectors were bit-identical
to the baseline prefix. Nevertheless, at the first scheduled intervention
(lane 4, after 14 steps), maximum corresponding position error was 6.559 mm,
qpos error .003798, qvel error 1.15143 and rotation error .137735 rad. These
exceeded the registered limits. Its preceding five grasp flags were still true;
the positional/velocity mismatch caused the stop.

The saved prefix audit found cube-B position drift .311 mm after the first step,
arm velocity exceeding tolerance after step 6, and arm/cube-A position exceeding
tolerance after step 10. Live FK remained accurate (maximum 1.743 micrometres,
2.342 microradians), so this was not a failed analytic-kinematics gate. Identical
measured reset states and actions do not establish identical hidden solver state.
The source of the physical drift is unresolved; do not label it an RNG bug,
incorrect controller reset, or a general failure of the frozen evaluator.

The narrow conclusion is that this repeated-reset prefix design did not achieve
the preregistered matching required for its paired intervention. There is no
grip-versus-command result, stack-performance result or sample-efficiency gain.
The existing low-alpha grasping result and historical PPO results are unchanged.

Accounting: 512 actual and attempted public transitions (400 baseline, 112
prefix), zero optimizer calls, zero protected evaluations, 520 safe policy rows,
1,032 analytic FK pose evaluations, zero Jacobian/IK evaluations. Child process
11.467 seconds; wrapper 15.751 seconds, on one L40S. The existing own-policy
256k-transition/195.43-second prefix and public-URDF assistance remain disclosed.

All seven remote files were downloaded and hash-verified in
`runs/lift-control-20261028/`, alongside exact sources/inputs, analysis and
`prefix-audit.json`. App `ap-AmV9OnsER4yeMkJlUmtn7n` was independently read back
STOPPED with zero tasks. Its $1 reservation was closed after archive/readback.
Reported cost increment was $0.01245698; reported shared-session total
$13.92165864, no active reservations. Metering may lag. No Runpod resources were
used and no new scientific training run is added to the 66-run ledger.

The next cheap check examines the frozen actor/online-critic gradients on existing
saved grasp states. This can describe the current surrogate, but cannot recover
unobserved physical counterfactual outcomes or reconstruct the historical PPO/SAC
training update.
