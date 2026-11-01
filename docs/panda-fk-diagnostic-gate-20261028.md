# Panda public-URDF kinematic gate

Registered before computation, 2026-10-28. No simulator creation, robot action, learning, scorer or cloud call is authorized by this gate. The public URDF, robot identity and task base placement are explicit structural assistance, not acquired sample efficiency or free pretraining.

Parse the complete URDF parent chain from `panda_link0` to `panda_hand_tcp`, including all fixed origins/rotations and seven named arm joint axes. URDF fixed-axis RPY uses `Rz(yaw)*Ry(pitch)*Rx(roll)`. Compose each origin before its joint motion; obtain world joint axes/origins for the geometric Jacobian `[axis×(TCP−origin); axis]`. Do not substitute hand-coded DH geometry. Pinned Panda source names `panda_hand_tcp`; StackCube and TableSceneBuilder set base pose `[-.615,0,0]` with identity orientation. Stop if these identities are ambiguous.

Input is the previously inventory-verified 256k temperature trajectory. Verify all 3,200 pre-step qpos/TCP pairs, and separately all 3,200 post-step pairs needed for grasp-flag alignment. Observation positions/quaternions are TCP `[18:21]` /`[21:25]`, quaternion order wxyz, qpos arm `[0:7]`. Normalize recorded quaternions only to remove storage roundoff; reject norm error above 1e−4.

**Fixed acceptance tolerances:** maximum position error 1e−4 m, orientation rotation-angle error 1e−3 rad, Jacobian finite-difference maximum absolute error 1e−4. Centered finite-difference delta is 1e−6 rad. Declared flat pre-step pose indices are `[0,63,512,1023,1536,2047,2560,3199]`; check all seven canonical joint directions at each pose (56 directions, 112 perturbed analytic FK evaluations). Angular derivative uses world-frame `dR*Rᵀ`, matching the spatial angular Jacobian. No tolerance tuning or geometry patch after seeing results. Preserve any failure and stop the IK analysis.

If verification passes, assess one counterfactual damped 6D step on the 1,904 grasped post-step states: desired translation `[0,0,.005]` m, zero rotation vector, `dq=Jᵀ(JJᵀ+.05²I)⁻¹ dx`, all six row weights1. Translation/rotation units are mixed by this explicitly fixed weighting, so condition numbers/damping are convention-dependent. No joint clipping, iteration, orientation retargeting, collision planning or force model. Report singular values, linear residual motion, joint/action bounds and one analytic nonlinear FK step. This is not physical tracking evidence or admission of an upward controller.

Implementation: `research/discovery/panda_fk_probe.py`. Exclusive receipt: `research/discovery/panda-fk-probe-20261028.json`. Original simulator/training files remain unchanged. Results will be appended after the single computation.

## Result: kinematic verification passed

No geometry or tolerance was changed after computation. The parsed chain has seven revolute arm joints, followed by fixed `panda_joint8`, `panda_hand_joint` and `panda_hand_tcp_joint`. The TCP fixed transforms, including the hand rotation and tool displacement, came from the public URDF. The input hash matches the previously inventory-verified temperature archive; the receipt records URDF, task, table builder, Panda robot, Pose and analysis helper hashes.

| Check | Measured maximum | Frozen tolerance |
|---|---:|---:|
| TCP position, all 3,200 pre-step poses | 6.733e−6 m | 1e−4 m |
| TCP orientation, all 3,200 pre-step poses | 2.497e−6 rad | 1e−3 rad |
| TCP position/orientation, all 3,200 post-step poses | same maxima | same |
| Full 6×7 Jacobian centered differences, 8 poses | 2.213e−10 | 1e−4 |

Pre-step median position error was 1.048e−6 m and orientation error 8.355e−7 rad. Stored quaternion norm error was at most 1.230e−7. Finite-difference linear and angular maximum errors were 9.777e−11 and 2.213e−10 respectively. These measurements validate the coordinate, joint-axis and TCP identities to the declared tolerances; they do not claim bit-identical GPU physics or state reconstruction.

## Counterfactual upward-step conditioning

The 1,904 grasped post-step poses are repeated time points from 64 training episodes, not independent new trials. Under the declared unscaled translation/rotation row weighting, full-pose Jacobian condition number had median 15.75 and maximum 2,550.97. Translation-only condition number had median 3.99 and maximum 7.38. A large full-pose condition number did not prevent this particular damped upward direction; other directions can differ.

For the single +5 mm/zero-angular-step DLS calculation:

| Quantity | Median | Maximum or range |
|---|---:|---:|
| Predicted linear z displacement | 4.910 mm | 4.754–4.928 mm |
| Nonlinear FK z displacement after `q+dq` | 4.907 mm | 4.757–4.926 mm |
| Nonlinear FK lateral displacement | 0.098 mm | 0.809 mm maximum |
| Nonlinear FK orientation change | 1.920e−5 rad | 1.619e−4 rad maximum |
| Largest absolute joint delta per pose | 0.01010 rad | 0.01247 rad maximum |
| Largest normalized ±0.1-rad controller action per pose | 0.1010 | 0.1247 maximum |

There were zero predicted joint-limit violations and zero normalized action-bound violations, without clipping. The upward direction is therefore kinematically available in this calculation. Nothing here establishes collision clearance, load-bearing contact, grip security, torque availability, realized PD tracking, or eventual lifting. An assisted intervention using these structural priors still requires its own protocol review; this result admits no robot action or controller deployment.

## Preserved output bug and accounting

The first calculation failed only when serializing a NumPy boolean in the final receipt. Its exact source, initial receipt and error record are preserved as `research/discovery/panda-fk-probe-serialization-failure-v1.{py,json}` and `panda-fk-probe-serialization-failure-v1-error.json`. Numeric outcomes from that attempt were not durably retained. The sole source correction was wrapping `verification_passed` in Python `bool`; the numeric calculations, source geometry, selected indices, directions, damping and tolerances were unchanged. One CPU repeat followed this clear code correction; no tuning or third execution occurred.

The successful calculation used **0.03194 wall /0.02784 process CPU seconds** after imports: 8,416 analytic FK pose evaluations (6,400 archived pre/post poses +112 finite-difference perturbations +1,904 counterfactual advanced poses) and 6,400 geometric Jacobians. The lost first attempt is charged conservatively at at most another 8,416 FK evaluations /6,400 Jacobians, so the combined upper bounds are **16,832 FK evaluations /12,800 Jacobians**. Tool-observed full command wall times were 0.31283 seconds failed and 0.07495 seconds successful, totaling 0.38778 seconds; first-attempt process CPU time was not captured. These are analytic structural-prior calculations, not simulator interactions, fitted samples or training updates.

No new data collection, simulator creation/state query/action, actor inference, learning update, scorer/confirmation query, cloud call or trusted-file edit occurred. Original URDF, archived trajectory and scientific inputs were preserved.
