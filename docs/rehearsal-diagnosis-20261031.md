# Preserving behavior can also preserve a failure

The next local probes narrowed the interpretation of the closed Grow–Plan–Distill pilot. Neither passed its gate, so **no new robot query or exported policy followed**. Historical success remains unchanged. These probes reused the acquired-policy corpus and its existing held-out episode split; after several diagnostic uses, that split is not a fresh generalization test.

## Exact finger position was not sufficient

The earlier learned feature decoder had several millimetres of finger-position error. To test whether that alone explained poor distillation, the CPU readout was given the exact observed hold coordinate `(q7-.015)/.025` as an additional feature. Everything else stayed fixed: original actor raw means, 384 teacher rows, 1,024 rehearsal rows, weight 4 and ridge .001. This was a diagnostic feature expansion, not an exported policy.

| Parent | Original held-out finger-target RMSE | Exact coordinate available |
|---|---:|---:|
| 2 | 13.23 mm | 11.98 mm |
| 3 | 10.29 mm | 9.86 mm |

The gate required <1 mm and less than half the original error in both parents, without greater old executed-action change. It failed. Missing finger-coordinate precision alone is not a sufficient explanation under this fixed objective. Two linear solves took .148 CPU seconds; no optimizer updates, exports or simulator interactions occurred. [Registration](../cumulative/experiments/exact-coordinate-CPU-v1.json), [result](../cumulative/results/exact-coordinate-CPU-20261031.json).

## What the rehearsal labels preserve

An exploratory audit aligned saved physical stage flags to **pre-action** observations. Saved flags were recorded after each action, so they were shifted by one step; reset frames without measured prior flags were excluded. Source observations matched the original rehearsal dataset exactly.

Of the measured rehearsal fitting rows, **445/993 (44.8%)** in parent 2 and **502/991 (50.7%)** in parent 3 were already `on_top AND grasped`. Their preserved parent mean commands all requested further gripper closure. Among time steps 20–29, these states represented 297/338 and 315/334 rows. Thus the chosen first-30-step memory was not exclusively approach/grasp behavior: much of it preserved continued closure in geometrically placed states.

This is a potential objective conflict, not proof that those actions were wrong. None of those rehearsal rows also met the strict object-static predicate. Different histories and velocities can require different actions despite sharing a coarse stage. The [audit](../cumulative/results/rehearsal-stage-audit-20261031.json) reports all denominators, command directions and time bins. No fitting or physical queries were used for the audit.

## Stage selection versus simply weakening rehearsal

A fixed CPU comparison removed only the **gripper** rehearsal labels in those on-top/grasped states. All seven arm outputs retained their original losses and fitted coefficients. A control removed exactly the same number of gripper labels **at each time step**, using a fixed random seed. Both kept the original 1,024-row denominator, so neither received a hidden renormalization or parameter retune.

| Parent | Original error | Stage-selective error | Time-matched random removal |
|---|---:|---:|---:|
| 2 | 13.23 mm | 5.96 mm | 8.16 mm |
| 3 | 10.29 mm | 4.91 mm | 5.24 mm |

The gate required less than half the original error and at least 25% lower error than random removal in both parents, with no increased executed-action change on held-out non-on-top memory. Parent 2 missed the retention requirement (.0921 versus .0889 RMS change). Parent 3 missed the stage-specific advantage over random removal. **The joint gate failed.**

These results support the narrower observation that reducing rehearsal constraints improves the fit to late corrective examples. They do not establish a reliable benefit from the chosen stage rule, retention of actual grasp competence, or improved closed-loop task success. Four solves took .161 CPU seconds, with zero new transitions, optimizer updates, exports or robot assessments. [Registration](../cumulative/experiments/selective-rehearsal-CPU-v1.json), [result](../cumulative/results/selective-rehearsal-CPU-20261031.json).

## Implication for cumulative learning

The rehearsal endpoint already showed that discarding memory entirely destroyed actual grasping. The new audit shows why preserving every old action is also an imperfect definition of retaining a skill. A future mechanism needs to preserve useful competence while allowing corrections to unsuccessful behavior. This is a research requirement, not evidence that we have solved the problem with a reliability gate or a new loss.

Before implementing a larger learner, distinguish that requirement from merely lowering rehearsal weight or selecting late states. Any eventual robot comparison must include matched deletion/weight controls and independently measure retained task competence. The current failed gates do not authorize score-driven refits or another automatic GPU sweep.

We also reviewed existing gradient logs before pursuing a critic-clipping explanation. In the original 10.24M GAE runs, median actor norms were about 4.4–4.6 versus critic norms about .15–.16; critic domination was mostly an early transient. These are first-minibatch-per-update logs, not complete gradients, but they do not support persistent critic suppression as the main explanation. Prior temporal-noise and short-history studies were rechecked to avoid presenting those already tested ideas as new discoveries.
