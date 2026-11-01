# Conservative MC validity check before reading the gate result

While the first fixed-state computation was running, independent review pointed
out that a zero coordinate-mixture median from512 samples/state could make the
point-estimate5x rule pass vacuously. Zero counts are unresolved probabilities,
not exact zeros. The root has not inspected this computation's output yet.

Keep the registered point gate and original receipt unchanged. Before any
interpretation or candidate handoff, additionally require a positive resolved
coordinate median and a working-MC bound check: median of vector proxy lower
bounds >=5times median of coordinate proxy upper bounds. Derive bounds from
existing counts only: original component Wilson intervals for the vector
mixture, new Wilson intervals for the coordinate arm estimate, multiplied by
the exact appropriate gripper probabilities. This weakens admission; it never
rescues a failed point gate. No resampling, new inference/FK/simulator query,
endpoint choice or training is authorized.

These working intervals are a sensitivity check, not simultaneous guaranteed
coverage over78states, a proof of a density inequality, or a robot learning
result. Report point-gate and validity-check outcomes separately. If unresolved,
the candidate handoff remains denied rather than describing an infinite gain.
