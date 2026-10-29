# Lift-control diagnostic: prepared locally, not launched

The user cancelled cloud migration and requested continued local Codex work.
Codex remains on the laptop. This does not migrate the agent or automatically
resume its paused persistent research goal.

The registered protocol remains [the four-condition mechanical probe](lift-control-probe-plan-20261028.md).
Implementation: `research/discovery/lift_control_robot.py`; isolated bounded
Modal adapter: `deploy/modal/lift_control.py`; all-lane analysis:
`scripts/analyze_lift_control.py`. Input plan is prepared with `admitted=false`
in `.runtime/lift-control-inputs/plan.json`. No reservation or launch was made.

Independent review identified two issues before admission. Matching measured
poses alone does not establish that repeated conditions still have the five
contact flags that defined eligibility. Each intervention now checks and saves
its own preceding five flags. Also, saving only when a condition exits loses
partial data if a simulator call hangs. Initial/partial trajectories are now
saved atomically after reset and every returned step; the attempted-transition
counter is saved before every simulator call. The reviewer reread these fixes
and reported no remaining substantive blocker.

CPU checks validated safe inference on 64 stored observation rows and deliberate
qpos, qvel, position and orientation matching failures. Further checks executed
the exact production AST fragments for contact eligibility and partial saving:
missing/short contact history fails, a partial NPZ is readable, and attempted
counts greater than returned counts survive. Synthetic analysis checks suppress
claims for invalid or too-small samples. These checks used zero simulator
transitions and zero optimizer updates; they are not GPU tests.

Receipts: `.runtime/lift-worker-cpu-review.json` (initial worker version),
`.runtime/lift-reviewed-edge-cases.json` (final reviewed worker hash), and
`.runtime/lift-analysis-cpu-review.json`. The prepared input plan was refreshed
to the final reviewed source hashes. Historical receipts remain intact.

Automatic approval review rejected the read-only Modal billing refresh because
it interpreted autonomous research as still paused. No alternate execution path
was attempted. An explicit question to resume bounded Modal research under the
existing total $100 cap is pending. Budget freshness, reservation and exact-source
admission must precede a launch if authorization is resolved. Last verified
research jobs were stopped; there is no new live-job claim or fresh spend figure.
