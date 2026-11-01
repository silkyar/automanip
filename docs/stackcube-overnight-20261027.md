# Bounded overnight StackCube research

Authorized on 2026-10-26, including unattended follow-ups. The first discount
sweep remains separate and its source snapshot stays unchanged. Discount 0.9
finished development evaluation at 72/128, 100/128 and 101/128 successes:
**273/384 = 71.09375%**, versus 121/384 = 31.5104% for the official 0.8 baseline.
This is development evidence, not confirmation or a private holdout result.

## Search and decisions

All runs start from random initialization and reuse the pinned official PPO
source, stock reward, task, 48-state/eight-action interface, controller, and
frozen independent evaluator. Training seeds remain 1, 2 and 3. No vision input,
architecture change, reward modification, confirmation query, or automatic
promotion is included in this queue.

The detached controller first waits for the original six-run sweep to finish
successfully. It verifies the staged source manifest and runs a GPU smoke test.
It then executes up to three phases:

1. **Refinement, 12 runs:** discount 0.875 or 0.925 with zero entropy bonus;
   entropy coefficient 0.001 or 0.003 at discount 0.9. Each is one change from
   the 0.9 anchor, repeated across three seeds at 25M requested transitions.
   Actual budget is 24,985,600 transitions per run. A new research anchor must
   gain at least two percentage points in mean success, with no paired training
   seed losing more than five points. Otherwise retain 0.9.
2. **Half-budget efficiency, nine runs:** selected recipe unchanged, doubled
   learning rate (0.0006 versus 0.0003), or four PPO epochs versus eight. Each
   is trained across three paired seeds for 12.5M requested transitions,
   actually 12,492,800. Compare methods within this phase at the same budget.
   Historical 25M comparisons are explicitly labelled budget-unmatched.
3. **Conditional verification, three runs:** if a changed half-budget recipe
   gains at least three success points, or retains success within three points
   while reducing trainer time by at least 10%, repeat it at the original 25M
   budget across three seeds. No paired seed may lose more than ten points
   to qualify. If neither variant qualifies, stop for a plateau.

Maximum additional scientific runs: **24**, plus a small smoke test. Independent
curves use the preselected official checkpoint cadence (0, 5.12M, 10.24M, etc.,
plus final), truncated appropriately for the shorter budget. Record training,
rollout, optimization and evaluation times separately. Selection for further
research never constitutes confirmation or promotion. Stop on an execution,
artifact, protocol, or evaluator error. Stop after this finite search even if
a variant is promising, leaving the evidence for review.

The working target is about 90% state-based success; training seconds are a
separate metric. Vision needs a separate, explicitly versioned observation
interface and baseline. It is not mixed into these state-based comparisons.

## Execution and spending

Current Pod: `2c1h36dlfn0t1f`, NVIDIA L4 in EU-RO-1, read back running at
$0.59/hour. The supervisor uses a conservative $0.60/hour envelope, a **$10
total Pod compute cap**, and a **16-hour deadline from the original Pod start**
(2026-10-27 00:09:55 UTC). This is a compute guard, not an account-wide billing
limit; persistent storage remains separate.

The controller runs on the GPU Pod and survives SSH disconnects. The shutdown
supervisor runs independently on this laptop under `caffeinate -is`; its API
key remains outside the GPU Pod. **Keep the laptop open, on power and online.**
Closing its lid can suspend the supervisor while GPU billing continues.

The supervisor watches `overnight.exit`, backs up the enclosing campaign,
checks the archive digest, requests stop and verifies the stopped state. A
deadline or cost stop takes priority over waiting for a backup; durable files
already mirrored to the network volume remain there.

Persistent parent:
`/workspace/automanip-research-20261026-credit-v7` on volume `zvx3i1sjnx`.

Useful files beneath it:

- `overnight/progress.json`: decisions, active phase and stop reason.
- `overnight/{tune,short,verify}/summary.json`: completed results.
- `overnight/{tune,short,verify}/records/*/run.json`: per-run provenance.
- `overnight-controller.log`: orchestration errors.
- `overnight-staged/source/source-manifest.json`: follow-up source hashes.

Local supervisor state:
`.runtime/autoresearch-20261027-supervisor.json`.
Archive target: `.runtime/overnight-20261027-results.tar.gz`.

Implementation: `src/automanip/overnight.py` contains the finite recipes and
decision rules; `scripts/run_overnight_research.py` orchestrates the phases;
`scripts/run_stackcube_experiments.py` retains the separate trainer, converter,
evaluator accounts and Landlock restrictions. Shared kernel/GPU limitations
still apply. Local tests passed 54 cases with one Linux-only skip; all 48
allowlisted Linux tests passed on the Pod, including Landlock enforcement.
