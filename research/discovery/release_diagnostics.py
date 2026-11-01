"""Passive training-episode diagnostics, separate from protected scoring.

Inputs are the public upstream task flags and existing PPO rollout scalars.
Do not pass reset observations as final successors or use these outputs as rewards.
"""
from __future__ import annotations

import math


def summarize_episode(snapshots: list[dict], transitions: list[dict], gamma: float = .9) -> dict:
    """Summarize one complete 50-step training episode, keeping every outcome.

    Snapshot0 precedes action1; snapshot[t] is the pre-reset successor of action t.
    V/next_V are pre-update estimates at those exact states. Truncation bootstraps;
    a true terminal does not. TD residuals are noisy diagnostics, not true values.
    """
    if len(snapshots) != 51 or len(transitions) != 50:
        raise ValueError("Expected initial state plus all 50 pre-reset successors")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Invalid gamma")
    flags = ("on_stack", "grasped", "static")
    measurements = ("xy_offset", "z_error", "linear_speed", "angular_speed")
    for t, state in enumerate(snapshots):
        if state["step"] != t or any(type(state[k]) is not bool for k in flags):
            raise ValueError("Invalid snapshot index or public flags")
        if any(not math.isfinite(state[k]) or state[k] < 0 for k in measurements):
            raise ValueError("Offsets, absolute z error and speed must be finite/nonnegative")
    for t, row in enumerate(transitions, 1):
        if row["step"] != t:
            raise ValueError("Transition order mismatch")
        if any(type(row[k]) is not bool for k in ("true_terminal", "truncated")):
            raise ValueError("Explicit terminal and truncation masks required")
        if t < 50 and (row["true_terminal"] or row["truncated"]):
            raise ValueError("Episode boundary inside fixed full-horizon record")
        if any(not math.isfinite(row[k]) for k in ("reward", "value", "next_value", "advantage", "normalized_advantage")):
            raise ValueError("Nonfinite rollout scalar")
    if not (transitions[-1]["true_terminal"] or transitions[-1]["truncated"]):
        raise ValueError("Final transition must declare its episode boundary")

    def upstream_success(state):
        return state["on_stack"] and state["static"] and not state["grasped"]

    releases = []
    groups = {}
    for t, row in enumerate(transitions, 1):
        previous, current = snapshots[t-1], snapshots[t]
        if previous["grasped"] and not current["grasped"]:
            releases.append(dict(
                step=t, remaining_steps=50-t,
                on_stack_before=previous["on_stack"], on_stack_after=current["on_stack"],
                xy_offset_after=current["xy_offset"], z_error_after=current["z_error"],
                linear_speed_after=current["linear_speed"], angular_speed_after=current["angular_speed"],
            ))
        phase = ("off_stack" if not previous["on_stack"] else
                 "on_stack_grasped" if previous["grasped"] else
                 "on_stack_released_static" if previous["static"] else "on_stack_released_moving")
        bucket = groups.setdefault(phase, [])
        delta = row["reward"] + gamma * (not row["true_terminal"]) * row["next_value"] - row["value"]
        bucket.append((delta, row["advantage"], row["normalized_advantage"]))
    first_on = next((s["step"] for s in snapshots if s["on_stack"]), None)
    first_success = next((s["step"] for s in snapshots if upstream_success(s)), None)
    first_release = releases[0]["step"] if releases else None
    return dict(
        scope="Training-only public-task diagnostics; not protected evaluation or causal attribution",
        first_on_stack_step=first_on, releases=releases,
        first_on_stack_to_first_release_steps=(None if first_on is None or first_release is None else first_release-first_on),
        first_upstream_success_step=first_success,
        upstream_success_lost_later=(first_success is not None and any(not upstream_success(s) for s in snapshots[first_success+1:])),
        geometry_lost_after_release=any(not s["on_stack"] for r in releases if r["on_stack_after"] for s in snapshots[r["step"]+1:]),
        grasp_resumed_after_release=any(s["grasped"] for r in releases for s in snapshots[r["step"]+1:]),
        final_upstream_success=upstream_success(snapshots[-1]),
        phase_statistics={phase: dict(
            count=len(rows), mean_td_residual=sum(r[0] for r in rows)/len(rows),
            mean_abs_td_residual=sum(abs(r[0]) for r in rows)/len(rows),
            mean_raw_advantage=sum(r[1] for r in rows)/len(rows),
            raw_to_normalized_advantage_sign_changes=sum((r[1] > 0) != (r[2] > 0) for r in rows),
        ) for phase, rows in groups.items()},
    )
