"""Create an offline report from the completed, unmodified baseline outputs."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def diagnostic_report(results, output):
    directory = results / "diagnostics"
    if not (directory / "summary.json").exists(): return None
    metadata = json.loads((directory / "summary.json").read_text())
    history = json.loads((directory / "stage-history-summary.json").read_text())
    summaries = []
    final_groups, lost_groups, velocity_limits = Counter(), Counter(), Counter()
    for seed in (1, 2, 3):
        original = json.loads((results / "runs" / f"stackcube-seed-{seed}" / "final-development.json").read_text())
        counts, rewards, held_commands = Counter(), Counter(), []
        released_steps = moving_release_steps = 0
        with gzip.open(directory / f"seed-{seed}-development-steps.jsonl.gz", "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"] == 50:
                    flags = row["flags"]
                    group = ("success" if flags["stock_success"] else "geometry_not_on_top"
                             if not flags["placement"] else "still_grasped" if flags["grasp"]
                             else "on_top_released_velocity_failure")
                    final_groups[group] += 1
                    if group == "on_top_released_velocity_failure":
                        velocity_limits["linear_threshold_exceeded"] += row["reward"]["cube_speed_m_s"] > .01
                        velocity_limits["angular_threshold_exceeded"] += row["reward"]["cube_angular_speed_rad_s"] > .5
                    label = original["episodes"][row["development_episode_index"]]["diagnostics"]["first_unmet_stage"]
                    if label == "lost_stack_before_horizon": lost_groups[group] += 1
                branch = row["reward"]["active_branch"]
                counts[branch] += 1
                rewards[branch] += row["reward"]["selected_normalized"]
                if row["flags"]["placement"] and row["flags"]["grasp"]:
                    held_commands.append(row["action"][-1])
                if row["flags"]["release"]:
                    released_steps += 1
                    moving_release_steps += (row["reward"]["cube_speed_m_s"] > .01 or
                                             row["reward"]["cube_angular_speed_rad_s"] > .5)
        if sum(counts.values()) != 6400: raise ValueError("incomplete development trajectory trace")
        first_steps = {}
        import statistics
        for stage in ("grasp", "lift", "placement", "release", "stock_success"):
            values = [e["first_step_reached"][stage] for e in history["episodes"]
                      if e["training_seed"] == seed and stage in e["first_step_reached"]]
            first_steps[stage] = dict(episodes_reaching=len(values),
                                     median_first_step_if_reached=statistics.median(values) if values else None)
        summaries.append(dict(training_seed=seed, control_steps=sum(counts.values()),
                              active_reward_branch_steps=dict(counts),
                              mean_normalized_reward_by_active_branch={k: rewards[k] / n for k, n in counts.items()},
                              held_placement_steps=len(held_commands),
                              mean_normalized_gripper_command_during_held_placement=(
                                  statistics.mean(held_commands) if held_commands else None),
                              strong_open_command_fraction_during_held_placement=(
                                  sum(a >= .5 for a in held_commands) / len(held_commands) if held_commands else None),
                              released_placement_steps=released_steps,
                              released_steps_exceeding_stock_velocity_threshold=moving_release_steps,
                              first_stage_times=first_steps))
    report = dict(runs=summaries, replay_matches=history["reference_matches"],
                  replay_episodes=history["episode_count"], video_cases=len(metadata["cases"]),
                  development_final_frame_groups=dict(final_groups),
                  lost_stock_success_final_frame_groups=dict(lost_groups),
                  released_on_top_velocity_failures=dict(velocity_limits),
                  video_reference_matches=sum(c["reference_matches_replay"] for c in metadata["cases"]),
                  note="Descriptive development replay statistics. Step-weighted quantities are not episode success rates or causal explanations.")
    (output / "diagnostic-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib.pyplot as plt
    for case in metadata["cases"]:
        trace = json.loads((directory / (case["case_id"] + ".json")).read_text())
        rows = trace["steps"]
        x = [r["step"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(x, [r["reward"]["reach_raw"] / 8 for r in rows], label="Reach branch")
        ax.plot(x, [(4 + r["reward"]["placement_proximity_raw"]) / 8 for r in rows], label="Grasped branch")
        ax.plot(x, [(6 + (r["reward"]["ungrasp_raw"] + r["reward"]["static_raw"]) / 2) / 8 for r in rows], label="On-top branch")
        ax.plot(x, [r["reward"]["selected_normalized"] for r in rows], color="black", linewidth=2.5, label="Applied stock reward")
        ax.set(xlabel="Control step", ylabel="Normalized reward", ylim=(0, 1.05),
               title=case["case_id"] + "\nBranch values shown before gating; only black curve is applied")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.2)
        fig.savefig(output / (case["case_id"] + "-reward-branches.png"), dpi=150)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = json.loads((args.results / "summary.json").read_text())
    if (args.results / "batch.exit").read_text().strip() != "0":
        raise ValueError("batch did not finish successfully")
    runs = sorted(batch["runs"], key=lambda r: r["seed"])
    if [r["seed"] for r in runs] != [1, 2, 3]:
        raise ValueError("expected three baseline seeds")
    stages = ("grasp", "lift", "transport", "placement", "release",
              "contact_free_placement", "stable_stack_10_steps")
    rows, failures, stage_counts = [], Counter(), Counter()
    curves = []
    for run in runs:
        if run["status"] != "complete" or run["environment_transitions"] != 24_985_600:
            raise ValueError("incomplete baseline seed")
        dev = run["final_evaluation"]["development"]
        confirm = run["final_evaluation"]["confirmation"]
        for report, count in ((dev, 128), (confirm, 512)):
            if report["episode_count"] != count or report["failed_episodes"]:
                raise ValueError("unexpected evaluation count or execution failure")
            if report["protocol_sha256"] != batch["protocol_sha256"]:
                raise ValueError("protocol mismatch")
        rows.append(dict(seed=run["seed"], environment_transitions=run["environment_transitions"],
                         trainer_process_seconds=run["trainer_process_seconds"],
                         rollout_seconds=run["rollout_seconds"],
                         optimization_seconds=run["optimization_seconds"],
                         development_successes=dev["successes_at_end"],
                         development_rate=dev["success_at_end_rate"],
                         confirmation_successes=confirm["successes_at_end"],
                         confirmation_rate=confirm["success_at_end_rate"],
                         confirmation_wilson_95=confirm["wilson_95_interval"],
                         confirmation_diagnostics=confirm["diagnostics_rates"],
                         confirmation_failure_stages=confirm["first_unmet_stages"],
                         weights_sha256=confirm["weights_sha256"]))
        failures.update(confirm["first_unmet_stages"])
        for episode in confirm["episodes"]:
            for stage in stages:
                stage_counts[stage] += int(episode["diagnostics"]["ever"][stage])
        directory = args.results / "runs" / f"stackcube-seed-{run['seed']}"
        curve = json.loads((directory / "independent-learning-curve.json").read_text())
        logged = json.loads((directory / "learning-curves.json").read_text())
        step_times = {int(event["value"]): event["wall_time"] - run["started_at"]
                      for event in logged["time/step"]}
        for point in curve:
            steps = point["environment_steps"]
            point["training_process_seconds"] = (
                0 if steps == 0 else run["trainer_process_seconds"]
                if steps == run["environment_transitions"] else step_times[steps])
        rows[-1]["first_observed_development_thresholds"] = {
            str(threshold): next((dict(environment_steps=p["environment_steps"],
                                      training_process_seconds=p["training_process_seconds"])
                                  for p in curve if p["report"]["success_at_end_rate"] >= threshold), None)
            for threshold in (.5, .9, .95)
        }
        curves.append(curve)
    total = sum(row["confirmation_successes"] for row in rows)
    report = dict(protocol_sha256=batch["protocol_sha256"], runs=rows,
                  total_training_transitions=sum(row["environment_transitions"] for row in rows),
                  total_trainer_process_seconds=sum(row["trainer_process_seconds"] for row in rows),
                  pooled_confirmation_successes=total, pooled_confirmation_episodes=1536,
                  pooled_confirmation_rate=total / 1536,
                  pooled_confirmation_failure_stages=dict(failures),
                  pooled_confirmation_diagnostic_counts=dict(stage_counts),
                  independent_curves=[dict(seed=row["seed"], points=[
                      dict(environment_steps=p["environment_steps"],
                           training_process_seconds=p["training_process_seconds"],
                           development_rate=p["report"]["success_at_end_rate"])
                      for p in curve]) for row, curve in zip(rows, curves)],
                  interpretation="Pooled episodes summarize these three trained policies; they do not measure uncertainty across all training seeds.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, diag) = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for row, curve in zip(rows, curves):
        x = [point["environment_steps"] / 1e6 for point in curve]
        y = [point["report"]["success_at_end_rate"] * 100 for point in curve]
        ax.plot(x, y, marker="o", label=f"Seed {row['seed']}")
    ax.set(xlabel="Training transitions (millions)", ylabel="Final success (%)",
           title="Frozen development evaluation · 128 episodes", ylim=(-2, 102))
    ax.axhline(60, color="grey", linestyle="--", linewidth=1, label="Proposed research target")
    ax.grid(alpha=.2)
    ax.legend()
    labels = ("Grasp", "Lift", "Transport", "Placement", "Release",
              "Contact-free\nplacement", "Stable for\n10 steps")
    for i, row in enumerate(rows):
        diag.bar([j + (i - 1) * .25 for j in range(len(stages))],
                 [row["confirmation_diagnostics"][k] * 100 for k in stages],
                 width=.24, label=f"Seed {row['seed']}")
    diag.set_xticks(range(len(stages)), labels, rotation=35, ha="right")
    diag.set(ylabel="Episodes reaching diagnostic (%)", ylim=(0, 102),
             title="Independent confirmation · 512 episodes per seed")
    diag.grid(axis="y", alpha=.2)
    diag.legend()
    fig.savefig(args.output / "learning-and-diagnostics.png", dpi=180)
    fig.savefig(args.output / "learning-and-diagnostics.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for row, curve in zip(rows, curves):
        ax.plot([p["training_process_seconds"] / 60 for p in curve],
                [p["report"]["success_at_end_rate"] * 100 for p in curve],
                marker="o", label=f"Seed {row['seed']}")
    ax.set(xlabel="Elapsed training process time (minutes)", ylabel="Final success (%)",
           title="StackCube official PPO · L4 · frozen development episodes", ylim=(-2, 102))
    ax.axhline(60, color="grey", linestyle="--", linewidth=1, label="Proposed research target")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.output / "learning-by-training-time.png", dpi=180)
    plt.close(fig)
    diagnostics = diagnostic_report(args.results, args.output)
    if diagnostics: report["diagnostic_replay_summary"] = diagnostics
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
