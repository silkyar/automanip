"""Post-baseline development replays; never changes benchmark scores or rewards."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import gzip
import json
from pathlib import Path

from automanip.evaluator import EpisodeOutcome, validate_actions
from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import load_policy
from automanip.stackcube import DEVELOPMENT_SEEDS, check_frozen_protocol
from automanip.stackcube_adapter import StackCubeAdapter


def reward_components(env):
    """Reconstruct each stock branch, and check against the untouched task method."""
    t = env.torch
    task = env.env
    info = task.evaluate()
    a, b = task.cubeA, task.cubeB
    tcp_distance = t.linalg.vector_norm(task.agent.tcp.pose.p - a.pose.p, dim=1)
    goal = b.pose.p.clone()
    goal[:, 2] += task.cube_half_size[2] * 2
    goal_distance = t.linalg.vector_norm(goal - a.pose.p, dim=1)
    reach = 2 * (1 - t.tanh(5 * tcp_distance))
    place = 1 - t.tanh(5 * goal_distance)
    width = t.sum(task.agent.robot.get_qpos()[:, -2:], dim=1)
    maximum_width = task.agent.robot.get_qlimits()[0, -1, 1] * 2
    opening = width / maximum_width
    opening[~info["is_cubeA_grasped"]] = 1
    speed = t.linalg.vector_norm(a.linear_velocity, dim=1)
    angular_speed = t.linalg.vector_norm(a.angular_velocity, dim=1)
    stillness = 1 - t.tanh(10 * speed + angular_speed)
    selected = reach.clone()
    grasped, on_top, success = (info[k] for k in
        ("is_cubeA_grasped", "is_cubeA_on_cubeB", "success"))
    selected[grasped] = (4 + place)[grasped]
    selected[on_top] = (6 + (opening + stillness) / 2)[on_top]
    selected[success] = 8
    official = task.compute_dense_reward(None, None, info)
    if not t.allclose(selected, official, atol=1e-6, rtol=0):
        raise RuntimeError("diagnostic reward reconstruction differs from stock task")
    values = dict(reach_raw=reach, placement_proximity_raw=place,
                  ungrasp_raw=opening, static_raw=stillness, selected_dense=selected,
                  selected_normalized=selected / 8, tcp_distance_m=tcp_distance,
                  goal_distance_m=goal_distance, gripper_width_m=width,
                  cube_speed_m_s=speed, cube_angular_speed_rad_s=angular_speed)
    packed = {k: v.detach().cpu().tolist() for k, v in values.items()}
    flags = {k: v.detach().cpu().tolist() for k, v in info.items()}
    result = []
    for i in range(env.num_envs):
        stage = ("success_override" if flags["success"][i] else "on_top_release_static"
                 if flags["is_cubeA_on_cubeB"][i] else "grasped_placement"
                 if flags["is_cubeA_grasped"][i] else "reach")
        result.append(dict(**{k: v[i] for k, v in packed.items()}, active_branch=stage))
    return result


def select_cases(results):
    """One successful development case per seed plus four distinct failure stages."""
    candidates = []
    for seed in (1, 2, 3):
        directory = results / "runs" / f"stackcube-seed-{seed}"
        report = json.loads((directory / "final-development.json").read_text())
        for index, episode in enumerate(report["episodes"]):
            candidates.append(dict(training_seed=seed, episode_index=index,
                                   reference_episode=episode, role=episode["diagnostics"]["first_unmet_stage"]))
    chosen = []
    for seed in (1, 2, 3):
        case = next((c for c in candidates if c["training_seed"] == seed and c["role"] == "success"), None)
        if case: chosen.append(case)
    for role in ("grasped_not_lifted", "placed_still_grasped", "released_not_stable", "lost_stack_before_horizon"):
        case = next((c for c in candidates if c["role"] == role), None)
        if case: chosen.append(case)
    return chosen


def plot_trace(trace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = trace["steps"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, layout="constrained")
    axes[0].plot(steps, [r["cubeA_pose"][2] for r in rows], label="Upper cube height")
    axes[0].axhline(rows[0]["cubeA_pose"][2] + .03, linestyle="--", color="grey", label="Lift diagnostic threshold")
    axes[0].set_ylabel("Height (m)")
    axes[1].plot(steps, [r["reward"]["goal_distance_m"] for r in rows], label="Distance to stacking pose")
    axes[1].plot(steps, [r["reward"]["cube_speed_m_s"] for r in rows], label="Cube speed (m/s)")
    axes[1].set_ylabel("Distance / speed")
    axes[2].plot(steps, [r["reward"]["gripper_width_m"] for r in rows], label="Gripper width (m)")
    axes[2].plot(steps[1:], [r["action"][-1] for r in rows[1:]], label="Normalized gripper action")
    axes[2].set_ylabel("Width / action")
    for key in ("reach_raw", "placement_proximity_raw", "ungrasp_raw", "static_raw", "selected_normalized"):
        axes[3].plot(steps, [r["reward"][key] for r in rows], label=key)
    axes[3].set(xlabel="Control step", ylabel="Reward terms")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(alpha=.2)
    fig.suptitle(trace["case_id"] + " | replay: " + trace["replay_diagnostics"]["first_unmet_stage"])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage_histories(results, output, protocol):
    """Trace every public development episode; confirmation is never replayed."""
    summaries = []
    for seed in (1, 2, 3):
        directory = results / "runs" / f"stackcube-seed-{seed}"
        reference = json.loads((directory / "final-development.json").read_text())
        env = StackCubeAdapter(8, "physx_cuda")
        try:
            artifact = load_policy(directory / "final-policy", env.schema)
            policy = NumpyPolicy(artifact, env.schema)
            with gzip.open(output / f"seed-{seed}-development-steps.jsonl.gz", "wt") as stream:
                for start in range(0, 128, 8):
                    observations = env.reset(DEVELOPMENT_SEEDS[start:start + 8])
                    first, changes, previous = [dict() for _ in range(8)], [[] for _ in range(8)], [None] * 8
                    ever_success, terminated = [False] * 8, [False] * 8
                    for step in range(1, 51):
                        actions = policy.act(observations)
                        validate_actions(actions, batch_size=8, action_dim=8)
                        observations, success, terms, trunc = env.step(actions)
                        if any(trunc) and step < 50: raise RuntimeError("unexpected diagnostic truncation")
                        ever_success = [a or b for a, b in zip(ever_success, success)]
                        terminated = [a or b for a, b in zip(terminated, terms)]
                        rewards = reward_components(env)
                        positions = [body.pose.p.detach().cpu().tolist() for body in (env.env.cubeA, env.env.cubeB)]
                        for slot in range(8):
                            frame = env.diagnostics[slot].last
                            flags = dict(grasp=frame.grasped, lift=frame.lifted, transport=frame.transported,
                                         placement=frame.on_top, release=frame.on_top and not frame.grasped,
                                         contact_free_placement=frame.on_top and frame.contact_free,
                                         stock_success=frame.success,
                                         stable_stack_10_steps=env.diagnostics[slot].stable_steps >= 10)
                            for key, value in flags.items():
                                if value and key not in first[slot]: first[slot][key] = step
                            if flags != previous[slot]:
                                changes[slot].append(dict(step=step, flags=flags))
                                previous[slot] = flags
                            row = dict(training_seed=seed, development_episode_index=start + slot, step=step,
                                       action=actions[slot], flags=flags, reward=rewards[slot],
                                       cubeA_position=positions[0][slot], cubeB_position=positions[1][slot])
                            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    for slot in range(8):
                        outcome = EpisodeOutcome(env.diagnostics[slot].last.success, ever_success[slot], terminated[slot], True)
                        actual = env.diagnostics[slot].report(outcome)
                        expected = reference["episodes"][start + slot]["diagnostics"]
                        summaries.append(dict(training_seed=seed, development_episode_index=start + slot,
                                              first_step_reached=first[slot], transitions=changes[slot],
                                              replay_diagnostics=actual, reference_matches_replay=actual == expected))
                    print("STAGE_HISTORY", seed, start + 8, "/ 128", flush=True)
        finally: env.close()
    report = dict(protocol_sha256=protocol, episode_count=len(summaries), episodes=summaries,
                  reference_matches=sum(s["reference_matches_replay"] for s in summaries),
                  note="Development-only descriptive replays; original frozen benchmark reports remain authoritative.")
    (output / "stage-history-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "episodes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = check_frozen_protocol()
    if (args.results / "batch.exit").read_text().strip() != "0":
        raise ValueError("baseline must complete before diagnostics")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = select_cases(args.results)
    groups = defaultdict(list)
    for case in cases:
        groups[case["training_seed"], case["episode_index"] // 8 * 8].append(case)
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    summaries = []
    for (seed, start), selected in groups.items():
        env = StackCubeAdapter(8, "physx_cuda")
        writers = {}
        traces = {}
        try:
            directory = args.results / "runs" / f"stackcube-seed-{seed}"
            artifact = load_policy(directory / "final-policy", env.schema)
            policy = NumpyPolicy(artifact, env.schema)
            observations = env.reset(DEVELOPMENT_SEEDS[start:start + 8])
            ever_success, terminated = [False] * 8, [False] * 8
            for case in selected:
                slot = case["episode_index"] - start
                name = f"seed-{seed}-dev-{case['episode_index']}-{case['role']}"
                traces[slot] = dict(case_id=name, training_seed=seed,
                                   development_episode_index=case["episode_index"],
                                   protocol_sha256=protocol, weights_sha256=artifact.weights_sha256,
                                   selection="first matching public development episode; descriptive, not representative sampling",
                                   reference_diagnostics=case["reference_episode"]["diagnostics"],
                                   reference_success=case["role"] == "success", steps=[])
                writers[slot] = imageio.get_writer(str(args.output / (name + ".mp4")),
                    fps=env.env.control_freq, codec="libx264", quality=7)
            actions = None
            for step in range(51):
                rewards = reward_components(env)
                task = env.env
                states = {k: v.detach().cpu().tolist() for k, v in dict(
                    cubeA_pose=task.cubeA.pose.raw_pose, cubeB_pose=task.cubeB.pose.raw_pose,
                    tcp_pose=task.agent.tcp.pose.raw_pose, joint_positions=task.agent.robot.get_qpos(),
                    joint_velocities=task.agent.robot.get_qvel(),
                    cubeA_linear_velocity=task.cubeA.linear_velocity,
                    cubeA_angular_velocity=task.cubeA.angular_velocity,
                    left_finger_force=task.scene.get_pairwise_contact_forces(task.agent.finger1_link, task.cubeA),
                    right_finger_force=task.scene.get_pairwise_contact_forces(task.agent.finger2_link, task.cubeA)).items()}
                images = task.render_rgb_array("render_camera")
                if hasattr(images, "detach"): images = images.detach().cpu().numpy()
                if images.shape != (8, 512, 512, 3): raise ValueError("unexpected diagnostic render shape")
                for slot, trace in traces.items():
                    tracker = env.diagnostics[slot]
                    trace["steps"].append(dict(step=step, action=None if actions is None else actions[slot],
                        **{k: v[slot] for k, v in states.items()}, reward=rewards[slot],
                        stage_flags=None if tracker.last is None else asdict(tracker.last),
                        consecutive_success_steps=tracker.stable_steps))
                    frame = np.asarray(images[slot], dtype=np.uint8).copy()
                    if frame.std() < 1: raise ValueError("diagnostic image is blank")
                    cv2.rectangle(frame, (0, 0), (512, 42), (20, 20, 20), -1)
                    cv2.putText(frame, f"Seed {seed} | dev {trace['development_episode_index']} | step {step}",
                                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    cv2.putText(frame, rewards[slot]["active_branch"], (8, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                    writers[slot].append_data(frame)
                    if step in (0, 25, 50): imageio.imwrite(args.output / f"{trace['case_id']}-frame-{step}.png", frame)
                if step == 50: break
                actions = policy.act(observations)
                validate_actions(actions, batch_size=8, action_dim=8)
                observations, success, terms, trunc = env.step(actions)
                if any(trunc) and step + 1 < 50: raise RuntimeError("unexpected diagnostic truncation")
                ever_success = [a or b for a, b in zip(ever_success, success)]
                terminated = [a or b for a, b in zip(terminated, terms)]
            for slot, trace in traces.items():
                outcome = EpisodeOutcome(env.diagnostics[slot].last.success, ever_success[slot], terminated[slot], True)
                trace["replay_diagnostics"] = env.diagnostics[slot].report(outcome)
                trace["replay_success"] = outcome.success_at_end
                trace["reference_matches_replay"] = (trace["reference_success"] == trace["replay_success"] and
                    trace["reference_diagnostics"] == trace["replay_diagnostics"])
                (args.output / (trace["case_id"] + ".json")).write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
                plot_trace(trace, args.output / (trace["case_id"] + "-trace.png"))
                summaries.append({k: v for k, v in trace.items() if k != "steps"})
                print("TRACE", trace["case_id"], "replay", trace["replay_diagnostics"]["first_unmet_stage"],
                      "matches", trace["reference_matches_replay"], flush=True)
        finally:
            for writer in writers.values(): writer.close()
            env.close()
    for video in args.output.glob("*.mp4"):
        capture = cv2.VideoCapture(str(video))
        try:
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != 51 or not capture.read()[0]:
                raise RuntimeError("diagnostic video failed decode/frame-count check")
        finally: capture.release()
    history = stage_histories(args.results, args.output, protocol)
    report = dict(protocol_sha256=protocol, cases=summaries,
                  stage_histories=history,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note="Additional development-only replays. Frozen baseline scores and confirmation episodes are unchanged.")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
