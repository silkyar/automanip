"""Read-only archived training-trajectory analysis; no simulator or fitting.

NPZ input is numeric-only (allow_pickle=False). Actor evaluation is explicit
NumPy algebra, never execution of archived learner code. Frozen-checkpoint
outputs on behavior observations are counterfactual, not behavior parameters.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT/"runs/temperature-sufficiency-20261028"
OUTPUT = ROOT/"research/discovery/temperature-grasp-diagnostic-20261028.json"
API = ROOT/".venv311/lib/python3.11/site-packages/mani_skill"
SLICES = dict(qpos=[0, 9], qvel=[9, 18], tcp_pose=[18, 25], cubeA_pose=[25, 32],
              cubeB_pose=[32, 39], tcp_to_cubeA_pos=[39, 42],
              tcp_to_cubeB_pos=[42, 45], cubeA_to_cubeB_pos=[45, 48])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(value):
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                minimum=float(x.min()), q25=float(np.quantile(x, .25)),
                median=float(np.median(x)), q75=float(np.quantile(x, .75)),
                q90=float(np.quantile(x, .9)), maximum=float(x.max()))


def numeric_npz(path):
    if path.stat().st_size > 32*1024*1024:
        raise ValueError("Input file bound")
    with zipfile.ZipFile(path) as z:
        if sum(v.file_size for v in z.infolist()) > 64*1024*1024:
            raise ValueError("Expanded archive bound")
        if any("/" in v.filename or not v.filename.endswith(".npy") for v in z.infolist()):
            raise ValueError("Unexpected NPZ members")
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    for key, value in result.items():
        if key == "note":
            assert value.dtype.kind == "U"
        elif value.dtype.kind not in "bifu" or not np.isfinite(value).all():
            raise ValueError("Not finite numeric arrays: "+key)
    return result


def actor_outputs(state, obs):
    def linear(x, stem):
        return x@state[stem+".weight"].astype(np.float64).T+state[stem+".bias"]
    x = np.tanh(linear(obs.astype(np.float64), "actor.backbone.0"))
    x = np.tanh(linear(x, "actor.backbone.2"))
    mean = linear(x, "actor.mean")
    logstd = -5+3.5*(np.tanh(linear(x, "actor.logstd"))+1)
    return mean, logstd


def run():
    if OUTPUT.exists():
        raise FileExistsError("Preserve existing analysis receipt")
    started, cpu_started = time.monotonic(), time.process_time()
    inventory = {v["path"]: v for v in json.loads((ARCHIVE/"inventory.json").read_text())}
    inputs = {}
    def checked(name):
        path = ARCHIVE/name
        if path.is_symlink() or not path.resolve().is_relative_to(ARCHIVE.resolve()):
            raise ValueError("Input path bounds")
        entry = inventory[name]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Archived input identity: "+name)
        inputs[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return path
    trajectory_path = checked("run/actor-step-256000-trajectory.npz")
    state_path = checked("run/actor-step-256000-state.npz")
    final_path = checked("run/actor-final-trajectory.npz")
    checked("run/actor-final-state.npz")
    data, state = numeric_npz(trajectory_path), numeric_npz(state_path)
    assert sha(trajectory_path) == sha(final_path)
    assert sha(state_path) == sha(ARCHIVE/"run/actor-final-state.npz")
    obs, nxt, action = (data[k].astype(np.float64) for k in ("obs", "next_obs", "action"))
    flags = data["stage_flags"]
    assert obs.shape == nxt.shape == (50, 64, 48) and action.shape == (50, 64, 8)
    assert flags.shape == (50, 64, 4) and flags.dtype == bool
    assert data["truncated"].shape == data["stop"].shape == (50, 64)
    assert not data["truncated"][:-1].any() and data["truncated"][-1].all() and not data["stop"].any()
    assert np.max(np.abs(obs[1:]-nxt[:-1])) == 0
    assert np.max(np.abs(action)) <= 1
    relative_errors = {}
    for name, x in (("obs", obs), ("next_obs", nxt)):
        for label, stored, calculated in (("tcp_to_A", x[..., 39:42], x[..., 25:28]-x[..., 18:21]),
                                          ("tcp_to_B", x[..., 42:45], x[..., 32:35]-x[..., 18:21]),
                                          ("A_to_B", x[..., 45:48], x[..., 32:35]-x[..., 25:28])):
            relative_errors[name+"/"+label] = float(np.abs(stored-calculated).max())
    assert max(relative_errors.values()) <= 1e-6
    source_files = ["envs/tasks/tabletop/stack_cube.py", "envs/sapien_env.py", "agents/base_agent.py",
                    "utils/common.py", "utils/structs/pose.py", "utils/structs/actor.py", "utils/structs/link.py",
                    "agents/robots/panda/panda.py", "agents/robots/panda/panda_wristcam.py",
                    "agents/controllers/base_controller.py", "agents/controllers/pd_joint_pos.py",
                    "assets/robots/panda/panda_v3.urdf"]
    api_hashes = {name: sha(API/name) for name in source_files}
    assert api_hashes["envs/tasks/tabletop/stack_cube.py"] == "ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440"
    # Two previously downloaded files are byte-equal to pinned commit6226567.
    pinned = ROOT/"research/discovery/upstream/state-api-6226567/mani_skill"
    pinned_equal = {name: sha(API/name) == sha(pinned/name) for name in
                    ("envs/sapien_env.py", "agents/controllers/base_controller.py", "agents/controllers/pd_joint_pos.py")}
    assert all(pinned_equal.values()) and importlib.metadata.version("mani_skill") == "3.0.0b21"
    actor = ET.parse(API/"assets/robots/panda/panda_v3.urdf").getroot()
    joint_names = ["panda_joint"+str(i) for i in range(1, 8)]
    limits = np.array([[float(next(j for j in actor.findall("joint") if j.attrib["name"] == name).find("limit").attrib[v])
                        for v in ("lower", "upper")] for name in joint_names])
    grasp = flags[..., 0]
    previous = np.concatenate((np.zeros((1, 64), dtype=bool), grasp[:-1]))
    starts, drops = grasp & ~previous, ~grasp & previous
    ever = grasp.any(0)
    first = np.where(ever, grasp.argmax(0), 50)
    A, B, tcp = nxt[..., 25:28], nxt[..., 32:35], nxt[..., 18:21]
    preA, preTCP = obs[..., 25:28], obs[..., 18:21]
    initialA, initialB = obs[0, :, 25:28], obs[0, :, 32:35]
    dz, dtcp = A-preA, tcp-preTCP
    height = A[..., 2]-initialA[:, 2]
    goal = B.copy(); goal[..., 2] += .04
    goal_distance = np.linalg.norm(A-goal, axis=-1)
    tcp_distance = np.linalg.norm(A-tcp, axis=-1)
    durations, episodes = [], []
    longest = np.zeros(64, dtype=int)
    for lane in range(64):
        padded = np.concatenate(([False], grasp[:, lane], [False])).astype(int)
        begins, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
        lengths = ends-begins
        durations.extend(lengths.tolist())
        longest[lane] = lengths.max() if len(lengths) else 0
        onset = int(first[lane])
        episodes.append(dict(lane=lane, ever_grasp=bool(ever[lane]), first_grasp_step=onset+1 if ever[lane] else None,
            grasp_steps=int(grasp[:, lane].sum()), longest_grasp_run=int(longest[lane]),
            segments=int(starts[:, lane].sum()), drops=int(drops[:, lane].sum()), final_grasp=bool(grasp[-1, lane]),
            max_height_gain_m=float(height[:, lane].max()), final_height_gain_m=float(height[-1, lane]),
            final_cube_xy_displacement_m=float(np.linalg.norm(A[-1, lane, :2]-initialA[lane, :2])),
            final_goal_distance_m=float(goal_distance[-1, lane]),
            cube_xy_displacement_after_first_grasp_m=float(np.linalg.norm(A[-1, lane, :2]-A[onset, lane, :2])) if ever[lane] else None,
            cube_z_change_after_first_grasp_m=float(A[-1, lane, 2]-A[onset, lane, 2]) if ever[lane] else None))
    masks = dict(before_first_grasp=np.arange(50)[:, None] < first,
                 grasp_onset=starts, grasp_continuation=grasp & previous, drop_transition=drops,
                 grasp_after_step=grasp, not_grasped_after_step=~grasp)
    mean, logstd = actor_outputs(state, obs.reshape(-1, 48))
    mean, logstd = mean.reshape(50, 64, 8), logstd.reshape(50, 64, 8)
    deterministic = np.tanh(mean)
    weights_path = checked("run/actor-step-256000/weights.json")
    manifest = json.loads(checked("run/actor-step-256000/manifest.json").read_text())
    assert sha(weights_path) == manifest["weights_sha256"]
    safe = json.loads(weights_path.read_text()); result = obs.reshape(-1, 48).copy()
    for i in range(4):
        w, b = (safe[f"layers.{i}.{v}"] for v in ("weight", "bias"))
        result = result@np.array(w["values"]).reshape(w["shape"]).T+np.array(b["values"])
        if i < 3: result = np.tanh(result)
    actor_parity = float(np.abs(result.reshape(50, 64, 8)-deterministic).max())
    assert actor_parity <= 1e-6
    joint_delta = nxt[..., :7]-obs[..., :7]
    requested_delta = .1*action[..., :7]
    phase = {}
    for name, mask in masks.items():
        moving = mask & (np.linalg.norm(dtcp, axis=-1) > 1e-5) & (np.linalg.norm(dz, axis=-1) > 1e-5)
        cosine = (dtcp*dz).sum(-1)/(np.maximum(np.linalg.norm(dtcp, axis=-1)*np.linalg.norm(dz, axis=-1), 1e-30))
        phase[name] = dict(frames=int(mask.sum()), reward=stats(data["reward"][mask]),
            cube_height_gain_m=stats(height[mask]), cube_z_delta_m_per_step=stats(dz[..., 2][mask]),
            cube_xy_delta_m_per_step=stats(np.linalg.norm(dz[..., :2], axis=-1)[mask]),
            tcp_z_delta_m_per_step=stats(dtcp[..., 2][mask]), tcp_xy_delta_m_per_step=stats(np.linalg.norm(dtcp[..., :2], axis=-1)[mask]),
            tcp_cube_distance_m=stats(tcp_distance[mask]), cube_goal_distance_m=stats(goal_distance[mask]),
            moving_tcp_cube_displacement_cosine=stats(cosine[moving]),
            tcp_cube_step_displacement_difference_m=stats(np.linalg.norm(dtcp-dz, axis=-1)[mask]),
            finger_qpos_m=[stats(nxt[..., j][mask]) for j in (7, 8)],
            actual_normalized_action=[stats(action[..., j][mask]) for j in range(8)],
            actual_behavior_logprob=stats(data["behavior_logprob"][..., 0][mask]),
            actual_action_abs_ge_095_fraction=[float((np.abs(action[..., j][mask]) >= .95).mean()) for j in range(8)],
            requested_arm_joint_delta_rad=[stats(requested_delta[..., j][mask]) for j in range(7)],
            observed_arm_joint_delta_rad=[stats(joint_delta[..., j][mask]) for j in range(7)],
            arm_target_tracking_error_rad=[stats((requested_delta-joint_delta)[..., j][mask]) for j in range(7)],
            arm_joint_within_001rad_limit_fraction=[float(((nxt[..., j][mask]-limits[j, 0] < .01) | (limits[j, 1]-nxt[..., j][mask] < .01)).mean()) for j in range(7)],
            gripper_requested_mimic_target_m=stats((.015+.025*action[..., 7])[mask]),
            counterfactual_checkpoint_actor=dict(
                label="Frozen256k actor on stored pre-step observations; NOT actual per-step behavior parameters",
                latent_mean=[stats(mean[..., j][mask]) for j in range(8)],
                latent_std=[stats(np.exp(logstd[..., j])[mask]) for j in range(8)],
                deterministic_action=[stats(deterministic[..., j][mask]) for j in range(8)]))
    old_path = ROOT/"runs/stackcube-sac-screen-seed1-20261027/development-diagnostic-summary.json"
    old = next(r for r in json.loads(old_path.read_text()) if r["steps"] == 256000)
    trusted = json.loads((ARCHIVE/"trusted-summary.json").read_text())
    receipt = dict(state="complete", analysis_source_sha256=sha(Path(__file__)), inputs=inputs,
        observation_layout_zero_based_half_open=SLICES, api_version="3.0.0b21", api_source_sha256=api_hashes,
        previously_pinned_commit="6226567ce326625fa6eacf9aeb3f4008e4339f7e", previously_pinned_api_byte_equal=pinned_equal,
        coordinate_checks=dict(relative_max_abs_errors=relative_errors, temporal_contiguity_max_abs=0.,
            all_truncations_only_last_step=True, terminations=0, flags_align_with="next_obs after actual env.step; final_observation at timeout"),
        scope=dict(stored_transitions_analyzed=3200, behavior_episodes=64, vector_steps=50,
            historical_training_transitions=256000, duplicate_final_trajectory_ignored=True, duplicate_final_state_ignored=True,
            new_simulator_transitions=0, training_optimizer_calls=0, new_evaluator_queries=0,
            counterfactual_actor_forward_rows=3200, safe_export_parity_forward_rows=3200,
            no_behavior_mean_std_reconstruction=True, no_Q_truth_or_historical_Q_reconstruction=True),
        trajectory_note=str(data["note"]), checkpoint_weights_sha256=sha(weights_path),
        actor_state_to_safe_export_max_abs=actor_parity, checkpoint_alpha=float(np.exp(state["log_alpha"])),
        episodes=episodes, phases=phase,
        summary=dict(ever_grasp_episodes=int(ever.sum()), final_grasp_episodes=int(grasp[-1].sum()),
            grasp_frames=int(grasp.sum()), first_grasp_step_among_ever=stats(first[ever]+1),
            grasp_steps_per_episode=stats(grasp.sum(0)), longest_grasp_run_among_ever=stats(longest[ever]),
            segment_lengths_steps=stats(durations), total_segments=int(starts.sum()), total_drops=int(drops.sum()),
            episodes_with_drops=int(drops.any(0).sum()), episodes_with_regrasp=int((starts.sum(0)>1).sum()),
            sustained_signal_ge10steps_no_3cm_lift=int(((longest>=10) & (height.max(0)<.03)).sum()),
            transient_signal_lt10steps_no_3cm_lift=int((ever & (longest<10) & (height.max(0)<.03)).sum()),
            no_grasp_episodes=int((~ever).sum()), inferred_3cm_lift_episodes=int((height.max(0)>=.03).sum()),
            max_height_gain_per_episode_m=stats(height.max(0)), final_cube_xy_displacement_m=stats(np.linalg.norm(A[-1, :, :2]-initialA[:, :2], axis=-1)),
            cube_B_xy_displacement_final_m=stats(np.linalg.norm(B[-1, :, :2]-initialB[:, :2], axis=-1)),
            placement_frames=int(flags[..., 1].sum()), static_frames=int(flags[..., 2].sum()), success_frames=int(flags[..., 3].sum())),
        cached_new_development=trusted["development_curve"], cached_primary_verdict=trusted["primary_verdict"],
        old_256k_aggregate=old, aggregate_source_sha256={str(old_path.relative_to(ROOT)): sha(old_path),
            str((ARCHIVE/"trusted-summary.json").relative_to(ROOT)): sha(ARCHIVE/"trusted-summary.json")},
        limitations=["64 recent stochastic training episodes, actor changes during them; not independent frozen-policy evaluations",
                    "Contact-force grasp signal is not a rigid grasp or proof of load-bearing; no raw forces retained",
                    "Joint commands are not task-space vertical commands; correlations and tracking residuals do not establish causes",
                    "Final checkpoint policy parameters cannot reconstruct historical behavior standard deviations",
                    "No old critic/logstd retained; no historical critic or target reconstruction"])
    receipt.update(cpu_wall_seconds=time.monotonic()-started, cpu_process_seconds=time.process_time()-cpu_started)
    with OUTPUT.open("x") as stream: json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(dict(receipt=str(OUTPUT.relative_to(ROOT)), summary=receipt["summary"], cpu_seconds=receipt["cpu_wall_seconds"])))


if __name__ == "__main__": run()
