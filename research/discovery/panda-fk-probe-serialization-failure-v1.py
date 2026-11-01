"""Public-URDF kinematic verification on archived qpos; no simulator/fitting.

All origins, axes and fixed TCP transforms are parsed from the URDF. The task's
documented base pose is a disclosed structural prior, not a learned model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from research.discovery.temperature_grasp_diagnostic import numeric_npz

ROOT = Path(__file__).resolve().parents[2]
API = ROOT/".venv311/lib/python3.11/site-packages/mani_skill"
URDF = API/"assets/robots/panda/panda_v3.urdf"
TRAJECTORY = ROOT/"runs/temperature-sufficiency-20261028/run/actor-step-256000-trajectory.npz"
OUTPUT = ROOT/"research/discovery/panda-fk-probe-20261028.json"
CONFIG = dict(base_link="panda_link0", tcp_link="panda_hand_tcp",
    base_translation_m=[-.615, 0., 0.], base_quaternion_wxyz=[1., 0., 0., 0.],
    joint_names=["panda_joint"+str(i) for i in range(1, 8)],
    position_max_tolerance_m=1e-4, orientation_max_tolerance_rad=1e-3,
    jacobian_max_abs_tolerance=1e-4, finite_difference_delta_rad=1e-6,
    finite_difference_flat_pose_indices=[0, 63, 512, 1023, 1536, 2047, 2560, 3199],
    finite_difference_directions="all seven canonical joint directions at each of eight poses",
    dls=dict(desired_translation_m=[0., 0., .005], desired_rotation_vector_rad=[0., 0., 0.],
             damping=.05, pose_row_weights=[1., 1., 1., 1., 1., 1.],
             unit_caveat="Translation rows are m/rad, rotation rows rad/rad; conditioning depends on this explicit unscaled weighting",
             iterations=1, joint_clipping=False),
    no_tolerance_tuning=True, no_simulator=True, no_fitting=True)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return dict(n=int(x.size), minimum=float(x.min()), median=float(np.median(x)),
                mean=float(x.mean()), q90=float(np.quantile(x, .9)), maximum=float(x.max()))


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw]); sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return rz@ry@rx  # URDF fixed-axis roll/pitch/yaw convention.


def parse_chain(path):
    root = ET.parse(path).getroot()
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        if child in by_child: raise ValueError("Ambiguous URDF parent")
        by_child[child] = joint
    chain, link = [], CONFIG["tcp_link"]
    while link != CONFIG["base_link"]:
        if link not in by_child: raise ValueError("TCP/base identity absent")
        joint = by_child[link]; origin = joint.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()] if origin is not None else [0., 0., 0.]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()] if origin is not None else [0., 0., 0.]
        axis_node = joint.find("axis")
        axis = [float(v) for v in axis_node.attrib.get("xyz", "1 0 0").split()] if axis_node is not None else [1., 0., 0.]
        entry = dict(name=joint.attrib["name"], type=joint.attrib["type"],
                     parent=joint.find("parent").attrib["link"], child=link,
                     origin_xyz=xyz, origin_rpy=rpy, axis=axis)
        if entry["type"] not in ("revolute", "fixed"): raise ValueError("Unexpected chain joint type")
        if entry["type"] == "revolute":
            if joint.find("mimic") is not None: raise ValueError("Unexpected mimic in arm chain")
            limit = joint.find("limit")
            entry["limits"] = [float(limit.attrib["lower"]), float(limit.attrib["upper"])]
            if abs(np.linalg.norm(axis)-1) > 1e-10: raise ValueError("Non-unit joint axis")
        chain.append(entry); link = entry["parent"]
        if len(chain)>30: raise ValueError("Cyclic/oversized chain")
    chain.reverse()
    if [j["name"] for j in chain if j["type"] == "revolute"] != CONFIG["joint_names"]:
        raise ValueError("Observation joint identity/order ambiguous")
    return chain


class Kinematics:
    def __init__(self, chain):
        self.chain = chain
        self.fk_pose_evaluations = 0
        self.geometric_jacobian_pose_evaluations = 0

    def fk(self, q, jacobian=False):
        q = np.asarray(q, dtype=np.float64).reshape(-1, 7)
        n = len(q); self.fk_pose_evaluations += n
        t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        t[:, :3, 3] = CONFIG["base_translation_m"]
        axes, positions, index = [], [], 0
        for joint in self.chain:
            origin = np.eye(4); origin[:3, :3] = rpy_matrix(joint["origin_rpy"])
            origin[:3, 3] = joint["origin_xyz"]
            t = t@origin
            if joint["type"] == "fixed": continue
            axis = np.asarray(joint["axis"])
            axes.append(t[:, :3, :3]@axis); positions.append(t[:, :3, 3].copy())
            x, y, z = axis
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            angle = q[:, index]
            rot = np.eye(3)+np.sin(angle)[:, None, None]*skew+(1-np.cos(angle))[:, None, None]*(skew@skew)
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy(); motion[:, :3, :3] = rot
            t = t@motion; index += 1
        if not jacobian: return t
        axes, positions = np.stack(axes, axis=-1), np.stack(positions, axis=-1)
        translation = np.cross(axes.transpose(0, 2, 1),
                               (t[:, :3, 3, None]-positions).transpose(0, 2, 1)).transpose(0, 2, 1)
        self.geometric_jacobian_pose_evaluations += n
        return t, np.concatenate((translation, axes), axis=1)


def quaternion_matrix(wxyz):
    q = wxyz/np.linalg.norm(wxyz, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(-1, 3, 3)


def angle_between(left, right):
    r = left@right.transpose(0, 2, 1)
    sine = np.linalg.norm(np.stack((r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0],
                                    r[:, 1, 0]-r[:, 0, 1]), axis=-1), axis=-1)/2
    cosine = (np.trace(r, axis1=1, axis2=2)-1)/2
    return np.arctan2(sine, cosine)


def run():
    if OUTPUT.exists(): raise FileExistsError("One preregistered gate; preserve receipt")
    started, cpu_started = time.monotonic(), time.process_time()
    prior = json.loads((ROOT/"research/discovery/temperature-grasp-diagnostic-20261028.json").read_text())
    assert sha(TRAJECTORY) == prior["inputs"]["run/actor-step-256000-trajectory.npz"]["sha256"]
    assert sha(URDF) == prior["api_source_sha256"]["assets/robots/panda/panda_v3.urdf"]
    files = [Path(__file__), URDF, TRAJECTORY,
        API/"envs/tasks/tabletop/stack_cube.py", API/"utils/scene_builder/table/scene_builder.py",
        API/"agents/robots/panda/panda.py", API/"agents/robots/panda/panda_wristcam.py",
        API/"utils/structs/pose.py", ROOT/"research/discovery/temperature_grasp_diagnostic.py"]
    receipt = dict(state="started", config=CONFIG, source_and_input_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
                   structural_prior="Public pinned Panda URDF plus task-defined base transform; assisted diagnostic only",
                   simulator_queries=0, simulator_state_writes=0, training_optimizer_calls=0, evaluator_queries=0)
    with OUTPUT.open("x") as f: json.dump(receipt, f, indent=2); f.write("\n")
    engine = None
    try:
        chain = parse_chain(URDF); engine = Kinematics(chain); receipt["parsed_chain"] = chain
        data = numeric_npz(TRAJECTORY)
        obs, nxt = (data[name].astype(np.float64).reshape(-1, 48) for name in ("obs", "next_obs"))
        assert obs.shape == nxt.shape == (3200, 48)
        transforms = {}
        for name, values in (("pre_step", obs), ("post_step", nxt)):
            t, j = engine.fk(values[:, :7], jacobian=True); transforms[name] = (t, j)
            quat = values[:, 21:25]; norm_error = np.max(np.abs(np.linalg.norm(quat, axis=-1)-1))
            if norm_error > 1e-4: raise ValueError("TCP quaternion identity/normalization differs")
            position_error = np.linalg.norm(t[:, :3, 3]-values[:, 18:21], axis=-1)
            orientation_error = angle_between(t[:, :3, :3], quaternion_matrix(quat))
            receipt[name] = dict(poses=3200, position_error_m=statistics(position_error),
                                 orientation_error_rad=statistics(orientation_error), quaternion_norm_max_error=float(norm_error))
        ids = CONFIG["finite_difference_flat_pose_indices"]
        selected = obs[ids, :7]; eps = CONFIG["finite_difference_delta_rad"]
        perturb = np.eye(7)*eps
        plus = engine.fk((selected[:, None, :]+perturb).reshape(-1, 7)).reshape(8, 7, 4, 4)
        minus = engine.fk((selected[:, None, :]-perturb).reshape(-1, 7)).reshape(8, 7, 4, 4)
        rotation_derivative = (plus[:, :, :3, :3]-minus[:, :, :3, :3])/(2*eps)
        omega = rotation_derivative@transforms["pre_step"][0][ids, None, :3, :3].transpose(0, 1, 3, 2)
        angular = np.stack((omega[:, :, 2, 1]-omega[:, :, 1, 2], omega[:, :, 0, 2]-omega[:, :, 2, 0],
                            omega[:, :, 1, 0]-omega[:, :, 0, 1]), axis=-1)/2
        linear = (plus[:, :, :3, 3]-minus[:, :, :3, 3])/(2*eps)
        fd = np.concatenate((linear, angular), axis=-1).transpose(0, 2, 1)
        errors = np.abs(fd-transforms["pre_step"][1][ids])
        receipt["finite_difference"] = dict(saved_flat_pose_indices=ids, canonical_directions_per_pose=7,
            analytic_fk_evaluations=112, max_abs_error=float(errors.max()),
            per_pose_max_abs_error=errors.max(axis=(1, 2)).tolist(),
            linear_max_abs_error=float(errors[:, :3].max()), angular_max_abs_error=float(errors[:, 3:].max()))
        receipt["verification_passed"] = (all(receipt[name]["position_error_m"]["maximum"] <= 1e-4
            and receipt[name]["orientation_error_rad"]["maximum"] <= 1e-3 for name in ("pre_step", "post_step"))
            and errors.max() <= 1e-4)
        if not receipt["verification_passed"]:
            receipt["state"] = "verification_failed_stop_no_controller"; return
        mask = data["stage_flags"][..., 0].reshape(-1)
        q = nxt[mask, :7]; j = transforms["post_step"][1][mask]; t = transforms["post_step"][0][mask]
        wanted = np.array([0., 0., .005, 0., 0., 0.])
        dq = (j.transpose(0, 2, 1)@np.linalg.solve(j@j.transpose(0, 2, 1)+.05**2*np.eye(6),
                np.broadcast_to(wanted, (len(q), 6))[..., None]))[..., 0]
        achieved_linear = (j@dq[..., None])[..., 0]
        singular = np.linalg.svd(j, compute_uv=False)
        positional_singular = np.linalg.svd(j[:, :3], compute_uv=False)
        advanced = engine.fk(q+dq)
        step = advanced[:, :3, 3]-t[:, :3, 3]
        rotation_step = angle_between(advanced[:, :3, :3], t[:, :3, :3])
        limits = np.array([v["limits"] for v in chain if v["type"] == "revolute"])
        receipt["counterfactual_dls"] = dict(grasped_post_step_poses=int(mask.sum()),
            full_pose_singular_minimum=statistics(singular[:, -1]), full_pose_condition_number=statistics(singular[:, 0]/singular[:, -1]),
            translation_jacobian_condition_number=statistics(positional_singular[:, 0]/positional_singular[:, -1]),
            requested_z_step_m=.005, predicted_linear_z_step_m=statistics(achieved_linear[:, 2]),
            predicted_linear_lateral_step_m=statistics(np.linalg.norm(achieved_linear[:, :2], axis=-1)),
            predicted_linear_angular_step_rad=statistics(np.linalg.norm(achieved_linear[:, 3:], axis=-1)),
            joint_delta_max_abs_rad_per_pose=statistics(np.abs(dq).max(axis=-1)),
            normalized_pd_joint_delta_max_abs=statistics(np.abs(dq).max(axis=-1)/.1),
            candidate_joint_limit_violations=int(((q+dq<limits[:, 0]) | (q+dq>limits[:, 1])).any(axis=-1).sum()),
            candidate_normalized_action_bound_violations=int((np.abs(dq)>.1).any(axis=-1).sum()),
            nonlinear_fk_z_step_m=statistics(step[:, 2]), nonlinear_fk_lateral_step_m=statistics(np.linalg.norm(step[:, :2], axis=-1)),
            nonlinear_fk_orientation_change_rad=statistics(rotation_step),
            interpretation="Counterfactual unconstrained kinematics only; no collision/contact/force/dynamics/tracking validation")
        receipt["state"] = "complete"
    except Exception as exc:
        receipt.update(state="failed_preserve_no_tuning", error=type(exc).__name__+": "+str(exc)); raise
    finally:
        receipt.update(cpu_wall_seconds=time.monotonic()-started, cpu_process_seconds=time.process_time()-cpu_started,
            analytic_fk_pose_evaluations=engine.fk_pose_evaluations if engine else 0,
            geometric_jacobian_pose_evaluations=engine.geometric_jacobian_pose_evaluations if engine else 0)
        temporary = OUTPUT.with_suffix(".tmp"); temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n"); temporary.replace(OUTPUT)
        print(json.dumps(dict(state=receipt["state"], verification=receipt.get("verification_passed"),
                             seconds=receipt["cpu_wall_seconds"], analytic_fk=receipt["analytic_fk_pose_evaluations"])))


if __name__ == "__main__": run()
