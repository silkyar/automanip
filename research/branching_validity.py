"""Training-only replay diagnostics. No claim to restore PhysX solver internals.

The augmented snapshot includes exposed controller buffers, drive targets,
generalized forces, elapsed steps, and random generators in addition to the
official physical state. It is an in-memory object, never a pickle artifact.
"""
import copy

import numpy as np


TOLERANCES = dict(position=1e-4, rotation=1e-3, qpos=1e-4, qvel=1e-2,
                  linear_velocity=1e-2, angular_velocity=5e-2)
BUFFERS = ("_step", "_start_qpos", "_target_qpos", "_step_size")
RNG_NAMES = ("_episode_rng", "_batched_episode_rng", "_main_rng", "_batched_main_rng")


def measure_frame(adapter):
    import torch
    e = adapter.env
    cpu = lambda t: t.detach().cpu().numpy().copy()
    return dict(position=cpu(torch.stack([e.cubeA.pose.p, e.cubeB.pose.p, e.agent.tcp.pose.p], 1)),
                quaternion=cpu(torch.stack([e.cubeA.pose.q, e.cubeB.pose.q, e.agent.tcp.pose.q], 1)),
                qpos=cpu(e.agent.robot.get_qpos()), qvel=cpu(e.agent.robot.get_qvel()),
                linear_velocity=cpu(torch.stack([e.cubeA.linear_velocity, e.cubeB.linear_velocity], 1)),
                angular_velocity=cpu(torch.stack([e.cubeA.angular_velocity, e.cubeB.angular_velocity], 1)))


def clone_tree(value):
    """Copy tensors immediately: upstream getters can return mutable views."""
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().clone()
    return copy.deepcopy(value)


def capture_rngs(env):
    # ManiSkill's BatchedRNG intercepts deepcopy/get_state and attempts to stack
    # heterogeneous RandomState tuples. Read the underlying generators instead.
    result = {}
    for name in RNG_NAMES:
        if not hasattr(env, name):
            continue
        rng = getattr(env, name)
        if rng is None:
            continue
        members = getattr(rng, 'rngs', None)
        result[name] = dict(batched=members is not None,
                            states=[copy.deepcopy(r.get_state()) for r in
                                    (members if members is not None else [rng])])
    return result


def restore_rngs(env, states):
    # Restore in place to preserve main/batched/episode alias relationships.
    for name, saved in states.items():
        rng = getattr(env, name)
        members = rng.rngs if saved['batched'] else [rng]
        if len(members) != len(saved['states']):
            raise ValueError('RNG batch size changed')
        for member, state in zip(members, saved['states']):
            member.set_state(copy.deepcopy(state))


def capture(env):
    import torch
    robot = env.agent.robot
    joint_indices = torch.arange(len(robot.active_joints), device=env.device,
                                 dtype=robot._data_index.dtype)
    indices = robot.get_joint_target_indices(joint_indices)
    px = env.scene.px
    return dict(
        physical=clone_tree(env.get_state_dict()), elapsed=env._elapsed_steps.clone(),
        controllers={name: {key: clone_tree(getattr(c, key)) for key in BUFFERS
                            if hasattr(c, key)}
                     for name, c in env.agent.controller.controllers.items()},
        drives=px.cuda_articulation_target_qpos.torch()[indices].clone(),
        drive_velocities=px.cuda_articulation_target_qvel.torch()[indices].clone(),
        qf=robot.get_qf().clone(),
        rng=capture_rngs(env),
        numpy_rng=copy.deepcopy(np.random.get_state()),
        torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()],
    )


def restore(env, snapshot, augmented):
    import torch
    # Use a new copy: a setter or controller must not mutate the saved anchor.
    env.set_state_dict(clone_tree(snapshot["physical"]))
    env._elapsed_steps.copy_(snapshot["elapsed"])
    if not augmented:
        return
    for name, fields in snapshot["controllers"].items():
        controller = env.agent.controller.controllers[name]
        for key, value in fields.items():
            setattr(controller, key, clone_tree(value))
    robot = env.agent.robot
    joint_indices = torch.arange(len(robot.active_joints), device=env.device,
                                 dtype=robot._data_index.dtype)
    robot.set_joint_drive_targets(snapshot["drives"].clone(),
                                 robot.active_joints, joint_indices)
    robot.set_joint_drive_velocity_targets(snapshot["drive_velocities"].clone(),
                                          robot.active_joints, joint_indices)
    robot.set_qf(snapshot["qf"].clone())
    env.scene._gpu_apply_all()
    env.scene.px.gpu_update_articulation_kinematics()
    env.scene._gpu_fetch_all()
    restore_rngs(env, snapshot['rng'])
    np.random.set_state(copy.deepcopy(snapshot["numpy_rng"]))
    torch.set_rng_state(snapshot["torch_rng"])
    torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def quaternion_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    # Sign invariant, numerically stable near identical orientations.
    distance = np.minimum(np.linalg.norm(a-b, axis=-1), np.linalg.norm(a+b, axis=-1))
    return 4*np.arcsin(np.clip(distance/2, 0, 1))


def compare(reference, replay):
    """All assigned lanes count; reference includes the restored initial state."""
    if set(reference) != set(replay):
        raise ValueError("Trace fields differ")
    for key in reference:
        if np.shape(reference[key]) != np.shape(replay[key]):
            raise ValueError("Trace shape differs: "+key)
        if not np.isfinite(reference[key]).all() or not np.isfinite(replay[key]).all():
            raise ValueError("Nonfinite trace")
    errors = {key: float(np.abs(reference[key]-replay[key]).max())
              for key in TOLERANCES if key != "rotation"}
    errors["rotation"] = float(quaternion_error(reference["quaternion"], replay["quaternion"]).max())
    # Flags describe post-action states only. Index 0=grasp; index 6=success.
    changed = np.any(reference["flags"][:, :, [0, 6]] != replay["flags"][:, :, [0, 6]], axis=(0, 2))
    final_changed = reference["flags"][-1, :, 6] != replay["flags"][-1, :, 6]
    passed = all(errors[k] <= limit for k, limit in TOLERANCES.items()) and not changed.any()
    return dict(maximum_errors=errors, grasp_or_success_changed_lanes=int(changed.sum()),
                final_success_changed_lanes=int(final_changed.sum()), passed=bool(passed))
