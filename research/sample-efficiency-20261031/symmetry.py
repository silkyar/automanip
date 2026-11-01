"""Exact base-yaw symmetry of state-based StackCube with pd_joint_delta_pos.

Rotating the whole scene by theta about the robot base's vertical axis is the same as adding theta to
Panda joint 1; joint-space actions and the dense reward are invariant. Observation layout (48):
qpos 0-8, qvel 9-17, tcp pose 18-24, cubeA pose 25-31, cubeB pose 32-38 (pos xyz + quat wxyz),
tcp->A 39-41, tcp->B 42-44, A->B 45-47.
"""
import torch

POSITIONS = (18, 25, 32)
QUATERNIONS = (21, 28, 35)
VECTORS = (39, 42, 45)


def rotate_xy(v, cos, sin):
    x, y = v[..., 0], v[..., 1]
    return torch.stack([cos*x - sin*y, sin*x + cos*y, v[..., 2]], -1)


def rotate_quaternion(q, half_cos, half_sin):
    # (cos(t/2), 0, 0, sin(t/2)) * (w, x, y, z), wxyz convention
    w, x, y, z = q.unbind(-1)
    return torch.stack([half_cos*w - half_sin*z, half_cos*x - half_sin*y,
                        half_cos*y + half_sin*x, half_cos*z + half_sin*w], -1)


def rotate_obs(obs, theta, center):
    """obs [..., 48], theta [...] radians, center [2] (robot base xy). Returns the rotated-scene observation."""
    out = obs.clone()
    cos, sin = torch.cos(theta), torch.sin(theta)
    half_cos, half_sin = torch.cos(theta/2), torch.sin(theta/2)
    out[..., 0] = obs[..., 0] + theta
    offset = torch.zeros(3, dtype=obs.dtype, device=obs.device); offset[:2] = center
    for i in POSITIONS:
        out[..., i:i+3] = rotate_xy(obs[..., i:i+3] - offset, cos, sin) + offset
    for i in QUATERNIONS:
        out[..., i:i+4] = rotate_quaternion(obs[..., i:i+4], half_cos, half_sin)
    for i in VECTORS:
        out[..., i:i+3] = rotate_xy(obs[..., i:i+3], cos, sin)
    return out
