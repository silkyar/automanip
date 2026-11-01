"""Public-URDF analytic kinematics for an assisted diagnostic, not a learner."""
import numpy as np
import xml.etree.ElementTree as ET
CONFIG={'base_link': 'panda_link0', 'tcp_link': 'panda_hand_tcp', 'base_translation_m': [-0.615, 0.0, 0.0], 'joint_names': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']}

def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw]); sr, sp, sy = np.sin([roll, pitch, yaw])
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return rz@ry@rx

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
