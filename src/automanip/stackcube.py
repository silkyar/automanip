"""Frozen StackCube protocol. PickCube TASK and its results remain unchanged."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from automanip.task_spec import TaskSpec

STACK_TASK = TaskSpec(env_id="StackCube-v1", robot="panda_wristcam",
                      control_mode="pd_joint_delta_pos", observation_dim=48, action_dim=8)
DEVELOPMENT_SEEDS = tuple(range(128))
CONFIRMATION_SEEDS = tuple(range(1_000_000, 1_000_512))
STACK_TASK_SOURCE_SHA256 = "ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440"
OFFICIAL_PPO_SHA256 = "3fa4818861d428480244aae889bc679b2c8e89b59d6289014003c266a84dbab0"
UPSTREAM_COMMIT = "6226567ce326625fa6eacf9aeb3f4008e4339f7e"


def check_frozen_protocol():
    root=Path(__file__).resolve().parents[2]
    protocol=root/"benchmarks/stackcube-v1/protocol.json"
    lock=json.loads(protocol.read_text())
    for name,digest in lock["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("frozen evaluator changed: "+name)
    return hashlib.sha256(protocol.read_bytes()).hexdigest()


def stackcube_success(offset, linear_speed, angular_speed, grasped):
    """Scalar stock rule for unit checks; adapter uses matching float32 tensors."""
    if len(offset) != 3 or any(not math.isfinite(v) for v in offset):
        raise ValueError("invalid cube offset")
    if any(not math.isfinite(v) or v < 0 for v in (linear_speed, angular_speed)) or type(grasped) is not bool:
        raise ValueError("invalid velocities or grasp flag")
    on_top = math.hypot(*offset[:2]) <= math.hypot(0.02, 0.02) + 0.005 and abs(offset[2]-0.04) <= 0.005
    return on_top and linear_speed <= 0.01 and angular_speed <= 0.5 and not grasped


@dataclass(frozen=True)
class StackFrame:
    grasped: bool
    lifted: bool
    transported: bool
    on_top: bool
    static: bool
    contact_free: bool

    @property
    def success(self):
        return self.on_top and self.static and not self.grasped


class StackDiagnostics:
    """Observational stage counters, never additional success requirements."""
    def __init__(self):
        self.ever = {k: False for k in ("grasp", "lift", "transport", "placement", "release", "contact_free_placement", "stable_stack_10_steps")}
        self.stable_steps = self.max_stable_steps = 0
        self.last = None

    def update(self, frame):
        for key, value in (("grasp", frame.grasped), ("lift", frame.lifted),
                           ("transport", frame.transported), ("placement", frame.on_top),
                           ("release", frame.on_top and not frame.grasped),
                           ("contact_free_placement", frame.on_top and frame.contact_free)):
            self.ever[key] |= value
        self.stable_steps = self.stable_steps + 1 if frame.success else 0
        self.max_stable_steps = max(self.max_stable_steps, self.stable_steps)
        self.ever["stable_stack_10_steps"] |= self.stable_steps >= 10
        self.last = frame

    def report(self, outcome):
        if outcome.failed:
            failure = "execution_failure"
        elif outcome.success_at_end:
            failure = "success"
        else:
            failure = next((label for key, label in (
                ("grasp", "never_grasped"), ("lift", "grasped_not_lifted"),
                ("transport", "lifted_not_transported"), ("placement", "transported_not_placed"),
                ("release", "placed_still_grasped"), ("stable_stack_10_steps", "released_not_stable")
            ) if not self.ever[key]), "lost_stack_before_horizon")
        return dict(ever=dict(self.ever), max_stable_steps=self.max_stable_steps,
                    first_unmet_stage=failure)
