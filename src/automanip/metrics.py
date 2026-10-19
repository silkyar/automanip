"""Pure scoring rules; inputs must come from evaluator-owned simulator state."""
from dataclasses import dataclass
import math


def _nonnegative(*values):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("expected finite nonnegative measurements")


def pickcube_success(target_distance: float, arm_qvel: tuple[float, ...]) -> bool:
    """Panda stock criterion; excludes finger joints and does not require release."""
    _nonnegative(target_distance)
    if len(arm_qvel) != 7 or any(not math.isfinite(v) for v in arm_qvel):
        raise ValueError("expected seven finite arm joint velocities")
    return target_distance <= 0.025 and max(abs(v) for v in arm_qvel) <= 0.2


@dataclass(frozen=True)
class PlacementSample:
    """For a separate table-placement task, never the floating PickCube goal."""
    in_target_region: bool
    table_supported: bool
    gripper_contact: bool
    linear_speed: float
    angular_speed: float


class PlacementTracker:
    """Require continuous release/support/stability for an evaluator-owned interval."""
    def __init__(self, settle_steps: int = 10):
        if type(settle_steps) is not int or settle_steps < 1:
            raise ValueError("settle_steps must be positive")
        self.settle_steps = settle_steps
        self.consecutive = 0

    def update(self, sample: PlacementSample) -> bool:
        _nonnegative(sample.linear_speed, sample.angular_speed)
        flags = (sample.in_target_region, sample.table_supported, sample.gripper_contact)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("contact and region flags must be boolean")
        valid = (sample.in_target_region and sample.table_supported and not sample.gripper_contact
                 and sample.linear_speed <= 0.01 and sample.angular_speed <= 0.1)
        self.consecutive = self.consecutive + 1 if valid else 0
        return self.consecutive >= self.settle_steps


def success_interval(successes: int, episodes: int) -> tuple[float, float]:
    """95% Wilson interval; counts must include failed/crashed episodes."""
    if type(successes) is not int or type(episodes) is not int or not 0 <= successes <= episodes or episodes < 1:
        raise ValueError("invalid episode counts")
    p, z = successes / episodes, 1.959963984540054
    denominator = 1 + z*z/episodes
    center = (p + z*z/(2*episodes))/denominator
    radius = z * math.sqrt(p*(1-p)/episodes + z*z/(4*episodes*episodes))/denominator
    return max(0.0, center-radius), min(1.0, center+radius)
