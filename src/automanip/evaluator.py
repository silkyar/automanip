"""Evaluator contracts only; simulator rollout implementation is milestone 3."""
from dataclasses import dataclass
import math
from typing import Protocol, Sequence


class Policy(Protocol):
    """Implemented only by trusted code constructing an allowed architecture."""
    def act(self, observations: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]: ...


def validate_actions(actions: Sequence[Sequence[float]], *, batch_size: int, action_dim: int = 7) -> None:
    """Reject invalid policy output rather than silently repairing the candidate."""
    if len(actions) != batch_size:
        raise ValueError("action batch mismatch")
    for action in actions:
        if len(action) != action_dim:
            raise ValueError("action dimension mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in action):
            raise ValueError("actions must be finite and within normalized bounds")


@dataclass(frozen=True)
class EpisodeOutcome:
    """Trusted rollout result. Failure overrides success even if previously seen.

    Horizon truncation ends an episode; it does not imply success. Report both
    terminal and ever-success explicitly to avoid confusing benchmark metrics.
    """
    terminal_success: bool
    ever_success: bool
    terminated: bool
    truncated: bool
    failed: bool = False

    @property
    def success_at_end(self) -> bool:
        return self.terminal_success and not self.failed and (self.terminated or self.truncated)
