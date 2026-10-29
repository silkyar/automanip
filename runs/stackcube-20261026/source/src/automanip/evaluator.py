"""Trusted fixed-horizon rollout and policy contracts."""
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


@dataclass(frozen=True)
class RolloutResult:
    outcome: EpisodeOutcome
    steps: int
    error: str = ""


def evaluate_batch(environment, policy: Policy, seeds: Sequence[int], *, horizon: int = 50, action_dim: int = 7):
    """Synchronous fixed-horizon evaluation of an adapter with no autoreset.

    Success terminations are recorded but do not reset/shorten a rollout, matching
    reference fixed-horizon evaluation. Early truncations are adapter errors.
    Terminated records whether termination was signalled at any point.
    A shared simulator/policy error conservatively fails the complete batch.
    Hard wall-time limits require an external process supervisor.
    """
    if type(horizon) is not int or horizon < 1 or not seeds:
        raise ValueError("positive horizon and nonempty seeds required")
    if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("invalid evaluation seed")
    batch = len(seeds)
    ever = [False]*batch
    terminal = [False]*batch
    terminated = [False]*batch
    steps = 0
    try:
        observations = environment.reset(seeds)
        for step in range(horizon):
            actions = policy.act(observations)
            validate_actions(actions, batch_size=batch, action_dim=action_dim)
            observations, successes, terminations, truncations = environment.step(actions)
            if any(len(v) != batch for v in (observations, successes, terminations, truncations)):
                raise ValueError("simulator batch mismatch")
            if any(type(v) is not bool for vector in (successes, terminations, truncations) for v in vector):
                raise ValueError("invalid simulator flags")
            steps = step + 1
            if any(truncations) and steps < horizon:
                raise ValueError("unexpected early truncation")
            terminal = successes
            ever = [a or b for a, b in zip(ever, successes)]
            terminated = [a or b for a, b in zip(terminated, terminations)]
        return [RolloutResult(EpisodeOutcome(terminal[i], ever[i], terminated[i], True), steps)
                for i in range(batch)]
    except Exception as exc:
        # Do not put seeds, environment state, or exception messages in reports.
        return [RolloutResult(EpisodeOutcome(False, ever[i], terminated[i], False, True),
                              steps, type(exc).__name__) for i in range(batch)]
