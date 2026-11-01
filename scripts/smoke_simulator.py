"""Opt-in simulator/reset check. Run with PYTHONPATH=src python scripts/smoke_simulator.py."""
import json
from automanip.maniskill_adapter import ManiSkillAdapter
from automanip.task_spec import TASK
from automanip.evaluator import evaluate_batch


def main():
    env = ManiSkillAdapter()
    try:
        first = env.reset([123])
        env.step([[0.0]*7])
        repeated = env.reset([123])
        if len(first) != 1 or len(first[0]) != env.schema.observation_dim:
            raise RuntimeError("observation dimension mismatch")
        if any(abs(a-b) > 1e-6 for a, b in zip(first[0], repeated[0])):
            raise RuntimeError("seeded reset differs")
        class ZeroPolicy:
            def act(self, observations):
                return [[0.0]*7 for _ in observations]
        result, = evaluate_batch(env, ZeroPolicy(), [123], horizon=TASK.horizon)
        if result.outcome.failed or result.steps != TASK.horizon:
            raise RuntimeError("full-horizon smoke rollout failed")
        print(json.dumps({"rollout_steps": result.steps, "reset_reproducible": True, "observation_dim": env.schema.observation_dim,
                          "action_dim": 7, "task_fingerprint": TASK.fingerprint}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
