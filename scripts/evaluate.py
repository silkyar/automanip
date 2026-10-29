"""Evaluate data-only artifacts; no candidate Python imports or seed logging."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from automanip.evaluator import evaluate_batch
from automanip.inference import MLPPolicy
from automanip.maniskill_adapter import ManiSkillAdapter
from automanip.policy_io import load_policy
from automanip.task_spec import TASK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    # Development seeds only. Private evaluation must use a separate service.
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.episodes % args.num_envs:
        parser.error("episodes must be positive and divisible by num-envs")
    env = ManiSkillAdapter(args.num_envs, args.backend)
    try:
        artifact = load_policy(args.artifact, env.schema)
        policy = MLPPolicy(artifact, env.schema)
        results = []
        for start in range(0, args.episodes, args.num_envs):
            results.extend(evaluate_batch(env, policy, list(range(start, start+args.num_envs)), horizon=TASK.horizon))
        print(json.dumps({"task_fingerprint": TASK.fingerprint,
                          "weights_sha256": artifact.weights_sha256,
                          "schema": asdict(env.schema),
                          "episodes": [asdict(r) for r in results],
                          "success_at_end_rate": sum(r.outcome.success_at_end for r in results)/len(results)}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
