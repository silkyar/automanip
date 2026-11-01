"""Frozen StackCube evaluation: tensor data in, independent state scores out."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from automanip.evaluator import evaluate_batch
from automanip.metrics import success_interval
from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import load_policy
from automanip.stackcube import STACK_TASK, DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS, check_frozen_protocol
from automanip.stackcube_adapter import StackCubeAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact",type=Path)
    p.add_argument("--split",choices=("development","confirmation"),default="development")
    p.add_argument("--num-envs",type=int,default=8)
    p.add_argument("--backend",choices=("physx_cpu","physx_cuda"),default="physx_cuda")
    p.add_argument("--smoke",action="store_true",help="eight public episodes, explicitly not a benchmark score")
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    digest = check_frozen_protocol()
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else CONFIRMATION_SEEDS
    if args.smoke: seeds = DEVELOPMENT_SEEDS[:8]
    if args.num_envs < 1 or len(seeds)%args.num_envs:
        p.error("episode count must divide into complete vector batches")
    env = StackCubeAdapter(args.num_envs,args.backend)
    results,diagnostics = [],[]
    try:
        artifact = load_policy(args.artifact,env.schema)
        policy = NumpyPolicy(artifact,env.schema)
        for start in range(0,len(seeds),args.num_envs):
            batch = evaluate_batch(env,policy,seeds[start:start+args.num_envs],horizon=STACK_TASK.horizon,action_dim=STACK_TASK.action_dim)
            results.extend(batch)
            diagnostics.extend(d.report(r.outcome) for d,r in zip(env.diagnostics,batch))
        successes = sum(r.outcome.success_at_end for r in results)
        report = dict(task_fingerprint=STACK_TASK.fingerprint,protocol_sha256=digest,
                      weights_sha256=artifact.weights_sha256,split=args.split,smoke=args.smoke,
                      episode_count=len(results),successes_at_end=successes,
                      success_at_end_rate=successes/len(results),
                      success_ever_rate=sum(r.outcome.ever_success and not r.outcome.failed for r in results)/len(results),
                      failed_episodes=sum(r.outcome.failed for r in results),
                      wilson_95_interval=success_interval(successes,len(results)),
                      diagnostics_rates={k:sum(d["ever"][k] and not r.outcome.failed for d,r in zip(diagnostics,results))/len(results) for k in diagnostics[0]["ever"]},
                      first_unmet_stages=dict(Counter(d["first_unmet_stage"] for d in diagnostics)),
                      episodes=[dict(**asdict(r),diagnostics=d) for r,d in zip(results,diagnostics)])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(json.dumps({k:v for k,v in report.items() if k!="episodes"}))
        if report["failed_episodes"]:
            raise SystemExit(2)
    finally: env.close()


if __name__ == "__main__": main()
