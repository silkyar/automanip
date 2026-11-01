"""Preflight the frozen spaces and independent stock scorer against ManiSkill."""
import argparse
import json
from automanip.stackcube import STACK_TASK, check_frozen_protocol
from automanip.stackcube_adapter import StackCubeAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=("physx_cpu","physx_cuda"),default="physx_cuda")
    p.add_argument("--num-envs",type=int,default=8)
    args=p.parse_args()
    digest=check_frozen_protocol()
    env=StackCubeAdapter(args.num_envs,args.backend)
    try:
        seeds=list(range(args.num_envs))
        first=env.reset(seeds)
        env.step([[0.0]*8]*args.num_envs)
        again=env.reset(seeds)
        if any(abs(x-y)>1e-6 for a,b in zip(first,again) for x,y in zip(a,b)):
            raise RuntimeError("seeded resets differ")
        for _ in range(STACK_TASK.horizon):
            _,success,_,_=env.step([[0.0]*8]*args.num_envs)
            # Agreement check only; evaluator scoring does not consume these flags.
            if success!=env.env.evaluate()["success"].detach().cpu().tolist():
                raise RuntimeError("independent stock rule differs from upstream")
        print(json.dumps(dict(task=STACK_TASK.env_id,robot=STACK_TASK.robot,
                              observation_dim=48,action_dim=8,num_envs=args.num_envs,
                              horizon=50,scorer_parity=True,reset_reproducible=True,
                              task_fingerprint=STACK_TASK.fingerprint,protocol_sha256=digest)))
    finally:env.close()


if __name__=="__main__":main()
