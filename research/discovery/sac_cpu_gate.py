"""Reproducible CPU-only actor-learning and robot-schema export gate.

Contextual quadratic toy, not a robot task. Rewards are ordinary sampled task
feedback. Analytic optimal actions are used ONLY for the held-out toy diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from research.discovery.sac_baseline import Actor,Replay,SAC,export_actor,numerical_action


def run(output):
    start=time.monotonic();cpu=time.process_time()
    torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(6):
        torch.manual_seed(seed)
        learner=SAC(obs_dim=3,action_dim=2,width=64)
        replay=Replay(4096,3,2)
        obs=2*torch.rand(4096,3)-1
        action=2*torch.rand(4096,2)-1
        optimal=.6*torch.tanh(2*obs[:,:2])
        reward=-(action-optimal).square().mean(-1)
        replay.add(obs,torch.zeros_like(obs),action,reward,torch.ones(4096))
        probe=2*torch.rand(512,3)-1
        target=.6*torch.tanh(2*probe[:,:2])
        with torch.no_grad(): before=float((learner.actor.deterministic(probe)-target).square().mean())
        curve=[]
        for step in range(1200):
            metrics=learner.update(replay.sample(256))
            if (step+1)%200==0:
                with torch.no_grad(): error=float((learner.actor.deterministic(probe)-target).square().mean())
                curve.append(dict(gradient_steps=step+1,action_mse=error))
        after=curve[-1]["action_mse"]
        rows.append(dict(seed=seed,initial_action_mse=before,final_action_mse=after,
                         fraction_remaining=after/before,curve=curve,metrics=metrics))
        print(f"CPU toy seed {seed}: {before:.6f} -> {after:.6f}",flush=True)
    # A separate random robot-schema fixture, explicitly NOT a trained robot actor.
    torch.manual_seed(20261027)
    actor=Actor(width=256)
    exported=export_actor(actor,output/"robot-schema-fixture")
    states=torch.cat((torch.randn(128,48),torch.full((1,48),1000.),torch.full((1,48),-1000.)))
    raw=(output/"robot-schema-fixture"/"weights.json").read_bytes()
    with torch.no_grad():
        expected=actor.deterministic(states)
        actual=numerical_action(states,json.loads(raw))
    fixture=dict(meaning="random policy, serialization fixture only; no robot training",
                 state=states.tolist(),expected_action=expected.tolist(),
                 max_absolute_action_error=float((expected-actual).abs().max()),**exported)
    (output/"fixture-parity.json").write_text(json.dumps(fixture,allow_nan=False))
    report=dict(schema_version=1,scope="CPU contextual toy and independent random robot-schema export",
                torch=torch.__version__,device="cpu",source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                learner_source_sha256=hashlib.sha256(Path(__file__).with_name("sac_baseline.py").read_bytes()).hexdigest(),
                rows=rows,all_seeds_improved=all(r["final_action_mse"]<r["initial_action_mse"] for r in rows),
                export_fixture_max_error=fixture["max_absolute_action_error"],
                accounting=dict(toy_transitions=6*4096,toy_contextual_episodes=6*4096,
                    optimizer_rounds=6*1200,critic_optimizer_steps=6*1200,actor_optimizer_steps=6*1200,
                    alpha_optimizer_steps=6*1200,minibatch_presentations=6*1200*256,
                    robot_transitions=0,simulator_queries=0,teacher_demo_planning_data=0,
                    gpu_seconds=0,dollars_spent=0,wall_seconds=time.monotonic()-start,
                    process_cpu_seconds=time.process_time()-cpu))
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    if not report["all_seeds_improved"] or report["export_fixture_max_error"]>1e-6:
        raise RuntimeError("CPU actor/export gate failed; preserve report")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2))
