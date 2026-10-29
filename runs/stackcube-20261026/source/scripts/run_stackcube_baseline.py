"""Run the pinned, unchanged official StackCube PPO example and evaluate safely."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from automanip.stackcube import STACK_TASK, OFFICIAL_PPO_SHA256, UPSTREAM_COMMIT, check_frozen_protocol

ROOT=Path(__file__).resolve().parents[1]


def command(seed,name,smoke=False):
    return [sys.executable,str(ROOT/"third_party/maniskill/ppo.py"),
            "--env-id","StackCube-v1","--seed",str(seed),
            "--num-envs",str(8 if smoke else 1024),"--update-epochs","8",
            "--num-minibatches",str(2 if smoke else 32),
            "--total-timesteps",str(400 if smoke else 25_000_000),
            "--num-eval-envs","8","--no-capture-video","--no-track","--exp-name",name]


def run_child(command,cwd,log,timeout,heartbeat):
    env=os.environ.copy()
    env.pop("RUNPOD_API_KEY",None)
    env["PYTHONPATH"]=str(ROOT/"src")
    started=time.monotonic()
    with log.open("w") as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            while child.poll() is None:
                heartbeat.write_text(json.dumps(dict(time=time.time(),pid=child.pid,log=str(log)))+"\n")
                if time.monotonic()-started > timeout:
                    raise TimeoutError("process deadline: "+log.name)
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"process failed ({child.returncode}): {log.name}")
        finally:
            if child.poll() is None:
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
    return time.monotonic()-started


def events(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc=EventAccumulator(str(directory),size_guidance={"scalars":0})
    acc.Reload()
    selected={k:[dict(step=e.step,wall_time=e.wall_time,value=e.value if math.isfinite(e.value) else None) for e in acc.Scalars(k)]
              for k in acc.Tags()["scalars"] if k.startswith(("eval/","time/","charts/","losses/"))}
    (directory/"learning-curves.json").write_text(json.dumps(selected,indent=2,allow_nan=False)+"\n")
    with (directory/"learning-curves.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(("tag","environment_steps","wall_time","value"))
        for tag,values in selected.items():
            for e in values:writer.writerow((tag,e["step"],e["wall_time"],e["value"]))
    return selected


def evaluate_checkpoint(checkpoint,directory,label,out,heartbeat,smoke=False):
    policy=directory/(label+"-policy")
    run_child([sys.executable,str(ROOT/"scripts/export_official_policy.py"),str(checkpoint),str(policy)],
              out,directory/(label+"-export.log"),180,heartbeat)
    splits=("development",) if smoke or label!="final" else ("development","confirmation")
    reports={}
    for split in splits:
        report=directory/(label+"-"+split+".json")
        cmd=[sys.executable,str(ROOT/"scripts/evaluate_stackcube.py"),str(policy),"--split",split,
             "--output",str(report)]
        if smoke:cmd.append("--smoke")
        elapsed=run_child(cmd,out,directory/(label+"-"+split+".log"),1200,heartbeat)
        reports[split]=dict(json.loads(report.read_text()),evaluation_seconds=elapsed)
    return reports


def execute(out,seeds,smoke):
    if hashlib.sha256((ROOT/"third_party/maniskill/ppo.py").read_bytes()).hexdigest()!=OFFICIAL_PPO_SHA256:
        raise RuntimeError("official PPO source was modified")
    protocol_digest=check_frozen_protocol()
    if (out/"batch.exit").exists():raise RuntimeError("batch already completed; use a new output root")
    out.mkdir(parents=True,exist_ok=True)
    heartbeat=out/"heartbeat.json"
    overall=dict(upstream_commit=UPSTREAM_COMMIT,official_ppo_sha256=OFFICIAL_PPO_SHA256,
                 protocol_sha256=protocol_digest,task_fingerprint=STACK_TASK.fingerprint,
                 mode="smoke" if smoke else "baseline",requested_steps_per_seed=400 if smoke else 25_000_000,
                 expected_steps_per_seed=400 if smoke else 24_985_600,
                 versions={n:version(n) for n in ("mani_skill","torch","sapien","numpy","gymnasium","tensorboard")},runs=[])
    status=1
    try:
        for seed in seeds:
            name=f"stackcube-{'smoke-' if smoke else ''}seed-{seed}"
            directory=out/"runs"/name
            if directory.exists():raise RuntimeError("run directory already exists")
            directory.mkdir(parents=True)
            cmd=command(seed,name,smoke)
            record=dict(seed=seed,command=cmd,status="training",started_at=time.time(),
                        requested_steps=overall["requested_steps_per_seed"],upstream_commit=UPSTREAM_COMMIT,
                        official_ppo_sha256=OFFICIAL_PPO_SHA256,protocol_sha256=protocol_digest)
            (directory/"run.json").write_text(json.dumps(record,indent=2)+"\n")
            print("TRAINING",seed,flush=True)
            elapsed=run_child(cmd,out,directory/"train.log",900 if smoke else 7200,heartbeat)
            curves=events(directory)
            measured_steps=int(curves["time/step"][-1]["value"])
            if measured_steps!=overall["expected_steps_per_seed"]:
                raise RuntimeError("official trainer did not complete expected transitions")
            record.update(status="evaluating",environment_transitions=measured_steps,
                          trainer_process_seconds=elapsed,
                          rollout_seconds=sum(e["value"] for e in curves["time/rollout_time"]),
                          optimization_seconds=sum(e["value"] for e in curves["time/update_time"]))
            (directory/"run.json").write_text(json.dumps(record,indent=2)+"\n")
            curve=[]
            if not smoke:
                for iteration in (1,101,201,301,401):
                    checkpoint=directory/f"ckpt_{iteration}.pt"
                    print("CURVE_EVALUATION",seed,iteration,flush=True)
                    reports=evaluate_checkpoint(checkpoint,directory,f"step-{(iteration-1)*51200}",out,heartbeat)
                    curve.append(dict(environment_steps=(iteration-1)*51200,
                                      report=reports["development"]))
            print("FINAL_EVALUATION",seed,flush=True)
            final=evaluate_checkpoint(directory/"final_ckpt.pt",directory,"final",out,heartbeat,smoke)
            curve.append(dict(environment_steps=measured_steps,report=final["development"]))
            (directory/"independent-learning-curve.json").write_text(json.dumps(curve,indent=2)+"\n")
            record.update(status="complete",finished_at=time.time(),final_evaluation=final)
            (directory/"run.json").write_text(json.dumps(record,indent=2)+"\n")
            overall["runs"].append({k:v for k,v in record.items() if k!="command"})
            (out/"summary.json").write_text(json.dumps(overall,indent=2)+"\n")
            print("SEED_FINISHED",seed,final["development"]["success_at_end_rate"],flush=True)
        status=0
    except Exception as exc:
        overall["error"]=type(exc).__name__+": "+str(exc)
        (out/"summary.json").write_text(json.dumps(overall,indent=2)+"\n")
        raise
    finally:
        (out/"batch.exit").write_text(str(status)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,default=Path("runs/stackcube-official"))
    p.add_argument("--seeds",type=int,nargs="+",default=[1,2,3])
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or any(not 0<=s<2**31 for s in args.seeds):p.error("invalid seeds")
    if args.execute:execute(args.output_root.resolve(),args.seeds,args.smoke)
    else:
        for seed in args.seeds:print(json.dumps(command(seed,f"stackcube-seed-{seed}",args.smoke)))


if __name__=="__main__":main()
