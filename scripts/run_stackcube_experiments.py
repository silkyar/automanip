"""Run a bounded discount sweep with separate trainer, converter and evaluator UIDs."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import sys
import time

from automanip.policy_io import load_policy, PolicySchema, MAX_MANIFEST_BYTES, MAX_WEIGHTS_BYTES
from automanip.research import validate_campaign, canonical_digest, training_command, snapshot_file, paired_development, CheckpointMirror
from automanip.stackcube import STACK_TASK, OFFICIAL_PPO_SHA256, check_frozen_protocol
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_stackcube_baseline import run_child, events

ACCESS_PROBE = """import os,stat,sys
p,flag=sys.argv[1:]
try:
 directory=stat.S_ISDIR(os.stat(p).st_mode)
 if flag=='-r':
  if directory: os.listdir(p)
  else:
   with open(p,'rb') as f:f.read(1)
 elif directory:
  probe=os.path.join(p,'.automanip-write-check-'+str(os.getpid()))
  fd=os.open(probe,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  os.close(fd);os.unlink(probe)
 else:
  fd=os.open(p,os.O_WRONLY|os.O_APPEND);os.close(fd)
except PermissionError:sys.exit(13)
"""


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def account(name):
    try: return pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["useradd","--system","--user-group","--create-home","--shell","/usr/sbin/nologin",name],check=True)
        return pwd.getpwnam(name)


def directory(path, uid, gid, mode=0o700):
    path.mkdir(parents=True,exist_ok=True)
    if uid or gid:os.chown(path,uid,gid)
    path.chmod(mode)
    return path


def readonly_tree(path, gid):
    for item in [path,*path.rglob("*")]:
        if item.is_symlink(): raise ValueError("unexpected source symlink")
        os.chown(item,0,gid);item.chmod(0o750 if item.is_dir() else 0o640)


def role_command(user, pythonpath, command, deadline=None):
    role=user.removeprefix("am-")
    timing=[] if deadline is None else ['AUTOMANIP_TRAINING_DEADLINE='+str(deadline)]
    return ["runuser","-u",user,"--","env",*timing,"PYTHONPATH="+str(pythonpath),
            "PYTHONDONTWRITEBYTECODE=1","MS_ASSET_DIR="+str(ROOT.parent/"assets"),
            sys.executable,ROOT.parent/"sandbox_exec.py","--policy",ROOT.parent/"sandbox-policies"/(role+".json"),
            "--",*map(str,command)]


def prepare(out):
    if sys.platform!="linux" or os.geteuid()!=0: raise RuntimeError("execution requires Linux root controller")
    if ROOT!=Path("/opt/automanip-credit-20261026/source"):
        raise RuntimeError("execute only from a staged campaign source directory")
    users={role:account("am-"+role) for role in ("train","convert","eval")}
    directory(out,0,0)
    work=directory(ROOT.parent/("smoke-work" if out.name=="smoke" else "work"),0,0,0o711)
    readonly_tree(ROOT,users["eval"].pw_gid)
    program=directory(work/"trainer-program",0,users["train"].pw_gid,0o750)
    shutil.copyfile(ROOT/"third_party/maniskill/ppo.py",program/"ppo.py")
    # The only derivative adds the registered training entropy schedule.
    from automanip.settling import campaign as settling_campaign, trainer_bytes
    for candidate in settling_campaign()['candidates']:
        (program/(candidate['id']+'.py')).write_bytes(trainer_bytes((program/'ppo.py').read_bytes(),candidate))
    readonly_tree(program,users["train"].pw_gid)
    converter=directory(work/"converter-program",0,users["convert"].pw_gid,0o750)
    for name in ("scripts/export_official_policy.py","src/automanip/__init__.py",
                 "src/automanip/policy_io.py","src/automanip/task_spec.py","src/automanip/stackcube.py"):
        destination=converter/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    readonly_tree(converter,users["convert"].pw_gid)
    for name,role in (("training","train"),("conversion","convert")):
        directory(work/name,0,users[role].pw_gid,0o750)
    directory(work/"training/runs",0,users["train"].pw_gid,0o750)
    directory(work/"training/runs/permission-probe",users["train"].pw_uid,users["train"].pw_gid)
    directory(work/"evaluations",users["eval"].pw_uid,users["eval"].pw_gid)
    directory(work/"inputs",0,users["eval"].pw_gid,0o750)
    directory(out/"records",0,0)
    directory(out/"baseline-development",0,0)
    directory(out/"evaluations",0,0)
    directory(out/"artifacts",0,0)
    shutil.copyfile(ROOT/"scripts/sandbox_exec.py",ROOT.parent/"sandbox_exec.py")
    (ROOT.parent/"sandbox_exec.py").chmod(0o644)
    policies=directory(ROOT.parent/"sandbox-policies",0,0,0o755)
    directory(out/"sandbox-policies",0,0)
    shared=["/usr","/lib","/lib64","/bin","/sbin","/etc","/proc","/sys","/run","/var",
            str(Path(sys.executable).parents[1]),str(ROOT.parent/"assets")]
    for role,user in users.items():
        read={"train":[program],"convert":[converter],"eval":[ROOT,work/"inputs"]}[role]
        writable={"train":[work/"training"],"convert":[work/"conversion"],"eval":[work/"evaluations"]}[role]
        # CUDA names its own threads through procfs. The sandbox helper and its
        # exec target share a PID, so this grants only that worker's task tree.
        policy=dict(read_execute=shared+list(map(str,read)),read_write=["/tmp","/dev","/proc/self/task",user.pw_dir]+list(map(str,writable)))
        target=policies/(role+".json");target.write_text(json.dumps(policy,indent=2)+"\n");target.chmod(0o644)
        write_json(out/"sandbox-policies"/(role+".json"),policy)
    (work/"inputs/permission-probe").write_text("not a policy\n")
    os.chown(work/"inputs/permission-probe",0,users["eval"].pw_gid)
    (work/"inputs/permission-probe").chmod(0o440)
    checks=[]
    for role,flag,path,expected in (
        ("train","-r",ROOT/"scripts/evaluate_stackcube.py",False),
        ("train","-w",ROOT/"third_party/maniskill/ppo.py",False),
        ("train","-r",program/"ppo.py",True),
        ("train","-w",program/"ppo.py",False),
        ("train","-r",work/"inputs/permission-probe",False),
        ("train","-r",out,False),
        ("train","-w",out,False),
        ("train","-w",out/"records",False),
        ("train","-w",work/"training",False),
        ("train","-w",work/"training/runs/permission-probe",True),
        ("convert","-r",work/"inputs/permission-probe",False),
        ("convert","-w",work/"evaluations",False),
        ("eval","-r",work/"inputs/permission-probe",True),
        ("eval","-w",work/"inputs/permission-probe",False),
        ("eval","-w",ROOT/"scripts/evaluate_stackcube.py",False),
        ("eval","-r",work/"training/runs/permission-probe",False),
        ("eval","-w",work/"evaluations",True)):
        code=subprocess.run(role_command(users[role].pw_name,"",[sys.executable,"-c",ACCESS_PROBE,str(path),flag])).returncode
        if code not in (0,13):raise RuntimeError("access probe failed to execute")
        actual=code==0
        checks.append(dict(role=role,test=flag,path=str(path),expected=expected,actual=actual))
        write_json(out/"permission-checks.json",dict(checks=checks))
        if actual!=expected: raise RuntimeError(f"filesystem permission boundary failed: {role} {flag} {path}")
    write_json(out/"permission-checks.json",dict(users={k:dict(uid=v.pw_uid,gid=v.pw_gid,groups=os.getgrouplist(v.pw_name,v.pw_gid)) for k,v in users.items()},checks=checks,
        limitation="Landlock allowlists plus separate UIDs share a kernel/GPU. This is not VM isolation or protection from a root-capable controller."))
    return users,work


def execute(out, config, smoke=False):
    protocol=check_frozen_protocol()
    if (out/"research.exit").exists(): raise RuntimeError("completed campaign; use a new output root")
    users,work_root=prepare(out)
    heartbeat=out/"heartbeat.json"
    py=Path(sys.executable)
    overall=dict(campaign=config,configuration_sha256=canonical_digest(config),protocol_sha256=protocol,
                 official_ppo_sha256=OFFICIAL_PPO_SHA256,mode="smoke" if smoke else "research",
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 evaluation_split="development",confirmation_queries=0,
                 versions={n:version(n) for n in ("mani_skill","torch","sapien","numpy","gymnasium","tensorboard")},runs=[])
    write_json(out/"campaign.json",overall)
    status=1
    def run(role, command, cwd, log, seconds, deadline=None):
        search_path=(work_root/"converter-program/src") if role=="convert" else (ROOT/"src" if role=="eval" else "")
        return run_child(role_command(users[role].pw_name,search_path,command,deadline),cwd,log,seconds,heartbeat)
    def evaluate(checkpoint, record_dir, label, name):
        converter=users["convert"];evaluator=users["eval"]
        work=directory(work_root/"conversion"/(name+"-"+label),converter.pw_uid,converter.pw_gid)
        native_digest=snapshot_file(checkpoint,work/"checkpoint.pt",64*1024*1024,converter.pw_uid,converter.pw_gid)
        run("convert",[py,work_root/"converter-program/scripts/export_official_policy.py",work/"checkpoint.pt",work/"policy"],
            work,record_dir/(label+"-export.log"),180)
        inputs=directory(work_root/"inputs"/(name+"-"+label),0,evaluator.pw_gid,0o750)
        persisted=directory(out/"artifacts"/(name+"-"+label),0,0)
        for filename,limit in (("manifest.json",MAX_MANIFEST_BYTES),("weights.json",MAX_WEIGHTS_BYTES)):
            snapshot_file(work/"policy"/filename,inputs/filename,limit,0,evaluator.pw_gid)
            snapshot_file(inputs/filename,persisted/filename,limit)
        policy=load_policy(inputs,PolicySchema(STACK_TASK.fingerprint,48,8))
        report=work_root/"evaluations"/(name+"-"+label+"-development.json")
        command=[py,ROOT/"scripts/evaluate_stackcube.py",inputs,"--split","development","--output",report]
        if smoke: command.append("--smoke")
        elapsed=run("eval",command,work_root/"evaluations",record_dir/(label+"-evaluation.log"),1200)
        snapshot_file(report,out/"evaluations"/report.name,8*1024*1024)
        result=json.loads(report.read_text())
        if result["weights_sha256"]!=policy.weights_sha256 or result["protocol_sha256"]!=protocol:
            raise RuntimeError("evaluation artifact/protocol mismatch")
        return dict(report=result,evaluation_seconds=elapsed,checkpoint_sha256=native_digest,weights_sha256=policy.weights_sha256)
    try:
        run("eval",[py,ROOT/"scripts/check_stackcube.py"],work_root/"evaluations",out/"preflight.log",180)
        seeds=[1] if smoke else config["training_seeds"]
        candidates=config["candidates"][:1] if smoke and config['campaign_id']!='stackcube-settling-time-20261027' else config["candidates"]
        for seed in seeds:
            for candidate in candidates:
                if check_frozen_protocol()!=protocol: raise RuntimeError("frozen protocol changed")
                trainer=work_root/"trainer-program/ppo.py"
                if hashlib.sha256(trainer.read_bytes()).hexdigest()!=OFFICIAL_PPO_SHA256: raise RuntimeError("trainer source changed")
                if 'max_training_seconds' in candidate:
                    from automanip.settling import trainer_bytes
                    expected=trainer_bytes(trainer.read_bytes(),candidate)
                    trainer=trainer.with_name(candidate['id']+'.py')
                    if trainer.read_bytes()!=expected:raise RuntimeError('entropy trainer derivative changed')
                name=candidate["id"]+f"-seed-{seed}"+("-smoke" if smoke else "")
                record_dir=directory(out/"records"/name,0,0)
                command=training_command(py,trainer,candidate,seed,smoke,config["requested_transitions"])
                record=dict(candidate=candidate,seed=seed,name=name,status="training",command=command,
                            started_at=time.time(),configuration_sha256=overall["configuration_sha256"],
                            protocol_sha256=protocol,hypothesis=config["hypothesis"],
                            trainer_sha256=hashlib.sha256(trainer.read_bytes()).hexdigest(),
                            trainer_kind='budget-and-audit-derivative' if trainer.name!='ppo.py' else 'official-unmodified',
                            max_training_seconds=candidate.get('max_training_seconds'))
                write_json(record_dir/"run.json",record)
                print("TRAINING",name,flush=True)
                trained=directory(work_root/"training/runs"/name,users["train"].pw_uid,users["train"].pw_gid)
                with CheckpointMirror(trained,out/"training/runs"/name) as mirror:
                    seconds=900 if smoke else candidate.get('max_training_seconds',7200)
                    deadline=time.monotonic()+seconds-20 if 'max_training_seconds' in candidate else None
                    elapsed=run("train",command,work_root/"training",record_dir/"train.log",seconds,deadline)
                curves=events(trained)
                mirror.poll(force=True)
                steps=int(curves["time/step"][-1]["value"])
                expected_steps=400 if smoke else config["requested_transitions"]//51_200*51_200
                if 'max_training_seconds' in candidate and not smoke:
                    if not 0<steps<=expected_steps or elapsed>seconds:raise RuntimeError('training budget exceeded or invalid step count')
                elif steps!=expected_steps: raise RuntimeError("transition budget incomplete")
                record.update(status="evaluating",environment_transitions=steps,trainer_process_seconds=elapsed,
                              rollout_seconds=sum(e["value"] for e in curves["time/rollout_time"]),
                              optimization_seconds=sum(e["value"] for e in curves["time/update_time"]))
                write_json(record_dir/"run.json",record)
                curve=[]
                if not smoke:
                    for iteration in range(1,steps//51_200+1,100):
                        label=f"step-{(iteration-1)*51200}"
                        print("CURVE",name,label,flush=True)
                        result=evaluate(trained/f"ckpt_{iteration}.pt",record_dir,label,name)
                        curve.append(dict(environment_steps=(iteration-1)*51200,**result))
                final=evaluate(trained/"final_ckpt.pt",record_dir,"final",name)
                if final["checkpoint_sha256"]!=mirror.manifest["final_ckpt.pt"]["sha256"]:
                    raise RuntimeError("durable checkpoint differs from evaluated checkpoint")
                curve.append(dict(environment_steps=steps,**final))
                write_json(record_dir/"independent-learning-curve.json",curve)
                record.update(status="complete",finished_at=time.time(),final=final)
                if not smoke:
                    baseline=json.loads((out/"baseline-development"/f"seed-{seed}.json").read_text())
                    record["paired_baseline_comparison"]=paired_development(final["report"],baseline)
                    anchor=out/"anchor-development"/f"seed-{seed}.json"
                    if anchor.exists():record["paired_anchor_comparison"]=paired_development(final["report"],json.loads(anchor.read_text()))
                    control=next((r for r in overall['runs'] if r['seed']==seed and r['candidate']['id']=='settle-timed-control'),None)
                    if control is not None:record['paired_time_control_comparison']=paired_development(final['report'],control['final']['report'])
                write_json(record_dir/"run.json",record)
                overall["runs"].append(record)
                write_json(out/"summary.json",overall)
                with (out/"ledger.jsonl").open("a") as stream:stream.write(json.dumps(record,allow_nan=False)+"\n")
                print("RUN_FINISHED",name,final["report"]["success_at_end_rate"],flush=True)
        if not smoke:
            comparisons=[]
            for candidate in candidates:
                runs=[r for r in overall["runs"] if r["candidate"]==candidate]
                mean=sum(r["final"]["report"]["success_at_end_rate"] for r in runs)/3
                deltas=[r["paired_baseline_comparison"]["success_rate_delta"] for r in runs]
                fixed_steps=all(r['environment_transitions']==24_985_600 for r in runs)
                timed='max_training_seconds' in candidate
                control_deltas=[r['paired_time_control_comparison']['success_rate_delta'] for r in runs if 'paired_time_control_comparison' in r]
                comparisons.append(dict(candidate=candidate,mean_development_success=mean,paired_seed_deltas=deltas,
                    mean_paired_delta=sum(deltas)/3,ready_for_confirmation_review=(mean>=.9 and len(control_deltas)==3 and min(control_deltas)>=0) if timed else (fixed_steps and mean>=.6 and min(deltas)>=0),
                    comparison_transition_budget_matched=fixed_steps,
                    comparison_budget='at most 1320 trainer seconds and 25M requested transitions' if timed else 'fixed transitions',
                    paired_time_control_deltas=control_deltas,
                    promoted=False,confirmation_queries=0))
            overall["candidate_comparisons"]=comparisons
            write_json(out/"summary.json",overall)
        status=0
    except Exception as exc:
        overall["error"]=type(exc).__name__+": "+str(exc)
        write_json(out/"summary.json",overall)
        raise
    finally:(out/"research.exit").write_text(str(status)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,default=ROOT/"configs/stackcube-credit-20261026.json")
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--execute",action="store_true")
    args=p.parse_args();requested=json.loads(args.campaign.read_text())
    if requested.get('campaign_id')=='stackcube-settling-20261027':
        from automanip.settling import upgrade_legacy
        config=upgrade_legacy(requested)
        if args.execute:
            args.output_root.mkdir(parents=True,exist_ok=True)
            write_json(args.output_root/'time-budget-amendment.json',dict(requested=requested,effective=config,
                       reason='User requires no increase in trainer time; enforce 1320 seconds and add a paired control.'))
    else:config=requested
    config=validate_campaign(config)
    if args.execute:execute(args.output_root.resolve(),config,args.smoke)
    else:
        for seed in config["training_seeds"]:
            for candidate in config["candidates"]: print(json.dumps(training_command(sys.executable,ROOT/"third_party/maniskill/ppo.py",candidate,seed,requested_transitions=config["requested_transitions"])))


if __name__=="__main__":main()
