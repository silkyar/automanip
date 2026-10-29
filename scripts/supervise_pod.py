"""Detached local shutdown watchdog; its key never enters the GPU Pod.

The host must remain awake and connected. This is not an always-on cloud service.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from automanip.supervisor import RunpodClient, StopPolicy, load_api_key


def ssh_prefix(pod,args):
    address=pod.get("ssh",{}).get("direct")
    if not address:return None
    return ["ssh","-F","/dev/null","-i",str(args.ssh_key),"-o","IdentitiesOnly=yes",
            "-o","BatchMode=yes","-o","ConnectTimeout=5","-o","ServerAliveInterval=5",
            "-o","ServerAliveCountMax=1","-o","StrictHostKeyChecking=accept-new",
            "-o","UserKnownHostsFile="+str(args.known_hosts),"-p",str(address["port"]),
            address["username"]+"@"+address["host"]]


def completion_probe_command(remote_root, completion_file):
    if not completion_file or Path(completion_file).name != completion_file or completion_file in (".", ".."):
        raise ValueError("completion file must be a basename")
    root=shlex.quote(remote_root)
    target=shlex.quote(remote_root.rstrip("/")+"/"+completion_file)
    return (f"test -f {target} || "
            f"(test -f {root}/setup.exit && test \"$(cat {root}/setup.exit)\" != 0) || "
            f"(test -f {root}/batch.exit && test \"$(cat {root}/batch.exit)\" != 0)")


def probe(pod,args):
    prefix=ssh_prefix(pod,args)
    if not prefix:return False
    command=completion_probe_command(args.remote_root,args.completion_file)
    try:
        return subprocess.run(prefix+[command],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,timeout=15).returncode==0
    except (subprocess.TimeoutExpired,OSError):return False


def backup(pod,args):
    prefix=ssh_prefix(pod,args)
    if not prefix:return False
    remote=args.remote_root.rstrip("/")+"-results.tar.gz"
    parent,name=str(Path(args.remote_root).parent),Path(args.remote_root).name
    command="tar -czf "+shlex.quote(remote)+" -C "+shlex.quote(parent)+" "+shlex.quote(name)+" && sha256sum "+shlex.quote(remote)
    try:
        result=subprocess.run(prefix+[command],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,timeout=90,check=True,text=True)
        remote_digest=result.stdout.split()[0]
        address=pod["ssh"]["direct"]
        target=address["username"]+"@"+address["host"]+":"+remote
        scp=["scp","-F","/dev/null","-i",str(args.ssh_key),"-o","IdentitiesOnly=yes","-o","BatchMode=yes",
             "-o","ConnectTimeout=5","-o","StrictHostKeyChecking=accept-new",
             "-o","UserKnownHostsFile="+str(args.known_hosts),"-P",str(address["port"]),target,str(args.archive)]
        subprocess.run(scp,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=90,check=True)
        return hashlib.sha256(args.archive.read_bytes()).hexdigest()==remote_digest
    except (subprocess.SubprocessError,OSError):return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id",required=True)
    p.add_argument("--env-file",type=Path,default=Path(".env"))
    p.add_argument("--ssh-key",type=Path,required=True)
    p.add_argument("--known-hosts",type=Path,required=True)
    p.add_argument("--remote-root",required=True)
    p.add_argument("--completion-file",default="batch.exit")
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--max-seconds",type=float,default=28800)
    p.add_argument("--max-usd",type=float,default=5)
    p.add_argument("--hourly-rate",type=float,required=True)
    p.add_argument("--check-access",action="store_true")
    args=p.parse_args()
    completion_probe_command(args.remote_root,args.completion_file)
    client=RunpodClient(load_api_key(args.env_file),args.pod_id)
    pod=client.get()
    if args.check_access:
        print(json.dumps(dict(pod_id=pod["id"],status=pod["status"],authenticated=True)))
        return
    started=datetime.fromisoformat(pod["startedAt"].replace("Z","+00:00")).timestamp()
    policy=StopPolicy(started,args.max_seconds,args.max_usd,args.hourly_rate)
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    record=dict(pod_id=args.pod_id,started_at=started,max_seconds=args.max_seconds,max_usd=args.max_usd,
                hourly_rate=args.hourly_rate,supervisor_started_at=time.time(),state="watching",
                completion_file=args.completion_file)
    while True:
        try:
            pod=client.get()
            if pod["status"]=="EXITED":
                record.update(state="stopped",pod_status=pod["status"],observed_at=time.time())
                args.state.write_text(json.dumps(record,indent=2)+"\n")
                return
            reason=policy.reason(time.time(),probe(pod,args))
            record.update(last_poll_at=time.time(),pod_status=pod["status"])
            if reason:
                remaining=min(policy.max_seconds,policy.max_usd/policy.hourly_rate*3600)-(time.time()-started)
                # A backup must not consume the final deadline/cost reserve.
                record.update(stop_reason=reason,backup_ok=backup(pod,args) if reason=="batch_finished" and remaining>200 else False)
                if record["backup_ok"]:
                    record["archive_sha256"]=hashlib.sha256(args.archive.read_bytes()).hexdigest()
                client.stop()
                record["state"]="stopping"
            args.state.write_text(json.dumps(record,indent=2)+"\n")
        except Exception as exc:
            record["last_error"]=type(exc).__name__+": "+str(exc)
            args.state.write_text(json.dumps(record,indent=2)+"\n")
        time.sleep(15)


if __name__=="__main__":main()
