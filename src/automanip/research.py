"""Bounded experiment configuration and data snapshots; no candidate code loading."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time


def validate_campaign(value):
    if type(value) is not dict or set(value) != {
        "campaign_id", "hypothesis", "requested_transitions", "training_seeds", "candidates"
    }:
        raise ValueError("unexpected campaign fields")
    overnight=type(value["campaign_id"]) is str and value["campaign_id"].startswith("stackcube-overnight-20261027-")
    settling=value["campaign_id"]=="stackcube-settling-time-20261027"
    if value["campaign_id"] != "stackcube-credit-20261026" and not overnight and not settling:
        raise ValueError("unknown campaign")
    if type(value["hypothesis"]) is not str or not 1 <= len(value["hypothesis"]) <= 1000:
        raise ValueError("hypothesis required")
    if type(value["requested_transitions"]) is not int or value["requested_transitions"] not in ((12_500_000,25_000_000) if overnight else (25_000_000,)):
        raise ValueError("transition budget is fixed")
    if value["training_seeds"] != [1, 2, 3] or any(type(s) is not int for s in value["training_seeds"]):
        raise ValueError("paired training seeds are fixed")
    if overnight:
        from automanip.overnight import validate_overnight
        return validate_overnight(value)
    if settling:
        from automanip.settling import validate_settling
        return validate_settling(value)
    if type(value["candidates"]) is not list or len(value["candidates"]) != 2:
        raise ValueError("expected two preregistered candidates")
    for candidate, expected in zip(value["candidates"], (("gamma090", .9), ("gamma095", .95))):
        if type(candidate) is not dict or set(candidate) != {"id", "gamma"}:
            raise ValueError("only discount may vary")
        if (candidate["id"], candidate["gamma"]) != expected or type(candidate["gamma"]) is not float:
            raise ValueError("unregistered discount")
    return value


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def training_command(python, trainer, candidate, seed, smoke=False, requested_transitions=25_000_000):
    if candidate not in ({"id": "gamma090", "gamma": .9}, {"id": "gamma095", "gamma": .95}):
        if type(candidate) is dict and str(candidate.get('id','')).startswith('settle-'):
            from automanip.settling import validate_recipe
        else:
            from automanip.overnight import validate_recipe
        validate_recipe(candidate)
    if type(requested_transitions) is not int or requested_transitions not in (12_500_000,25_000_000):
        raise ValueError("unknown transition budget")
    if type(seed) is not int or seed not in (1, 2, 3): raise ValueError("unknown training seed")
    name = candidate["id"] + f"-seed-{seed}" + ("-smoke" if smoke else "")
    command=[str(python), str(trainer), "--env-id", "StackCube-v1", "--seed", str(seed),
            "--num-envs", "8" if smoke else "1024", "--update-epochs", str(candidate.get("update_epochs",8)),
            "--num-minibatches", "2" if smoke else "32", "--total-timesteps",
            "400" if smoke else str(requested_transitions), "--num-eval-envs", "8",
            "--no-capture-video", "--no-track", "--exp-name", name, "--gamma", str(candidate["gamma"])]
    if "ent_coef" in candidate:command.extend(["--ent-coef",str(candidate["ent_coef"]),"--learning-rate",str(candidate["learning_rate"])])
    if candidate.get('partial_reset') is False:command.append('--no-partial-reset')
    return command


def snapshot_file(source, destination, limit, uid=None, gid=None):
    """Copy a bounded regular file through one descriptor; refuse symlink inputs."""
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError("regular input required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit: raise ValueError("snapshot byte limit")
    finally: os.close(fd)
    destination = Path(destination)
    with destination.open("xb") as stream: stream.write(raw)
    if uid is not None: os.chown(destination, uid, gid)
    destination.chmod(0o440)
    return hashlib.sha256(raw).hexdigest()


def paired_development(candidate, baseline):
    for report in (candidate, baseline):
        if report.get("split") != "development" or report.get("smoke") or report.get("episode_count") != 128:
            raise ValueError("full development reports required")
        if report.get("failed_episodes"): raise ValueError("evaluation execution failure")
    if candidate["protocol_sha256"] != baseline["protocol_sha256"]: raise ValueError("protocol mismatch")
    def outcomes(report):
        return [bool(e["outcome"]["terminal_success"] and not e["outcome"]["failed"] and
                     (e["outcome"]["terminated"] or e["outcome"]["truncated"]))
                for e in report["episodes"]]
    a, b = outcomes(candidate), outcomes(baseline)
    if len(a) != 128 or len(b) != 128: raise ValueError("episode count mismatch")
    wins = sum(x and not y for x, y in zip(a, b))
    losses = sum(y and not x for x, y in zip(a, b))
    return dict(candidate_successes=sum(a), baseline_successes=sum(b),
                paired_wins=wins, paired_losses=losses, success_rate_delta=(wins-losses)/128)


class CheckpointMirror:
    """Root copies completed/stable training files to durable storage every five seconds."""
    def __init__(self, source, destination):
        self.source,self.destination=Path(source),Path(destination)
        self.seen,self.copied,self.manifest={},{},{}
        self.stop=threading.Event();self.error=None

    def poll(self, force=False):
        if not self.source.exists(): return
        self.destination.mkdir(parents=True,exist_ok=True)
        for source in self.source.iterdir():
            if not (re.fullmatch(r"(?:ckpt_\d+|final_ckpt)\.pt",source.name) or
                    source.name.startswith("events.out.tfevents.") or
                    source.name in ("learning-curves.json","learning-curves.csv")):
                continue
            signature=(source.lstat().st_size,source.lstat().st_mtime_ns)
            previous=self.seen.get(source.name);self.seen[source.name]=signature
            if self.copied.get(source.name)==signature or (not force and previous!=signature):continue
            temporary=self.destination/(source.name+".copying")
            temporary.unlink(missing_ok=True)
            digest=snapshot_file(source,temporary,64*1024*1024)
            temporary.replace(self.destination/source.name)
            self.copied[source.name]=signature
            self.manifest[source.name]=dict(sha256=digest,bytes=signature[0],copied_at=time.time())
        target=self.destination/"durable-files.json"
        temporary=target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest,indent=2)+"\n");temporary.replace(target)

    def __enter__(self):
        def worker():
            while not self.stop.wait(5):
                try:self.poll()
                except Exception as exc:self.error=exc;return
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join()
        self.poll(force=True)
        if self.error:raise RuntimeError("durable checkpoint mirroring failed") from self.error
