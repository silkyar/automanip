"""Training-side conversion of an official tensor checkpoint to our safe format.

This utility is never imported by the evaluator. It only accepts the known
official actor/critic tensor schema and uses PyTorch's restricted weights loader.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path

from automanip.policy_io import load_policy, PolicySchema
from automanip.stackcube import STACK_TASK, OFFICIAL_PPO_SHA256


def convert(checkpoint, destination):
    import torch
    checkpoint,destination = Path(checkpoint),Path(destination)
    with checkpoint.open("rb") as stream:
        raw = stream.read(64*1024*1024+1)
    if len(raw)>64*1024*1024:
        raise ValueError("checkpoint byte limit")
    state = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    shapes = {"actor_logstd":(1,8)}
    for prefix,outputs in (("actor_mean",8),("critic",1)):
        dims = (48,256,256,256,outputs)
        for i,(left,right) in enumerate(zip(dims,dims[1:])):
            shapes[f"{prefix}.{2*i}.weight"] = (right,left)
            shapes[f"{prefix}.{2*i}.bias"] = (right,)
    if type(state) not in (dict,OrderedDict) or set(state)!=set(shapes):
        raise ValueError("unexpected official checkpoint keys")
    for key,shape in shapes.items():
        v=state[key]
        if type(v) is not torch.Tensor or tuple(v.shape)!=shape or v.dtype!=torch.float32 or not torch.isfinite(v).all():
            raise ValueError("invalid official tensor: "+key)
    tensors={}
    for i in range(4):
        for suffix in ("weight","bias"):
            tensor=state[f"actor_mean.{2*i}.{suffix}"]
            tensors[f"layers.{i}.{suffix}"]=dict(shape=list(tensor.shape),values=tensor.flatten().tolist())
    weights=json.dumps(tensors,allow_nan=False,separators=(",",":")).encode()
    digest=hashlib.sha256(weights).hexdigest()
    manifest=dict(format="automanip-json-f32-v1",task_fingerprint=STACK_TASK.fingerprint,
                  observation_dim=48,action_dim=8,hidden_sizes=[256,256,256],
                  activation="tanh",weights_sha256=digest)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"weights.json").write_bytes(weights)
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    load_policy(destination,PolicySchema(STACK_TASK.fingerprint,48,8))
    record=dict(checkpoint_sha256=hashlib.sha256(raw).hexdigest(),weights_sha256=digest,
                expected_official_ppo_sha256=OFFICIAL_PPO_SHA256,
                conversion="restricted weights-only load, exact tensor schema, actor-only JSON export")
    destination.with_suffix(".export.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    print(json.dumps(convert(args.checkpoint,args.destination)))


if __name__=="__main__":main()
