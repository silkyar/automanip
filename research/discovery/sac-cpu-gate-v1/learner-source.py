"""Standalone, bounded SAC control adapted from pinned ManiSkill v3.0.0b21.

Candidate trainer only. Trusted evaluation imports none of this module.
No checkpoint loading, demos, planning, teacher or pretrained policy support.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_SHA256 = "27e5d7b90a431ed9af1a6a05e593d6404a67a8cce96f8ae066ddf95f0ab6c51a"
UPSTREAM_COMMIT = "6226567ce326625fa6eacf9aeb3f4008e4339f7e"
TASK_FINGERPRINT = "b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"


def mlp(widths, activation):
    layers = []
    for i, (a,b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a,b))
        if i != len(widths)-2: layers.append(activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim=48, action_dim=8, width=256):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim,width), nn.Tanh(),
                                      nn.Linear(width,width), nn.Tanh())
        self.mean = nn.Linear(width,action_dim)
        self.logstd = nn.Linear(width,action_dim)

    def forward(self, obs):
        feature = self.backbone(obs)
        mean = self.mean(feature)
        logstd = -5 + 3.5*(torch.tanh(self.logstd(feature))+1)
        return mean, logstd

    def deterministic(self, obs):
        return torch.tanh(self.mean(self.backbone(obs)))

    def sample(self, obs):
        mean, logstd = self(obs)
        normal = torch.distributions.Normal(mean,logstd.exp())
        latent = normal.rsample()
        action = torch.tanh(latent)
        # Stable exact tanh Jacobian, unlike subtracting 1-tanh(x)^2 in tails.
        jacobian = 2*(math.log(2)-latent-F.softplus(-2*latent))
        logprob = (normal.log_prob(latent)-jacobian).sum(-1,keepdim=True)
        return action, logprob


class Q(nn.Module):
    def __init__(self, obs_dim=48, action_dim=8, width=256):
        super().__init__()
        self.net = mlp([obs_dim+action_dim,width,width,width,1],nn.ReLU)

    def forward(self, obs, action):
        return self.net(torch.cat((obs,action),-1))


class Replay:
    """Uniform one-transition replay; no sequence can cross episode boundaries."""
    def __init__(self, capacity, obs_dim=48, action_dim=8, device="cpu"):
        if capacity < 1: raise ValueError("positive replay capacity required")
        self.capacity, self.position, self.size = capacity, 0, 0
        self.storage = {
            "obs":torch.empty(capacity,obs_dim,device=device),
            "next_obs":torch.empty(capacity,obs_dim,device=device),
            "action":torch.empty(capacity,action_dim,device=device),
            "reward":torch.empty(capacity,1,device=device),
            "stop":torch.empty(capacity,1,device=device),
        }

    def add(self, obs, next_obs, action, reward, stop):
        count = len(obs)
        if count > self.capacity: raise ValueError("vector batch exceeds replay capacity")
        values = dict(obs=obs,next_obs=next_obs,action=action,
                      reward=reward.reshape(-1,1),stop=stop.reshape(-1,1))
        index = (torch.arange(count,device=self.storage["obs"].device)+self.position)%self.capacity
        for key,value in values.items():
            if not torch.isfinite(value).all(): raise ValueError("nonfinite replay transition")
            self.storage[key][index] = value.detach()
        self.position = (self.position+count)%self.capacity
        self.size = min(self.capacity,self.size+count)

    def sample(self, count):
        if self.size == 0: raise ValueError("cannot sample empty replay")
        index = torch.randint(self.size,(count,),device=self.storage["obs"].device)
        return {k:v[index] for k,v in self.storage.items()}


def transition_successor(next_obs, terminations, truncations, infos):
    """Keep actual final observation; truncate bootstraps, true terminal stops.

    ManiSkill's auto-reset observation is next episode's initial state. The
    pinned wrapper stores a full vector final_observation plus a boolean mask.
    Missing final data at an auto-reset boundary is an error, never silent reuse.
    """
    done = terminations.bool() | truncations.bool()
    actual = next_obs.clone()
    if done.any():
        final = infos.get("final_observation")
        mask = infos.get("_final_observation")
        if final is None or mask is None or not torch.equal(mask.bool(),done):
            raise ValueError("auto-reset boundary requires exact final-observation mask")
        if final.shape != next_obs.shape: raise ValueError("final observation vector shape mismatch")
        actual[done] = final[done]
    return actual, terminations.bool()


class SAC:
    def __init__(self, obs_dim=48, action_dim=8, width=256, gamma=.9,
                 tau=.01, learning_rate=3e-4, device="cpu"):
        self.actor = Actor(obs_dim,action_dim,width).to(device)
        self.q1, self.q2 = Q(obs_dim,action_dim,width).to(device), Q(obs_dim,action_dim,width).to(device)
        self.t1, self.t2 = Q(obs_dim,action_dim,width).to(device), Q(obs_dim,action_dim,width).to(device)
        self.t1.load_state_dict(self.q1.state_dict()); self.t2.load_state_dict(self.q2.state_dict())
        self.t1.requires_grad_(False); self.t2.requires_grad_(False)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.log_alpha = torch.zeros(1,requires_grad=True,device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=learning_rate)
        self.gamma, self.tau, self.target_entropy = gamma,tau,-action_dim
        self.updates = 0

    def update(self, data):
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            action, logprob = self.actor.sample(data["next_obs"])
            future = torch.minimum(self.t1(data["next_obs"],action),self.t2(data["next_obs"],action))-alpha*logprob
            target = data["reward"] + self.gamma*(1-data["stop"])*future
        q1, q2 = self.q1(data["obs"],data["action"]), self.q2(data["obs"],data["action"])
        qloss = F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.q_optimizer.zero_grad(); qloss.backward(); self.q_optimizer.step()
        self.q1.requires_grad_(False); self.q2.requires_grad_(False)
        action, logprob = self.actor.sample(data["obs"])
        actor_loss = (alpha*logprob-torch.minimum(self.q1(data["obs"],action),self.q2(data["obs"],action))).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        self.q1.requires_grad_(True); self.q2.requires_grad_(True)
        with torch.no_grad(): _, alpha_logprob = self.actor.sample(data["obs"])
        # Same alpha objective as pinned official source, not a new temperature rule.
        alpha_loss = -(self.log_alpha.exp()*(alpha_logprob+self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
        with torch.no_grad():
            for current,target_net in ((self.q1,self.t1),(self.q2,self.t2)):
                for parameter,target_parameter in zip(current.parameters(),target_net.parameters()):
                    target_parameter.lerp_(parameter,self.tau)
        self.updates += 1
        values = dict(q_loss=float(qloss.detach()),actor_loss=float(actor_loss.detach()),
                      alpha=float(self.log_alpha.detach().exp()),target_mean=float(target.mean()))
        if not all(math.isfinite(v) for v in values.values()): raise ValueError("nonfinite SAC update")
        return values


def numerical_layers(actor):
    linear = [actor.backbone[0],actor.backbone[2],actor.mean]
    result = {}
    for i,layer in enumerate(linear):
        for name in ("weight","bias"):
            tensor = getattr(layer,name).detach().to(device="cpu",dtype=torch.float32)
            result[f"layers.{i}.{name}"] = dict(shape=list(tensor.shape),values=tensor.flatten().tolist())
    action_dim = actor.mean.out_features
    for name,tensor in (("weight",torch.eye(action_dim)),("bias",torch.zeros(action_dim))):
        result[f"layers.3.{name}"] = dict(shape=list(tensor.shape),values=tensor.flatten().tolist())
    return result


def numerical_action(obs, layers):
    """Independent public numerical contract, no trusted evaluator imports."""
    result = obs
    for i in range(4):
        weight,bias = (layers[f"layers.{i}.{key}"] for key in ("weight","bias"))
        w = torch.tensor(weight["values"],dtype=torch.float32,device=obs.device).reshape(weight["shape"])
        b = torch.tensor(bias["values"],dtype=torch.float32,device=obs.device).reshape(bias["shape"])
        result = result @ w.T + b
        if i<3: result = torch.tanh(result)
    return result.clamp(-1,1)


def export_actor(actor, directory):
    if actor.backbone[0].in_features != 48 or actor.mean.out_features != 8:
        raise ValueError("robot artifact requires public 48/8 interface")
    width = actor.backbone[0].out_features
    if not 1<=width<=256: raise ValueError("actor exceeds public width contract")
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    layers = numerical_layers(actor)
    raw = json.dumps(layers,allow_nan=False,separators=(",",":")).encode()
    manifest = dict(format="automanip-json-f32-v1",task_fingerprint=TASK_FINGERPRINT,
                    observation_dim=48,action_dim=8,hidden_sizes=[width,width,8],
                    activation="tanh",weights_sha256=hashlib.sha256(raw).hexdigest())
    (directory/"weights.json").write_bytes(raw)
    (directory/"manifest.json").write_text(json.dumps(manifest,allow_nan=False,separators=(",",":")))
    return dict(path=str(directory),weights_sha256=manifest["weights_sha256"],bytes=len(raw))


@dataclass
class Config:
    seed: int = 1
    num_envs: int = 32
    total_transitions: int = 500_000
    buffer_size: int = 500_000
    batch_size: int = 1024
    learning_starts: int = 4000
    training_freq: int = 64
    utd: float = .5
    gamma: float = .9
    max_seconds: float = 1320
    checkpoint_every: int = 512_000
    device: str = "cuda"

    def validate(self):
        for name in ("num_envs","total_transitions","buffer_size","batch_size","learning_starts","training_freq","checkpoint_every"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError("positive integer config required")
        if self.training_freq%self.num_envs or self.total_transitions%self.num_envs:
            raise ValueError("transition budget and training frequency must divide num_envs")
        if self.buffer_size<self.num_envs or not 0<self.utd<=1 or int(self.training_freq*self.utd)<1:
            raise ValueError("invalid replay/update budget")
        if not 0<self.gamma<1 or not 20<self.max_seconds<=1320:
            raise ValueError("invalid discount/deadline")
        if self.device not in {"cpu","cuda"}: raise ValueError("explicit CPU/CUDA device required")
        return self


def metadata(config):
    versions = {}
    for package in ("mani_skill","torch","numpy","gymnasium","sapien"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    try: revision = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except subprocess.CalledProcessError: revision = None
    return dict(config=asdict(config),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                upstream_sha256=UPSTREAM_SHA256,upstream_commit=UPSTREAM_COMMIT,
                git_sha=revision,python=platform.python_version(),
                versions=versions,teacher_data=0,demos=0,planning_queries=0,pretrained_weights=False,
                task="StackCube-v1",robot="panda_wristcam",observation_dim=48,action_dim=8,
                controller="pd_joint_delta_pos",horizon=50,reward="normalized_dense",
                training_success_resets=False,algorithm="SAC adapted, not unmodified upstream")


def training_deadline(config, start, absolute=None):
    deadline=start+config.max_seconds-20
    if absolute is not None:
        trusted=float(absolute)
        if not math.isfinite(trusted): raise ValueError("finite absolute deadline required")
        deadline=min(deadline,trusted)
    if deadline<=start: raise TimeoutError("training lease already expired")
    return deadline


def train_robot(config, output):
    """Requires root-reviewed Linux simulator smoke; never called in CPU tests."""
    config.validate()
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    info = metadata(config)
    expected = {"mani_skill":"3.0.0b21","torch":"2.6.0","numpy":"1.26.4","gymnasium":"0.29.1"}
    for package,version in expected.items():
        actual=info["versions"][package]
        if actual is None or actual.split("+")[0] != version: raise ValueError(f"pinned {package} required")
    (output/"run.json").write_text(json.dumps(info,indent=2))
    start = time.monotonic()
    deadline=training_deadline(config,start,os.environ.get("AUTOMANIP_TRAINING_DEADLINE"))
    interrupted = [False]
    signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True))
    random.seed(config.seed); torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    import numpy as np
    np.random.seed(config.seed)
    import gymnasium as gym
    import mani_skill.envs  # register only; no candidate evaluation environment
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    env = gym.make("StackCube-v1",num_envs=config.num_envs,obs_mode="state",reward_mode="normalized_dense",
                   robot_uids="panda_wristcam",control_mode="pd_joint_delta_pos",max_episode_steps=50,
                   sim_backend="physx_cuda" if config.device=="cuda" else "physx_cpu",render_mode="rgb_array")
    env = ManiSkillVectorEnv(env,ignore_terminations=True,record_metrics=True)
    if env.single_observation_space.shape!=(48,) or env.single_action_space.shape!=(8,):
        env.close(); raise ValueError("fixed public interface mismatch")
    if not (np.all(env.single_action_space.low==-1) and np.all(env.single_action_space.high==1)):
        env.close(); raise ValueError("fixed action bounds mismatch")
    device = torch.device(config.device)
    if config.device=="cuda" and not torch.cuda.is_available(): env.close(); raise ValueError("CUDA unavailable")
    learner = SAC(gamma=config.gamma,device=device)
    replay = Replay(config.buffer_size,device=device)
    accounting = dict(transitions=0,vector_step_calls=0,resets=0,reset_calls=0,completed_episodes=0,
                      critic_optimizer_steps=0,actor_optimizer_steps=0,alpha_optimizer_steps=0,
                      minibatch_presentations=0,inference_seconds=0.,step_seconds_including_resets=0.,
                      reset_seconds=0.,replay_seconds=0.,update_seconds=0.,export_seconds=0.)
    def sync():
        if device.type=="cuda": torch.cuda.synchronize(device)
    original_reset = env.reset
    def timed_reset(*args,**kwargs):
        sync(); mark=time.monotonic()
        result=original_reset(*args,**kwargs)
        sync(); accounting["reset_seconds"]+=time.monotonic()-mark
        selected=kwargs.get("options",{}).get("env_idx")
        accounting["resets"]+=config.num_envs if selected is None else len(selected)
        accounting["reset_calls"]+=1
        return result
    env.reset = timed_reset
    exported=[]
    def save(label):
        sync(); mark=time.monotonic()
        exported.append(dict(transitions=accounting["transitions"],
                             **export_actor(learner.actor,output/label)))
        accounting["export_seconds"]+=time.monotonic()-mark
    try:
        obs,_=env.reset(seed=config.seed)
        obs=obs.to(device)
        save("actor-step-0")
        next_checkpoint=config.checkpoint_every
        logs=(output/"training.jsonl").open("x")
        with logs:
            while accounting["transitions"]<config.total_transitions and time.monotonic()<deadline and not interrupted[0]:
                collected=0
                for _ in range(config.training_freq//config.num_envs):
                    if time.monotonic()>=deadline or interrupted[0] or accounting["transitions"]>=config.total_transitions: break
                    sync();mark=time.monotonic()
                    with torch.no_grad():
                        action=(2*torch.rand(config.num_envs,8,device=device)-1 if accounting["transitions"]<config.learning_starts
                                else learner.actor.sample(obs)[0])
                    sync();accounting["inference_seconds"]+=time.monotonic()-mark
                    mark=time.monotonic()
                    next_obs,reward,term,trunc,infos=env.step(action)
                    sync();accounting["step_seconds_including_resets"]+=time.monotonic()-mark
                    accounting["completed_episodes"]+=int((term.bool()|trunc.bool()).sum())
                    if accounting["resets"]!=config.num_envs+accounting["completed_episodes"]:
                        raise ValueError("reset instrumentation disagrees with episode-boundary count")
                    mark=time.monotonic()
                    actual,stop=transition_successor(next_obs,term,trunc,infos)
                    replay.add(obs,actual.to(device),action,reward.to(device),stop.to(device))
                    obs=next_obs.to(device)
                    sync();accounting["replay_seconds"]+=time.monotonic()-mark
                    accounting["transitions"]+=config.num_envs;accounting["vector_step_calls"]+=1
                    collected+=config.num_envs
                metrics={}
                if accounting["transitions"]>=config.learning_starts:
                    for _ in range(int(collected*config.utd)):
                        if time.monotonic()>=deadline or interrupted[0]: break
                        sync();mark=time.monotonic()
                        metrics=learner.update(replay.sample(config.batch_size))
                        sync();accounting["update_seconds"]+=time.monotonic()-mark
                        for key in ("critic_optimizer_steps","actor_optimizer_steps","alpha_optimizer_steps"): accounting[key]+=1
                        accounting["minibatch_presentations"]+=config.batch_size
                if accounting["transitions"]>=next_checkpoint:
                    save(f"actor-step-{accounting['transitions']}")
                    next_checkpoint+=config.checkpoint_every
                logs.write(json.dumps(dict(**accounting,elapsed_seconds=time.monotonic()-start,metrics=metrics),allow_nan=False)+"\n")
                logs.flush()
            save("actor-final")
        report=dict(**accounting,elapsed_seconds=time.monotonic()-start,exports=exported,
                    stopped_by=("signal" if interrupted[0] else "transitions" if accounting["transitions"]>=config.total_transitions else "deadline"),
                    replay_size=replay.size,external_hard_timeout_required=True,
                    extra_simulator_queries=0,evaluation_transitions=0,provider_cost=None)
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        return report
    finally: env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key,field in Config.__dataclass_fields__.items():
        parser.add_argument("--"+key.replace("_","-"),default=field.default,type=type(field.default))
    parser.add_argument("--output",type=Path,required=True)
    args=vars(parser.parse_args());output=args.pop("output")
    print(json.dumps(train_robot(Config(**args),output),indent=2))


if __name__=="__main__": main()
