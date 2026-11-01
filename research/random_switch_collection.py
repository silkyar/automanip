"""TRAINING corpus from randomized switch times; no evaluator/benchmark imports.

The existing actor and fixed zero-arm/open-gripper option are explicit priors.
No geometry, task success, privileged velocity or outcome selects an action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


class RandomSwitchPolicy:
    def __init__(self,base,std,treatment,decision,noise_seed):
        self.base=base;self.std=np.asarray(std,dtype=float)
        self.treatment=np.asarray(treatment).copy();self.decision=np.asarray(decision).copy()
        if self.std.shape!=(8,) or not np.isfinite(self.std).all() or (self.std<=0).any() or (self.std>1).any():raise ValueError('Gaussian scale differs')
        if self.treatment.shape!=(64,) or sorted(self.treatment.tolist())!=[0]*32+[1]*32 or self.decision.shape!=(64,) or not np.isin(self.decision,np.arange(10,45)).all():raise ValueError('Closed balanced randomization required')
        self.rng=np.random.default_rng(noise_seed);self.step=0;self.applied=np.zeros(64,int)

    def act(self,observations):
        x=np.asarray(observations,dtype=float)
        if x.shape!=(64,48) or not np.isfinite(x).all():raise ValueError('Invalid observation batch')
        for i,(weight,bias) in enumerate(self.base.layers):
            x=x@weight.T+bias
            if i<len(self.base.layers)-1:x=np.tanh(x)
        self.raw_proposal=x+self.std*self.rng.standard_normal(x.shape)
        actions=np.clip(self.raw_proposal,-1,1)
        active=(self.treatment==1)&(self.step>=self.decision)
        actions[active,:7]=0;actions[active,7]=1
        self.applied[active]+=1;self.step+=1
        return actions.astype(np.float32)


def collect(inputs,output,index):
    import torch
    import gymnasium as gym
    import mani_skill.envs
    from importlib.metadata import version
    from mani_skill.envs.tasks.tabletop import stack_cube
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.numpy_inference import NumpyPolicy
    inputs,output=Path(inputs),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    if type(index) is not int or index not in [0,1]:raise ValueError('Closed source index')
    plan=json.loads((inputs/'plan.json').read_text());source=plan['sources'][index];env=None
    report=dict(state='started',partition='training',source=source,environment_reserved=0,environment_completed=0,optimizer_calls=0,actor_forward_rows=0,object_velocity_query_vectors_reserved=0,object_velocity_query_vectors_completed=0,episodes_completed=0,batches=[],protected_evaluator_queries=0,teacher_queries=0)
    journal=(output/'work.jsonl').open('x',buffering=1)
    def persist():
        report['wall_seconds']=time.monotonic()-started
        temp=output/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(output/'report.json')
    try:
        report['versions']={k:version(k) for k in plan['production_pins']}
        if report['versions']!=plan['production_pins']:raise ValueError('Dependency pin differs')
        report['GPU']=torch.cuda.get_device_name(0)
        if report['GPU']!='NVIDIA L40S':raise ValueError('Hardware differs')
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=plan['task_source_sha256']:raise ValueError('Pinned task differs')
        blob=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(blob.read_bytes()).hexdigest()!=plan['physx_sha256']:raise ValueError('PhysX pin differs')
        for name,digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=digest:raise ValueError('Safe parent policy differs')
        schema=PolicySchema(plan['task_fingerprint'],48,8)
        actor=NumpyPolicy(load_policy(inputs/source['policy'],schema),schema)
        owner=gym.make('StackCube-v1',robot_uids='panda_wristcam',obs_mode='state',control_mode='pd_joint_delta_pos',max_episode_steps=50,num_envs=64,sim_backend='physx_cuda',reward_mode='normalized_dense',reconfiguration_freq=1);env=owner.unwrapped
        if env.single_observation_space.shape!=(48,) or env.single_action_space.shape!=(8,) or env.control_freq!=20 or env.sim_freq!=100:raise ValueError('Observation/action/clock contract differs')
        for batch in range(32):
            assignment=np.random.default_rng(source['assignment_seed']+batch)
            treatment=assignment.permutation(np.tile([0,1],32));decision=assignment.integers(10,45,size=64)
            policy=RandomSwitchPolicy(actor,source['gaussian_std'],treatment,decision,source['noise_seed']+batch)
            seeds=list(range(source['reset_seed']+batch*64,source['reset_seed']+(batch+1)*64))
            obs,_=env.reset(seed=seeds);observations=[obs.detach().cpu().numpy().copy()]
            actions=[];raw=[];rewards=[];flags=[];terminated=[];truncated=[];linear=[];angular=[]
            state_at_decision=np.full((64,48),np.nan,dtype=np.float32)
            for step in range(50):
                state=observations[-1];selected=decision==step;state_at_decision[selected]=state[selected]
                action=policy.act(state);report['actor_forward_rows']+=64
                if report['environment_reserved']+64>102400:raise ValueError('Transition cap')
                report['environment_reserved']+=64;journal.write(json.dumps(dict(event='environment_reserved',count=report['environment_reserved']))+'\n')
                obs,reward,term,trunc,info=env.step(torch.tensor(action,device=env.device))
                report['environment_completed']+=64;journal.write(json.dumps(dict(event='environment_completed',count=report['environment_completed']))+'\n')
                observations.append(obs.detach().cpu().numpy().copy());actions.append(action.copy());raw.append(policy.raw_proposal.astype(np.float32));rewards.append(reward.detach().cpu().numpy().copy())
                flags.append(np.stack([info[k].detach().cpu().numpy() for k in ['is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success']],axis=-1))
                terminated.append(term.detach().cpu().numpy().copy());truncated.append(trunc.detach().cpu().numpy().copy())
                # Privileged diagnostic columns are excluded from outcome-model
                # features. Two native velocity vector reads per environment.
                report['object_velocity_query_vectors_reserved']+=128
                linear.append(env.cubeA.linear_velocity.detach().cpu().numpy().copy());angular.append(env.cubeA.angular_velocity.detach().cpu().numpy().copy())
                report['object_velocity_query_vectors_completed']+=128
            if not np.isfinite(state_at_decision).all() or not np.array_equal(policy.applied,np.where(treatment==1,50-decision,0)):raise ValueError('Decision or command budget differs')
            outcome=np.asarray(flags)[-1,:,3].astype(bool)
            np.savez_compressed(output/f'batch-{batch}.npz',observations=np.asarray(observations),executed_actions=np.asarray(actions),raw_proposals=np.asarray(raw),rewards=np.asarray(rewards),task_flags=np.asarray(flags),terminated=np.asarray(terminated),truncated=np.asarray(truncated),diagnostic_cube_linear_velocity=np.asarray(linear),diagnostic_cube_angular_velocity=np.asarray(angular),state_at_decision=state_at_decision,treatment=treatment,decision=decision,outcome=outcome,applied=policy.applied,reset_seeds=np.asarray(seeds))
            report['episodes_completed']+=64
            report['batches'].append(dict(batch=batch,episodes=64,base_successes=int(outcome[treatment==0].sum()),correction_successes=int(outcome[treatment==1].sum()),overridden_command_rows=int(policy.applied.sum())))
            persist()
        if report['environment_completed']!=102400 or report['episodes_completed']!=2048:raise ValueError('Incomplete corpus')
        report['state']='complete'
    except Exception as exc:
        import traceback
        report.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if env is not None:owner.close()
        journal.close();persist()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--index',required=True,type=int,choices=[0,1]);args=p.parse_args()
    result=collect(args.inputs,args.output,args.index);print(json.dumps({k:v for k,v in result.items() if k not in ['batches','traceback']}));raise SystemExit(0 if result['state']=='complete' else 1)
