"""One approved assisted actor-movement probe on fresh public TRAINING seeds.

No benchmark seeds/evaluator imports, no training continuation and no expansion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from first_update_probe import (ProbeArgs, load_production, new_agent, check_reconstruction,
                                kl_holdout, choose_branches, mean_kl, assert_same)

EXPERIMENT='stackcube-first-update-probe-20261028'
INITIAL_SHA='f41f449b6b8c9f655698c8d36fd8e808f0dc1dc1b31eef5aaea96e0b51453d58'
BC_SHA='4bab8b71061341ea985a6bb0384170c66a1b9a963d283942b84a99c2ad42c0a2'
TASK_SOURCE_SHA='ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440'
PINS={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'}
COLLECTION_SEEDS=(90282001,90282002,90282003)
PROBE_SEEDS=(90283001,90283002,90283003)


def instrument(agent, work):
    def actor_forward(module, inputs, output):
        work['actor_forward_rows']+=len(inputs[0])
        work['actor_forward_calls']+=1
    def critic_forward(module, inputs, output):
        work['critic_forward_rows']+=len(inputs[0])
        work['critic_forward_calls']+=1
    # deepcopy used by reference/branches keeps these functions and shared work.
    agent.actor_mean.register_forward_hook(actor_forward)
    agent.critic.register_forward_hook(critic_forward)


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_budget(progress, deadline, count):
    if time.monotonic()>=deadline: raise TimeoutError('Original probe deadline reached; no expansion')
    if progress['new_simulator_transitions']+count>201600: raise RuntimeError('201600 transition cap')
    # Increment BEFORE attempting a simulator step; a failed step is charged too.
    progress['new_simulator_transitions']+=count
    progress['vector_step_calls']+=1


def make_env(n, auto_reset, reconfiguration_freq):
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    owner=gym.make('StackCube-v1',num_envs=n,robot_uids='panda_wristcam',obs_mode='state',
        control_mode='pd_joint_delta_pos',reward_mode='normalized_dense',sim_backend='physx_cuda',
        render_mode='rgb_array',reconfiguration_freq=reconfiguration_freq)
    env=ManiSkillVectorEnv(owner,n,auto_reset=auto_reset,ignore_terminations=True,record_metrics=True)
    if env.single_observation_space.shape!=(48,) or env.single_action_space.shape!=(8,):
        raise RuntimeError('Frozen state/action interface differs')
    return env


@torch.no_grad()
def collect(env, agent, seed, args, out, progress, deadline):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda')
    following,_=env.reset(seed=seed)
    data={name:torch.zeros((50,1024,width),device=device) for name,width in [('obs',48),('actions',8)]}
    data.update({name:torch.zeros((50,1024),device=device) for name in ['rewards','values','logprobs','dones','final_values']})
    done=torch.zeros(1024,device=device)
    phases=[];executed=[]
    np.savez_compressed(out/'rng-before-collection.npz',cpu_rng=torch.get_rng_state().cpu().numpy(),
        cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),numpy_state=np.random.get_state()[1])
    for t in range(50):
        data['obs'][t]=following;data['dones'][t]=done
        action,logprob,_,value=agent.get_action_and_value(following)
        data['actions'][t]=action;data['logprobs'][t]=logprob;data['values'][t]=value.flatten()
        bounded=action.detach().clamp(-1,1)
        check_budget(progress,deadline,1024)
        following,reward,terminated,truncated,info=env.step(bounded)
        if terminated.any() or (t<49 and truncated.any()) or (t==49 and not truncated.all()):
            raise RuntimeError('Expected unchanged full50-step episodes')
        done=torch.logical_or(terminated,truncated).float()
        data['rewards'][t]=reward.view(-1)
        actual_info=info['final_info'] if 'final_info' in info else info
        phases.append(torch.stack([actual_info[k].bool() for k in
            ['is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success']],-1).cpu().numpy())
        executed.append(bounded.cpu().numpy())
        if 'final_info' in info:
            mask=info['_final_info']
            data['final_values'][t,torch.arange(1024,device=device)[mask]]=agent.get_value(info['final_observation'][mask]).view(-1)
    data.update(next_obs=following,next_done=done)
    np.savez_compressed(out/'collection.npz',**{k:v.cpu().numpy() for k,v in data.items()},
        executed_actions=np.stack(executed),stage_flags=np.stack(phases),
        stage_names=np.array(['grasp','on_top','static','stock_success']))
    np.savez_compressed(out/'rng-before-update.npz',cpu_rng=torch.get_rng_state().cpu().numpy(),
        cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),numpy_state=np.random.get_state()[1])
    return data,dict(final_successes=int(actual_info['success'].sum()),episodes=1024,
        raw_action_clip_fraction=float((data['actions'].abs()>1).float().mean()),
        mean_dense_return=float(data['rewards'].sum(0).mean()))


@torch.no_grad()
def task_probe(env, actor, seed, out, progress, deadline):
    obs,_=env.reset(seed=seed)
    base=env.unwrapped
    height=base.cubeA.pose.p[:,2].clone()
    distance=torch.linalg.vector_norm((base.cubeA.pose.p-base.cubeB.pose.p)[:,:2],dim=1)
    actions=[];rewards=[];phases=[];observations=[obs.cpu().numpy()]
    for t in range(50):
        action=actor.actor_mean(obs).clamp(-1,1)
        check_budget(progress,deadline,64)
        obs,reward,terminated,truncated,info=env.step(action)
        if terminated.any() or (t<49 and truncated.any()) or (t==49 and not truncated.all()):
            raise RuntimeError('Diagnostic horizon changed')
        lifted=base.cubeA.pose.p[:,2]>=height+.03
        transported=lifted & ((distance-torch.linalg.vector_norm((base.cubeA.pose.p-base.cubeB.pose.p)[:,:2],dim=1))>=.03)
        phases.append(torch.stack([info['is_cubeA_grasped'],lifted,transported,
            info['is_cubeA_on_cubeB'],~info['is_cubeA_grasped'],info['is_cubeA_static'],info['success']],-1).cpu().numpy())
        actions.append(action.cpu().numpy());rewards.append(reward.cpu().numpy());observations.append(obs.cpu().numpy())
    rewards=np.stack(rewards);stages=np.stack(phases)
    returns=rewards.sum(0)
    success=info['success'].cpu().numpy()
    np.savez_compressed(out,observations=np.stack(observations),executed_actions=np.stack(actions),
        dense_rewards=rewards,stage_flags=stages,stock_final_success=success,dense_returns=returns,
        stage_names=np.array(['grasp','lift','transport','on_top','not_grasped','static','stock_success']))
    return dict(successes=int(success.sum()),episodes=64,mean_dense_return=float(returns.mean()),
                episode_successes=success.tolist(),episode_dense_returns=returns.tolist(),
                stage_completion_counts=dict(zip(['grasp','lift','transport','on_top','not_grasped','static','stock_success'],
                    stages.any(0).sum(0).astype(int).tolist())),
                release_definition='not_grasped is a stage signal, not proof of physical contact-free release')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--production',required=True);parser.add_argument('--initial',required=True)
    parser.add_argument('--bc-actor',required=True);parser.add_argument('--output',required=True);parser.add_argument('--deadline',required=True,type=float)
    a=parser.parse_args();started=time.monotonic();out=Path(a.output);out.mkdir(exist_ok=False)
    progress=dict(new_simulator_transitions=0,vector_step_calls=0)
    work=dict(actor_forward_rows=0,actor_forward_calls=0,critic_forward_rows=0,critic_forward_calls=0)
    report=dict(experiment=EXPERIMENT,state='started',progress=progress,collections=[],forward_work=work,
        initialized_policy='Exact archived pre-update assisted Agent; fresh Adam. Actor branches fix original critic/std.',
        benchmark_comparison=False,development_queries=0,confirmation_queries=0,automatic_followups=False,
        initialization_seed=1,collection_seeds=list(COLLECTION_SEEDS),probe_reset_seeds=list(PROBE_SEEDS),
        startup_difference='Original builtin400-step evaluation omitted. Fresh reset/sampling seeds explicitly chosen; not historical trajectory replay.',
        teacher_prefix_transitions=49971200,demo_transitions=25600,bc_optimizer_updates=5000,
        teacher_prefix_trainer_seconds=2295.408300588,initial_checkpoint_sha256=INITIAL_SHA)
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;write_json(out/'report.json',report)
    train=probe=None
    try:
        report['versions']={k:importlib.metadata.version(k) for k in PINS}
        if report['versions']!=PINS: raise RuntimeError('Production pins differ')
        report['gpu_actual_name']=torch.cuda.get_device_name(0)
        if report['gpu_actual_name']!='NVIDIA L40S': raise RuntimeError('Production GPU differs')
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=TASK_SOURCE_SHA:
            raise RuntimeError('Frozen task source differs')
        cls,gae,update=load_production(a.production)
        initial_raw=Path(a.initial).read_bytes()
        if hashlib.sha256(initial_raw).hexdigest()!=INITIAL_SHA: raise RuntimeError('Initial artifact differs')
        # Restricted isolated candidate-side artifact reader, never a trusted evaluator.
        state=torch.load(a.initial,map_location='cpu',weights_only=True)
        initial=new_agent(cls).to('cuda');initial.load_state_dict(state,strict=True)
        if not all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values()): raise ValueError('Invalid finite f32 state')
        manifest=json.loads((Path(a.bc_actor)/'manifest.json').read_bytes())
        weights=(Path(a.bc_actor)/'weights.json').read_bytes()
        if manifest['weights_sha256']!=BC_SHA or hashlib.sha256(weights).hexdigest()!=BC_SHA:raise ValueError('BC actor provenance differs')
        encoded=json.loads(weights)
        for i,layer in enumerate(x for x in initial.actor_mean if isinstance(x,torch.nn.Linear)):
            for key in ('weight','bias'):
                target=getattr(layer,key)
                entry=encoded[f'layers.{i}.{key}']
                assert_same(target,torch.tensor(entry['values'],device='cuda').reshape(entry['shape']),'historical BC actor '+key)
        args=ProbeArgs();report['actual_args']=asdict(args)
        train=make_env(1024,True,None);probe=make_env(64,False,1)
        persist()
        for seed,probe_seed in zip(COLLECTION_SEEDS,PROBE_SEEDS):
            folder=out/str(seed);folder.mkdir()
            agent=new_agent(cls,'cuda');agent.load_state_dict(initial.state_dict());agent.eval();instrument(agent,work)
            data,collection=collect(train,agent,seed,args,folder,progress,a.deadline)
            row=dict(seed=seed,collection=collection,state='collected');report['collections'].append(row);persist()
            end,batch,stats,parity=check_reconstruction(agent,args,data,gae,update)
            row['parity']=parity
            # NO branch execution or score until bitwise real-data parity passes.
            states,holdout=kl_holdout(batch,stats['mb_indices'],args)
            branches,fractions,grid=choose_branches(agent,end,states)
            row.update(fractions=fractions,kl_grid=grid,kl_holdout_episodes=holdout.tolist(),
                branch_kl={name:mean_kl(agent,branch,states) for name,branch in branches.items()},
                first_update={k:v for k,v in stats.items() if k!='mb_indices'},state='parity_passed',diagnostics={})
            with torch.no_grad():
                p=torch.distributions.Normal(agent.actor_mean(states),agent.actor_logstd.exp())
                q=torch.distributions.Normal(end.actor_mean(states),end.actor_logstd.exp())
                row['joint_update_exact_kl']=float(torch.distributions.kl_divergence(p,q).sum(-1).mean())
                row['mean_only_full_kl']=mean_kl(agent,branches['full'],states)
                row['original_std']=agent.actor_logstd.exp().flatten().tolist()
                row['updated_std']=end.actor_logstd.exp().flatten().tolist()
            np.savez_compressed(folder/'update-state.npz',
                **{'initial_'+k:v.detach().cpu().numpy() for k,v in agent.state_dict().items()},
                **{'joint_update_'+k:v.detach().cpu().numpy() for k,v in end.state_dict().items()},
                fitting_indices=stats['mb_indices'],advantages=batch['b_advantages'].cpu().numpy(),
                returns=batch['b_returns'].cpu().numpy(),kl_holdout_episodes=holdout)
            persist()
            seen={}
            for name,branch in branches.items():
                if fractions[name] in seen:
                    row['diagnostics'][name]={**row['diagnostics'][seen[fractions[name]]],
                        'reused_identical_branch':seen[fractions[name]]}
                else:
                    row['diagnostics'][name]=task_probe(probe,branch,probe_seed,folder/(name+'.npz'),progress,a.deadline)
                    seen[fractions[name]]=name
                persist()
            row['state']='complete';persist()
            del data,batch,branches,end,agent
        report['state']='complete'
        report['optimizer_calls']={'production_reference':3,'reconstruction':3,'total':6,'fitting_rows_total':9600}
    except Exception as exc:
        report.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
    finally:
        for env in (train,probe):
            if env is not None: env.close()
        persist()


if __name__=='__main__': main()
