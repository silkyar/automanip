"""Bounded from-scratch replay-moment study; no evaluator or cloud imports."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict,dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from replay_learner import Learner, Replay, transition_successor, export_actor
from joint_exploration import Collector

PINS={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'}
TASK_SHA='ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440'
POINTS=(51200,102400,256000,512000,1024000)


@dataclass
class Config:
    method:str='target-sac'
    seed:int=1
    num_envs:int=64
    total_transitions:int=1024000
    max_seconds:float=600.
    batch_size:int=2048
    collection_block:int=128
    critic_updates_per_block:int=8
    learning_starts:int=5000
    replay_capacity:int=1048576
    critic_width:int=256
    gamma:float=.99
    learning_rate:float=.001
    policy_delay:int=3

    def validate(self):
        if self.method not in ('target-sac','joint-moments','split-moments'):raise ValueError('Unregistered method')
        if self.num_envs!=64 or self.collection_block!=128 or self.critic_updates_per_block!=8 or self.batch_size!=2048:
            raise ValueError('Fixed vectorized replay budget required')
        if self.total_transitions not in (5120,1024000) or not 20<=self.max_seconds<=600:
            raise ValueError('Only preregistered smoke/scientific budgets')
        return self


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def train(config,output):
    config.validate();started=time.monotonic()
    deadline=min(started+config.max_seconds-15,float(os.environ['AUTOMANIP_TRAINING_DEADLINE']))
    if deadline<=started:raise RuntimeError('Original lease expired')
    output=Path(output);output.mkdir(exist_ok=False)
    versions={k:importlib.metadata.version(k) for k in PINS}
    if versions!=PINS or torch.cuda.get_device_name(0)!='NVIDIA L40S':raise RuntimeError('Production pins/GPU differ')
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.envs.tasks.tabletop import stack_cube
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=TASK_SHA:raise RuntimeError('Frozen task source differs')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic=True
    env=ManiSkillVectorEnv(gym.make('StackCube-v1',num_envs=64,robot_uids='panda_wristcam',obs_mode='state',
        reward_mode='normalized_dense',control_mode='pd_joint_delta_pos',sim_backend='physx_cuda',render_mode='rgb_array'),
        ignore_terminations=True,record_metrics=True)
    if env.single_observation_space.shape!=(48,) or env.single_action_space.shape!=(8,):raise ValueError('Fixed interface differs')
    # Explicit paired random initialization after simulator construction.
    torch.manual_seed(config.seed)
    learner=Learner(config.method,width=config.critic_width,gamma=config.gamma,learning_rate=config.learning_rate,
                    policy_delay=config.policy_delay,device='cuda')
    replay=Replay(config.replay_capacity,device='cuda')
    study_arm=os.environ['AUTOMANIP_FULL_CURVE_ARM']
    if study_arm not in ('baseline','coordinate'):raise ValueError('Unknown full-curve arm')
    collector=Collector('coordinate',config.seed,'cuda',epsilon=0. if study_arm=='baseline' else .1)
    count=dict(transitions=0,vector_step_calls=0,completed_episodes=0,explicit_reset_episodes=64,
        inference_seconds=0.,simulator_seconds=0.,replay_seconds=0.,learner_seconds=0.,export_seconds=0.,
        training_success_at_end_sum=0,initial_random_transitions=0)
    report=dict(state='started',config=asdict(config),versions=versions,gpu='NVIDIA L40S',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),counts=count,
        source_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('replay_learner.py'),Path(__file__).with_name('sac_baseline.py'),Path(__file__).with_name('joint_exploration.py'))},
        checkpoint_exports=[],optimizer_work=learner.work,new_teacher_transitions=0,demonstrations=0,
        pretrained_weights=False,planning_queries=0,extra_simulator_queries=0,evaluation_transitions=0,
        observation_dim=48,action_dim=8,horizon=50,task_source_sha256=TASK_SHA,
        algorithm='Freshseed2 ordinary low-alpha SAC vscoordinate mixture; same learner and dual600s/1.024M ceilings',
        collector_work=collector.work,behavior_mode=collector.mode,behavior_epsilon=collector.epsilon,study_arm=study_arm,
        proposal_logprob_note='Unmodified proposal density; not executed mixture-action density; never used in learner updates',
        warmup_note='BRN100000 critic updates; <=64000 updates here, so normalized arms remain in BN warmup.')
    atomic_json(output/'metadata.json',report)
    recent=deque(maxlen=50)
    def sync():torch.cuda.synchronize()
    def persist():
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'report.json',report)
    def save(label,trajectory=True):
        sync();mark=time.monotonic()
        artifact=export_actor(learner.actor,output/label)
        with torch.no_grad():
            test=torch.linspace(-1,1,48,device='cuda').reshape(1,-1).repeat(4,1)
            # Numeric safe export checked independently against the live squash.
            try:from sac_baseline import numerical_action,numerical_layers
            except ImportError:raise
            numerical=numerical_action(test,numerical_layers(learner.actor))
            if not torch.allclose(numerical,learner.actor.deterministic(test),atol=2e-6,rtol=2e-6):
                raise ValueError('Safe actor inference parity failure')
        values={}
        for prefix,module in [('actor',learner.actor),('q1',learner.q1),('q2',learner.q2),('target1',learner.target1),('target2',learner.target2)]:
            for k,v in module.state_dict().items():values[prefix+'.'+k]=v.detach().cpu().numpy()
        values['log_alpha']=learner.log_alpha.detach().cpu().numpy()
        values['torch_cpu_rng']=torch.get_rng_state().cpu().numpy()
        values['torch_cuda_rng']=torch.cuda.get_rng_state().cpu().numpy()
        np.savez_compressed(output/(label+'-state.npz'),**values)
        if trajectory and len(recent)==50:
            np.savez_compressed(output/(label+'-trajectory.npz'),
                **{key:np.stack([frame[key].detach().cpu().numpy() for frame in recent]) for key in recent[0]},
                note=np.array('Recent50 training steps; behavior policy can change within an episode. Not a frozen-policy rollout.'))
        count['export_seconds']+=time.monotonic()-mark
        report['checkpoint_exports'].append(dict(label=label,transitions=count['transitions'],
            elapsed_seconds=time.monotonic()-started,weights_sha256=artifact['weights_sha256'],
            critic_updates=learner.updates,actor_updates=learner.actor_updates))
        persist()
    try:
        obs,_=env.reset(seed=config.seed)
        save('actor-step-0',False)
        with (output/'learning.jsonl').open('x') as log:
            while count['transitions']<config.total_transitions and time.monotonic()<deadline:
                collected=0
                for _ in range(2):
                    if count['transitions']>=config.total_transitions or time.monotonic()>=deadline:break
                    sync();mark=time.monotonic()
                    with torch.no_grad():
                        if count['transitions']<config.learning_starts:
                            action=2*torch.rand(64,8,device='cuda')-1
                            logprob=torch.full((64,1),-8*np.log(2),device='cuda')
                            count['initial_random_transitions']+=64
                            proposal_action=action
                            component_mask=torch.ones_like(action,dtype=torch.bool)
                            proposal_is_uniform=True
                        else:
                            action,logprob,proposal_action,component_mask=collector.sample(learner.actor,obs)
                            proposal_is_uniform=False
                    sync();count['inference_seconds']+=time.monotonic()-mark
                    mark=time.monotonic()
                    following,reward,term,trunc,info=env.step(action)
                    sync();count['simulator_seconds']+=time.monotonic()-mark
                    actual,stop=transition_successor(following,term,trunc,info)
                    mark=time.monotonic();replay.add(obs,actual,action,reward,stop)
                    actual_info=info.get('final_info',info)
                    recent.append(dict(obs=obs.detach().clone(),next_obs=actual.detach().clone(),action=action.detach().clone(),
                        reward=reward.detach().clone(),stop=stop.detach().clone(),truncated=trunc.detach().clone(),
                        proposal_logprob=logprob.detach().clone(),proposal_action=proposal_action.detach().clone(),
                        uniform_component_mask=component_mask.detach().clone(),
                        proposal_is_warmup_uniform=torch.full((64,),proposal_is_uniform,dtype=torch.bool,device='cuda'),
                        stage_flags=torch.stack([actual_info[k].bool() for k in
                            ('is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success')],dim=-1).detach().clone()))
                    sync();count['replay_seconds']+=time.monotonic()-mark
                    obs=following;count['transitions']+=64;count['vector_step_calls']+=1;collected+=64
                    count['completed_episodes']+=int((term|trunc).sum())
                    if (term|trunc).any():count['training_success_at_end_sum']+=int(actual_info['success'].sum())
                stats={}
                if count['transitions']>=config.learning_starts:
                    sync();mark=time.monotonic()
                    for _ in range(collected//16):
                        if time.monotonic()>=deadline:break
                        stats=learner.update(replay.sample(config.batch_size))
                    sync();count['learner_seconds']+=time.monotonic()-mark
                    if stats and not bool(torch.stack(list(stats.values())).isfinite().all()):raise ValueError('Nonfinite learner statistic')
                if count['transitions'] in POINTS:
                    save('actor-step-'+str(count['transitions']))
                if count['transitions']%5120==0:
                    row=dict(transitions=count['transitions'],elapsed_seconds=time.monotonic()-started,
                        critic_updates=learner.updates,actor_updates=learner.actor_updates,
                        metrics={k:float(v) for k,v in stats.items()},counts=count.copy(),
                        collector_work=collector.work.copy())
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();persist()
            save('actor-final')
        report.update(state='complete',stopped_by='steps' if count['transitions']==config.total_transitions else 'deadline',
            replay_size=replay.size,critic_updates=learner.updates,actor_updates=learner.actor_updates,
            expected_critic_updates_without_deadline=max(0,(count['transitions']-4992)//16),
            normalization_steps=[int(m.steps) for m in learner.q1.modules() if hasattr(m,'steps')])
        persist();return report
    except Exception as exc:
        report.update(state='failed',error=type(exc).__name__+': '+str(exc));persist();raise
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--total-transitions',type=int,default=1024000);p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--output',required=True);a=p.parse_args()
    r=train(Config(method=a.method,seed=a.seed,total_transitions=a.total_transitions,max_seconds=a.max_seconds),a.output)
    print(json.dumps({'state':r['state'],'transitions':r['counts']['transitions'],'elapsed_seconds':r['elapsed_seconds']}))
