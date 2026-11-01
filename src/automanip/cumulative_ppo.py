"""Thin actor-only PPO adaptation using existing PPO math and episode storage.

Training code only. Run against an explicitly supplied compatible environment;
this module never constructs a simulator or launches a service. No full learner
resume or cross-task replay is supported. Evaluation consumes actor JSON only.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.distributions import Normal

import train as ppo
from research.discovery.amp_v1 import StudentActor, export_actor
from automanip.cumulative import ROOT, SOURCE, compatible, cost, lifetime_cost, register_asset, write_json, relative_path, directory_digest
from automanip.cumulative_episodes import EpisodeRecorder
from automanip.policy_io import load_policy


@dataclass(frozen=True)
class Budget:
    training_interactions: int
    optimizer_calls: int
    seconds: float
    development_interactions: int = 64000

    def validate(self):
        if any(type(v) is not int or v < 0 for v in (self.training_interactions,self.optimizer_calls,self.development_interactions)):
            raise ValueError('Interaction/update ceilings must be nonnegative integers')
        if type(self.seconds) not in (int,float) or not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError('Time ceiling must be finite and nonnegative')
        return self


def default_config(seed=41):
    return ppo.PPOConfig(seed=seed,num_envs=64,num_steps=50,total_timesteps=2048000,
        max_seconds=600.,gamma=.9,ent_coef=.003,hidden_sizes=(256,256,256),
        update_epochs=8,num_minibatches=32,partial_reset=False,bootstrap_on_termination=False)


class AdaptationAgent(nn.Module):
    def __init__(self, config, artifact=None, initial_logstd=-.5):
        super().__init__()
        if not math.isfinite(initial_logstd) or not -8 <= initial_logstd <= 2:
            raise ValueError('Invalid fresh exploration scale')
        self.actor = StudentActor(tuple(config.hidden_sizes))
        self.actor.net = ppo.mlp(48,8,config.hidden_sizes,.01*math.sqrt(2))
        self.critic = ppo.mlp(48,1,config.hidden_sizes,1.)
        self.logstd = nn.Parameter(torch.full((1,8),initial_logstd))
        if artifact is not None:
            if tuple(config.hidden_sizes) != artifact.hidden_sizes:
                raise ValueError('Warm-start architecture differs; no implicit conversion')
            with torch.no_grad():
                for i,layer in enumerate(m for m in self.actor.net if isinstance(m,nn.Linear)):
                    for name in ('weight','bias'):
                        tensor=getattr(layer,name)
                        tensor.copy_(torch.tensor(artifact.tensors[f'layers.{i}.{name}'],dtype=torch.float32).reshape(tensor.shape))

    def action_and_value(self, observations, actions=None):
        distribution=Normal(self.actor(observations),self.logstd.exp())
        if actions is None:actions=distribution.sample()
        return actions,distribution.log_prob(actions).sum(-1),distribution.entropy().sum(-1),self.critic(observations).squeeze(-1)


def state_digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Exhausted(RuntimeError):pass


class CountedOptimizer:
    def __init__(self, optimizer, record, budget, deadline, clock, event):
        self.optimizer,self.record,self.budget,self.deadline,self.clock=optimizer,record,budget,deadline,clock
        self.param_groups=optimizer.param_groups;self.event=event
    def zero_grad(self, **kwargs):return self.optimizer.zero_grad(**kwargs)
    def step(self):
        if self.record['optimizer_calls_attempted'] >= self.budget.optimizer_calls or self.clock() >= self.deadline:
            raise Exhausted('update_or_time_budget')
        self.record['optimizer_calls_attempted']+=1
        self.event('optimizer_reserved',self.record['optimizer_calls_attempted'])
        self.optimizer.step()
        self.record['optimizer_calls_completed']+=1
        self.event('optimizer_completed',self.record['optimizer_calls_completed'])


def _adapt_impl(target, source, config, budget, env, output, *, _close, root=ROOT, environment_seed=93440000,
          auto_reset=False, device='cpu', initial_logstd=-.5, execution_kind='synthetic',
          feedback=None, feedback_at=(0,128000,512000,1024000,2048000), feedback_reserve=12800,
          clock=time.perf_counter):
    """Take ownership of env, adapt once, persist partial/error accounting.

    `feedback` is trusted orchestration accepting a safe policy directory and
    step count, returning development-only assessment records. Its reservation
    remains charged on exceptions; it is never called with final assessments.
    Inner wall time includes lineage lookup, initialization, resets, storage and feedback.
    Caller asset loading/imports/environment construction precede this function;
    a production outer clock must include those too.
    Native tensor/file operations are cooperative; a future production runner
    must externally enforce the hard process window. No resume claim is made.
    """
    started=clock()
    try:
        budget.validate();config.validate();target.require_stack()
        if source is not None:compatible(source,target)
        if getattr(env,'task_id',None)!=target.task_id or getattr(env,'interface_id',None)!=target.interface.fingerprint:
            raise ValueError('Environment contract differs before collection')
        if budget.training_interactions>config.total_timesteps:raise ValueError('Interaction budget exceeds declared PPO configuration')
        if budget.training_interactions % config.num_envs:
            raise ValueError('Training budget must contain whole vector steps')
        if execution_kind not in ('synthetic','authorized_simulator'):
            raise ValueError('Declare execution provenance')
        if type(feedback_reserve) is not int or feedback_reserve < 0:
            raise ValueError('Invalid development reservation')
    except Exception:
        try:_close()
        except Exception:pass
        raise
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=started+budget.seconds
    record=dict(format='automanip-adaptation-attempt-v1',status='running',task=target.to_dict(),
        task_id=target.task_id,interface_id=target.interface.fingerprint,execution_kind=execution_kind,
        wall_clock_scope='Inner adapt call; caller imports/source parsing/environment construction excluded; not an end-to-end minute claim',
        config=asdict(config),budget=asdict(budget),parent_assets=[] if source is None else [source.asset_id],
        environment_transitions_attempted=0,environment_transitions_confirmed=0,
        optimizer_calls_attempted=0,optimizer_calls_completed=0,development_interactions_charged=0,
        development_interactions_confirmed=0,explicit_reset_calls=0,explicit_reset_attempts=0,explicit_reset_episodes=0,
        policy_inference_rows=0,rollout_updates=0,feedback=[],episode_stores=[],errors=[],asset=None,
        initialization=dict(actor='fresh_random' if source is None else 'validated_source_actor',critic='fresh',
            optimizer='fresh Adam',exploration=f'fresh Gaussian logstd={initial_logstd}',normalizers='none',
            replay='none; cross-task replay prohibited',random_state=f'fresh seed{config.seed}',
            simulator='fresh explicit episode; not exact simulator resume',resume_supported=False),
        source_acquisition=None if source is None else lifetime_cost(source,root=root))
    recorder=None;agent=None;result=None;pending_reset=False
    journal=(output/'work.jsonl').open('x',buffering=1)
    def event(kind,count):journal.write(json.dumps(dict(event=kind,count=count),allow_nan=False)+'\n')
    def persist():
        record['wall_seconds']=clock()-started
        temporary=output/'attempt.tmp'
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        temporary.replace(output/'attempt.json')
    def make_recorder():
        path=output/'episodes'/f'shard-{len(record["episode_stores"]):05d}'
        writer=EpisodeRecorder(path,target.task_id,target.interface.fingerprint,'training',
            dict(parent_assets=record['parent_assets'],attempt=str(output),policy_revision=record['optimizer_calls_completed']),
            config.num_envs,auto_reset=auto_reset)
        record['episode_stores'].append(str(path))
        return writer
    def assess(step):
        if feedback is None or step not in feedback_at:return
        if record['development_interactions_charged']+feedback_reserve>budget.development_interactions or clock()>=deadline:
            raise Exhausted('development_or_time_budget')
        directory=output/'development-controllers'/f'step-{step}'
        artifact=export_actor(agent.actor,directory);load_policy(directory,target.schema)
        record['development_interactions_charged']+=feedback_reserve;event('feedback_reserved',record['development_interactions_charged']);persist()
        reports=feedback(directory,step)
        if type(reports) is not list:raise ValueError('Feedback must return development reports')
        consumed=0
        for report in reports:
            value=report['environment_transitions']
            if type(value) is not int or value<0:raise ValueError('Invalid feedback cost')
            consumed+=value
        record['development_interactions_confirmed']+=consumed
        record['development_interactions_charged']+=max(0,consumed-feedback_reserve)
        event('feedback_confirmed',record['development_interactions_confirmed']);persist()
        if consumed>feedback_reserve:raise ValueError('Development exceeded its declared reservation')
        if {r.get('task_id') for r in reports}!={SOURCE.task_id,target.task_id} or len(reports)!=len({SOURCE.task_id,target.task_id}):
            raise ValueError('Feedback must separately assess source retention and target domain')
        for report in reports:
            if report['partition']!='development' or report['weights_sha256']!=artifact['weights_sha256']:
                raise ValueError('Final/foreign-controller assessment cannot guide adaptation')
        record['development_interactions_charged']-=feedback_reserve-consumed
        event('feedback_charged',record['development_interactions_charged'])
        record['feedback'].append(dict(training_step=step,reports=reports));persist()
        if clock()>=deadline:raise Exhausted('time_budget_after_feedback')
    persist()
    try:
        if clock()>=deadline:raise Exhausted('time_budget_before_initialization')
        if device!='cpu' and execution_kind!='authorized_simulator':raise ValueError('Synthetic checks are CPU-only')
        devices=[] if device=='cpu' else [torch.device(device).index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed)
            agent=AdaptationAgent(config,None if source is None else source.policy,initial_logstd).to(device)
            raw_optimizer=torch.optim.Adam(agent.parameters(),lr=config.learning_rate,eps=1e-5)
            record['initialization'].update(critic_initial_sha256=state_digest(agent.critic),optimizer_initial_state_entries=len(raw_optimizer.state),actor_initial_sha256=state_digest(agent.actor))
            write_json(output/'initialization.json',record['initialization'])
            optimizer=CountedOptimizer(raw_optimizer,record,budget,deadline,clock,event)
            assess(0)
            if clock()>=deadline:raise Exhausted('time_budget_before_reset')
            record['explicit_reset_attempts']+=1;event('reset_reserved',record['explicit_reset_attempts'])
            observations,_=env.reset(seed=environment_seed)
            event('reset_completed',record['explicit_reset_attempts'])
            record['explicit_reset_calls']+=1;record['explicit_reset_episodes']+=config.num_envs
            recorder=make_recorder();recorder.start(observations,environment_seed)
            while record['environment_transitions_attempted']<budget.training_interactions:
                if clock()>=deadline:raise Exhausted('time_budget')
                if record['optimizer_calls_attempted']>=budget.optimizer_calls:raise Exhausted('update_budget')
                rollout=[]
                for _ in range(config.num_steps):
                    if record['environment_transitions_attempted']+config.num_envs>budget.training_interactions or clock()>=deadline:break
                    if pending_reset:
                        if recorder.saved_count>=2048:
                            recorder.finish('shard_boundary');recorder=make_recorder()
                        next_seed=environment_seed+record['explicit_reset_calls']
                        record['explicit_reset_attempts']+=1;event('reset_reserved',record['explicit_reset_attempts'])
                        observations,_=env.reset(seed=next_seed)
                        record['explicit_reset_calls']+=1;record['explicit_reset_episodes']+=config.num_envs
                        event('reset_completed',record['explicit_reset_calls'])
                        recorder.start(observations,next_seed);pending_reset=False
                    with torch.no_grad():
                        raw,lp,_,value=agent.action_and_value(observations.to(device))
                    record['policy_inference_rows']+=config.num_envs
                    if raw.shape!=(config.num_envs,8) or not torch.isfinite(raw).all():raise ValueError('Invalid policy output')
                    executed=raw.clamp(-1,1)
                    before=observations
                    record['environment_transitions_attempted']+=config.num_envs
                    event('environment_reserved',record['environment_transitions_attempted'])
                    following,reward,terminated,truncated,info=env.step(executed)
                    record['environment_transitions_confirmed']+=config.num_envs
                    event('environment_completed',record['environment_transitions_confirmed'])
                    recorder.step(before,executed,following,reward,terminated,truncated,info,
                        behavior=dict(parent_assets=record['parent_assets'],policy_revision=record['optimizer_calls_completed'],action_labels='executed clipped Gaussian proposals; raw likelihood retained only in on-policy PPO batch'))
                    done=terminated|truncated
                    boot_info=info if auto_reset else dict(final_observation=following,_final_observation=done)
                    with torch.no_grad():bootstrap=ppo.successor_values(agent,following.to(device),terminated.to(device),truncated.to(device),boot_info,config.bootstrap_on_termination)
                    if not torch.isfinite(reward).all() or not torch.isfinite(bootstrap).all():raise ValueError('Invalid rollout')
                    rollout.append((before.to(device),raw,lp,reward.to(device),value,bootstrap,done.to(device)))
                    observations=following
                    if not auto_reset and done.any():
                        if not bool(done.all()):raise ValueError('This thin non-autoreset runner requires synchronous episode boundaries')
                        pending_reset=True
                if not rollout:raise Exhausted('interaction_or_time_budget')
                obs,actions,lp,rewards,values,next_values,dones=[torch.stack(x) for x in zip(*rollout)]
                advantages,returns=ppo.generalized_advantage(rewards,values,next_values,dones,config.gamma,config.gae_lambda)
                batch=(obs.flatten(0,1),actions.flatten(0,1),lp.flatten(),advantages.flatten(),returns.flatten())
                ppo.ppo_update(agent,optimizer,batch,config,deadline)
                record['rollout_updates']+=1;persist()
                assess(record['environment_transitions_confirmed'])
            record['status']='training_interaction_ceiling'
    except Exhausted as exc:
        record.update(status='censored',censor_reason=str(exc))
    except Exception as exc:
        record['status']='failed';record['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        if recorder is not None:
            try:recorder.finish(record['status'])
            except Exception as exc:
                record['status']='failed';record['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        try:_close()
        except Exception as exc:
            record['status']='failed';record['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        if agent is not None and record['status']!='failed':
            try:
                policy_dir=output/'policy';artifact=export_actor(agent.actor,policy_dir);load_policy(policy_dir,target.schema)
                acquisition=cost(record['environment_transitions_attempted']+record['development_interactions_charged'],clock()-started,0. if execution_kind=='synthetic' else None)
                result=register_asset(output/'asset.json',policy_dir,target,acquisition,root=root,parents=() if source is None else (source,),
                    data_assets=[dict(path=relative_path(p,root),partition='training',task_id=target.task_id,sha256=directory_digest(p)) for p in record['episode_stores']])
                record['asset']=dict(asset_id=result.asset_id,manifest=str(result.manifest_path),weights_sha256=result.policy.weights_sha256)
            except Exception as exc:
                record['status']='failed';record['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        record['wall_seconds']=clock()-started
        record['wall_budget_overrun_seconds']=max(0.,record['wall_seconds']-budget.seconds)
        if record['wall_budget_overrun_seconds']>0 and record['status']!='failed':record['status']='censored'
        record['goal_state_refresh_calls']=getattr(env,'goal_state_refresh_calls',0)
        record['goal_gpu_sync_calls']=getattr(env,'goal_gpu_sync_calls',0)
        record['reset_refresh_accounting']='Each north refresh adds get_obs/evaluate; GPU adds one rigid apply/fetch, no physics step. Caller constructor/reset work is outside inner clock.'
        record['optimizer_calls_uncertain']=record['optimizer_calls_attempted']-record['optimizer_calls_completed']
        record['environment_transitions_uncertain']=record['environment_transitions_attempted']-record['environment_transitions_confirmed']
        journal.close()
        record['retention']='unassessed; requires same adapted-controller source-task outcomes'
        persist()
    return record,result


def adapt(target, source, config, budget, env, output, **kwargs):
    """Own and close env exactly once, including rejection/setup-I/O failures.

    See _adapt_impl for actor-only initialization, budgets and clock semantics.
    """
    closed=False
    def close():
        nonlocal closed
        if not closed:
            closed=True
            return env.close()
    try:
        return _adapt_impl(target,source,config,budget,env,output,_close=close,**kwargs)
    finally:
        close()


def recover_work(path):
    """Read flushed pre-call reservations after an interrupted attempt.

    Uncompleted reservations remain charged as uncertain work, not completed
    updates. This does not recover a simulator or make power-loss guarantees.
    """
    path=Path(path)
    if path.stat().st_size>64*1024*1024:raise ValueError('Journal byte bound')
    counters={name:0 for name in ('environment_reserved','environment_completed','optimizer_reserved','optimizer_completed','reset_reserved','reset_completed','feedback_confirmed')}
    charged=0;truncated=False
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                truncated=True;break
            item=json.loads(line)
            if set(item)!={'event','count'} or type(item['count']) is not int or item['count']<0:raise ValueError('Invalid work journal')
            kind,value=item['event'],item['count']
            if kind in ('feedback_reserved','feedback_charged'):charged=value
            elif kind in counters:
                if value<counters[kind]:raise ValueError('Nonmonotone work counter')
                counters[kind]=value
            else:raise ValueError('Unknown work event')
    for prefix in ('environment','optimizer','reset'):
        if counters[prefix+'_completed']>counters[prefix+'_reserved']:raise ValueError('Unreserved work completion')
    charged=max(charged,counters['feedback_confirmed'])
    return dict(counters,development_interactions_charged=charged,
        optimizer_calls_uncertain=counters['optimizer_reserved']-counters['optimizer_completed'],
        environment_transitions_uncertain=counters['environment_reserved']-counters['environment_completed'],
        truncated_last_event=truncated,scope='Process-interruption accounting only; no exact learner/simulator resume')
