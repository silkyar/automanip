"""Randomized2x2post-opening maintenance; manual diagnostic, not a learned policy.

No learner, critic, native checkpoint, old trajectories or final seeds are loaded.
Reuses the unchanged evaluator kernel, physical success and actual task reward.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


from research.release_maintenance_policy import ReleaseMaintenancePolicy,BRANCHES


class RewardCapture:
    """Transparent step observer: one original physics call, no reward queries."""
    def __init__(self,base,on_step):self.base,self.on_step=base,on_step
    def __getattr__(self,name):return getattr(self.base,name)
    def step(self,actions):
        self.on_step('reserved',None)
        result=self.base.step(actions)
        self.on_step('completed',result[1])
        return result


def run(inputs,output,collection_index):
    import torch
    from importlib.metadata import version
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy
    from automanip.stackcube import STACK_TASK,STACK_TASK_SOURCE_SHA256,check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    from automanip.policy_io import PolicySchema

    inputs,output=Path(inputs),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    plan=json.loads((inputs/'plan.json').read_bytes());n=64
    if type(collection_index) is not int or collection_index not in (0,1):raise ValueError('Closed collection index')
    plan=dict(plan,collection_seeds=[plan['collection_seeds'][collection_index]],
              diagnostic_seed_bases=[plan['diagnostic_seed_bases'][collection_index]],
              noise_seed_bases=[plan['noise_seed_bases'][collection_index]],
              max_public_transitions=plan['max_public_transitions_per_collection'])
    record=dict(state='started',new_simulator_transitions_reserved=0,new_simulator_transitions_completed=0,
        vector_step_calls=0,reset_calls=0,policy_inference_rows=0,records=[],optimizer_calls=0,
        protected_benchmark_queries=0,confirmation_queries=0,final_queries=0,
        partition='public-diagnostic-not-learner-input',initial_observation_max_difference=0.)
    journal=(output/'work.jsonl').open('x',buffering=1)
    def persist():
        record['wall_seconds']=time.monotonic()-started
        temporary=output/'report.json.tmp'
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(output/'report.json')
    env=None
    try:
        record['versions']={key:version(key) for key in plan['production_pins']}
        if record['versions']!=plan['production_pins']:raise ValueError('Production pins differ')
        record['gpu']=torch.cuda.get_device_name(0)
        if record['gpu']!='NVIDIA L40S':raise ValueError('Registered hardware differs')
        record['historical_kernel_sha256']=check_frozen_protocol()
        blob=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        record['physx_sha256']=hashlib.sha256(blob.read_bytes()).hexdigest()
        if record['physx_sha256']!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':raise ValueError('PhysX pin differs')
        schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        for path,digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/path).read_bytes()).hexdigest()!=digest:raise ValueError('Safe policy file differs')
        class Observer(StackCubeAdapter):
            def __init__(self):
                import gymnasium as gym
                import mani_skill.envs
                from mani_skill.envs.tasks.tabletop import stack_cube
                if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=STACK_TASK_SOURCE_SHA256:raise ValueError('Task source pin differs')
                self.torch=torch;self.num_envs=n
                kwargs=STACK_TASK.env_kwargs(num_envs=n,backend='physx_cuda');kwargs['reward_mode']='normalized_dense'
                # The frozen scoring adapter uses reward_mode=none. This NEW
                # public diagnostic enables only the unchanged official reward.
                self.owner=gym.make(STACK_TASK.env_id,reconfiguration_freq=1,**kwargs);self.env=self.owner.unwrapped
                if self.env.single_observation_space.shape!=(48,) or self.env.single_action_space.shape!=(8,) or not (self.env.single_action_space.low==-1).all() or not (self.env.single_action_space.high==1).all():
                    self.owner.close();raise ValueError('Spaces/action semantics differ')
                self.schema=schema
                if self.env.reward_mode!='normalized_dense':raise ValueError('Official dense reward disabled')
                self.env=RewardCapture(self.env,self.observe)
            def observe(self,event,reward):
                if event=='reserved':
                    if record['new_simulator_transitions_reserved']+n>plan['max_public_transitions']:raise ValueError('Transition cap')
                    record['new_simulator_transitions_reserved']+=n;record['vector_step_calls']+=1
                    journal.write(json.dumps(dict(event='environment_reserved',count=record['new_simulator_transitions_reserved']))+'\n')
                else:
                    record['new_simulator_transitions_completed']+=n
                    journal.write(json.dumps(dict(event='environment_completed',count=record['new_simulator_transitions_completed']))+'\n')
                    self.rewards.append(reward.detach().cpu().numpy().copy())
            def signals(self):
                self.last_signals=super().signals();return self.last_signals
            def reset(self,seeds):
                self.rewards=[];self.actions=[];self.phases=[]
                record['reset_calls']+=1;obs=super().reset(seeds);self.observations=[np.asarray(obs,dtype=np.float32)]
                journal.write(json.dumps(dict(event='reset',count=record['reset_calls']))+'\n')
                return obs
            def step(self,actions):
                record['policy_inference_rows']+=len(actions)
                result=super().step(actions)
                self.actions.append(np.asarray(actions,dtype=np.float32))
                self.observations.append(np.asarray(result[0],dtype=np.float32))
                signal=self.last_signals
                self.phases.append(np.stack([signal[k].detach().cpu().numpy() for k in ('grasped','lifted','transported','on_top','static','contact_free')]+[np.asarray(result[1])],axis=-1))
                return result
        env=Observer()
        if env.env.control_freq!=20 or env.env.sim_freq!=100:raise ValueError('Effective clocks differ')
        collection=plan['collection_seeds'][0];seed_base=plan['diagnostic_seed_bases'][0]
        actor=load_policy(inputs/plan['policies'][str(collection)],schema)
        rows={branch:dict(collection_seed=collection,branch=branch,episodes=0,successes=0,execution_failures=0,episode_successes=[],batch_success_rates=[],batch_stage_counts=[],fallback_count=0,trigger_counts={}) for branch in BRANCHES}
        for batch in range(32):
            assignment=np.random.default_rng(plan['assignment_seed_bases'][collection_index]+batch).permutation(np.tile(np.arange(4),16))
            policy=ReleaseMaintenancePolicy(NumpyPolicy(actor,schema),plan['gaussian_std'][str(collection)],assignment)
            policy.reset_noise(plan['noise_seed_bases'][0]+batch)
            seeds=list(range(seed_base+batch*n,seed_base+(batch+1)*n))
            rollouts=evaluate_batch(env,policy,seeds,horizon=50,action_dim=8)
            trace=dict(observations=np.asarray(env.observations),executed_actions=np.asarray(env.actions),dense_rewards=np.asarray(env.rewards),stage_flags=np.asarray(env.phases),override_masks=np.asarray(policy.override_masks),arm_masks=np.asarray(policy.arm_masks),maintenance_counts=policy.maintenance_counts,assignment=assignment,trigger=policy.trigger,applied=policy.applied,first_eligible=policy.first_eligible,fallback=policy.fallback)
            # Preserve the failed batch before enforcing validity: the frozen
            # evaluator deliberately returns only sanitized exception classes.
            np.savez_compressed(output/f'batch-{batch}.npz',**trace)
            if any(r.outcome.failed for r in rollouts):
                record['failed_batch']=dict(batch=batch,steps=[r.steps for r in rollouts],errors=[r.error for r in rollouts],policy_step=policy.step_index,command_counts=policy.applied.tolist())
                raise ValueError('Diagnostic execution failed; partial trace retained')
            if not np.all(policy.applied==5):raise ValueError('Intervention command budget differs')
            if trace['observations'].shape!=(51,64,48) or trace['stage_flags'].shape!=(50,64,7):raise ValueError('Trace layout differs')
            flags=trace['stage_flags'].astype(bool)
            for group,branch in enumerate(BRANCHES):
                selected=np.flatnonzero(assignment==group);row=rows[branch]
                successes=[bool(rollouts[i].outcome.success_at_end) for i in selected]
                row['episode_successes'].extend(successes);row['episodes']+=16;row['successes']+=sum(successes)
                row['batch_success_rates'].append(sum(successes)/16)
                row['batch_stage_counts'].append({k:int(flags[:,selected,j].any(0).sum()) for j,k in enumerate(['grasp','lift','transport','on_top','static','contact_free','success'])})
                row['fallback_count']+=int(policy.fallback[selected].sum())
                for trigger in policy.trigger[selected]:row['trigger_counts'][str(trigger)]=row['trigger_counts'].get(str(trigger),0)+1
            record['records']=list(rows.values());persist()
        if record['new_simulator_transitions_completed']!=plan['max_public_transitions']:raise ValueError('Incomplete diagnostic')
        record['scope']='Manual action interventions on frozen failed training policies; no learned-policy improvement or sample-efficiency claim'
        record['state']='complete'
    except Exception as exc:
        import traceback
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
        journal.close();persist()
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--collection-index',type=int,choices=(0,1),required=True)
    a=p.parse_args();r=run(a.inputs,a.output,a.collection_index)
    print(json.dumps({k:v for k,v in r.items() if k not in ('records','traceback')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
