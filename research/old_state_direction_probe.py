"""Trusted public direction assessment; accepts only safe actor JSON.

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


class RewardCapture:
    """Transparent step observer: one original physics call, no reward queries."""
    def __init__(self,base,on_step):self.base,self.on_step=base,on_step
    def __getattr__(self,name):return getattr(self.base,name)
    def step(self,actions):
        self.on_step('reserved',None)
        result=self.base.step(actions)
        self.on_step('completed',result[1])
        return result


def run(inputs,output):
    import torch
    from importlib.metadata import version
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy
    from automanip.stackcube import STACK_TASK,check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    from automanip.policy_io import PolicySchema

    inputs,output=Path(inputs),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    plan=json.loads((inputs/'plan.json').read_bytes());n=64
    record=dict(state='started',new_simulator_transitions_reserved=0,new_simulator_transitions_completed=0,
        vector_step_calls=0,reset_calls=0,policy_inference_rows=0,records=[],optimizer_calls=0,
        protected_benchmark_queries=0,confirmation_queries=0,final_queries=0,
        partition='public-diagnostic-not-learner-input',initial_observation_max_difference=0.)
    journal=(output/'work.jsonl').open('x',buffering=1)
    def persist():
        record['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    env=None
    try:
        record['versions']={key:version(key) for key in plan['production_pins']}
        if record['versions']!=plan['production_pins']:raise ValueError('Production pins differ')
        record['gpu']=torch.cuda.get_device_name(0)
        if record['gpu']!='NVIDIA L40S':raise ValueError('Registered hardware differs')
        record['historical_kernel_sha256']=check_frozen_protocol()
        schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        for path,digest in plan['policy_files'].items():
            if hashlib.sha256((inputs/path).read_bytes()).hexdigest()!=digest:raise ValueError('Safe policy file differs')
        class Observer(StackCubeAdapter):
            def __init__(self):
                super().__init__(n,'physx_cuda');self.env=RewardCapture(self.env,self.observe)
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
        for collection,seed_base in zip(plan['collection_seeds'],plan['diagnostic_seed_bases']):
            folder=output/str(collection);folder.mkdir();reference=[]
            for branch in plan['branch_order']:
                key='no_update' if branch=='no_update' else str(collection)+'/'+branch
                actor=load_policy(inputs/plan['policies'][key],schema)
                policy=NumpyPolicy(actor,schema);results=[];traces=[]
                for batch in range(8):
                    seeds=list(range(seed_base+batch*n,seed_base+(batch+1)*n))
                    # Check the reset before policy stepping, preserving the
                    # frozen evaluate_batch reset semantics via a one-shot shim.
                    original_reset=env.reset
                    def checked_reset(seeds,*,_reset=original_reset,_batch=batch):
                        obs=_reset(seeds);arr=np.asarray(obs,dtype=np.float32)
                        if branch=='no_update':reference.append(arr.copy())
                        else:
                            error=float(np.max(np.abs(arr-reference[_batch])))
                            record['initial_observation_max_difference']=max(record['initial_observation_max_difference'],error)
                            if error>1e-5:raise ValueError('Paired reset mismatch')
                        return obs
                    env.reset=checked_reset
                    try:rollouts=evaluate_batch(env,policy,seeds,horizon=50,action_dim=8)
                    finally:env.reset=original_reset
                    results.extend(rollouts)
                    if any(r.outcome.failed for r in rollouts):raise ValueError('Diagnostic execution failed, not a policy score')
                    traces.append(dict(observations=np.asarray(env.observations),executed_actions=np.asarray(env.actions),
                        dense_rewards=np.asarray(env.rewards),stage_flags=np.asarray(env.phases)))
                    persist()
                arrays={key:np.concatenate([t[key] for t in traces],axis=1) for key in traces[0]}
                returns=arrays['dense_rewards'].sum(0)
                np.savez_compressed(folder/(branch+'.npz'),**arrays,stock_final_success=np.asarray([r.outcome.success_at_end for r in results]))
                outcomes=[asdict(r) for r in results]
                (folder/(branch+'-outcomes.json')).write_text(json.dumps(outcomes,indent=2)+'\n')
                record['records'].append(dict(collection_seed=collection,diagnostic_seed_base=seed_base,
                    branch=branch,weights_sha256=actor.weights_sha256,episodes=512,successes=sum(r.outcome.success_at_end for r in results),
                    execution_failures=sum(r.outcome.failed for r in results),mean_dense_return=float(returns.mean()),
                    episode_dense_returns=returns.tolist(),episode_successes=[r.outcome.success_at_end for r in results]))
                persist()
        if record['new_simulator_transitions_completed']!=plan['max_public_transitions']:raise ValueError('Incomplete diagnostic')
        record['state']='complete'
    except Exception as exc:
        import traceback
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
        journal.close();persist()
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.inputs,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('records','traceback')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
