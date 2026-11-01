"""Registered sham-versus-impulse probe; no learning or final assessment.

New disturbance distribution only. Historical task/scorer files stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


PUSH_STEP=20
from research.recovery_impulse import apply_impulse,IMPULSE_NEWTON_SECONDS
SOURCE_WEIGHTS='ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16'


def direction(seed):
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid episode seed')
    return 1. if seed%2==0 else -1.


def impulse_vectors(seeds,enabled):
    return [[0.,IMPULSE_NEWTON_SECONDS*direction(seed) if enabled else 0.,0.] for seed in seeds]


def classify(sham_successes,push_successes,episodes=128,execution_failures=0):
    if any(type(v) is not int or v<0 for v in (sham_successes,push_successes,episodes,execution_failures)) or episodes!=128:
        raise ValueError('Exact registered count required')
    if max(sham_successes,push_successes)>episodes:raise ValueError('Invalid success count')
    if execution_failures:return 'invalid_execution'
    if sham_successes<116:return 'source_applicability_failed'
    if push_successes>=116:return 'zero_shot_coverage_no_learning_gap'
    if sham_successes-push_successes<13:return 'insufficient_material_gap'
    if push_successes<13:return 'large_deficit_check_feasibility_before_learning'
    return 'material_recovery_gap_adaptation_probe_eligible'


def run(output):
    import numpy as np
    import torch
    from importlib.metadata import version
    from automanip.cumulative import ROOT,SOURCE
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy
    from automanip.stackcube import check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter
    from research.discovery.amp_v1 import PRODUCTION_PINS

    output=Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    record=dict(state='started',training_updates=0,training_transitions=0,final_episodes=0,
                reserved_transitions=0,completed_transitions=0,conditions={})
    def persist():
        record['wall_seconds']=time.monotonic()-started
        tmp=output/'report.tmp';tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    persist()
    class Adapter(StackCubeAdapter):
        def __init__(self,enabled):
            self.enabled=enabled;self.attempted=0;self.completed=0;super().__init__(8,'physx_cuda')
        def reset(self,seeds):
            obs=super().reset(seeds);self.seeds=list(seeds);self.t=0
            self.trace=dict(observations=[np.asarray(obs,dtype=np.float32)],actions=[],success=[],flags=[],
                signals=[],b_positions=[self.env.cubeB.pose.p.detach().cpu().numpy().copy()],
                b_velocities=[self.env.cubeB.linear_velocity.detach().cpu().numpy().copy()],push_calls=0)
            return obs
        def step(self,actions):
            if self.t==PUSH_STEP:
                apply_impulse(self.env.cubeB,impulse_vectors(self.seeds,self.enabled))
                self.trace['push_calls']+=1
            self.attempted+=8;record['reserved_transitions']+=8
            obs,success,term,trunc=super().step(actions)
            self.completed+=8;record['completed_transitions']+=8;self.t+=1
            self.trace['observations'].append(np.asarray(obs,dtype=np.float32))
            self.trace['actions'].append(np.asarray(actions,dtype=np.float32))
            self.trace['success'].append(success);self.trace['flags'].append([term,trunc])
            signal=self.signals()
            self.trace['signals'].append(np.stack([signal[k].detach().cpu().numpy() for k in ('grasped','lifted','transported','on_top','static','contact_free')],axis=-1))
            self.trace['b_positions'].append(self.env.cubeB.pose.p.detach().cpu().numpy().copy())
            self.trace['b_velocities'].append(self.env.cubeB.linear_velocity.detach().cpu().numpy().copy())
            if self.t%10==0:persist()
            return obs,success,term,trunc
    env=None
    try:
        record['versions']={name:version(name) for name in PRODUCTION_PINS}
        if record['versions']!=PRODUCTION_PINS:raise ValueError('Production pins differ')
        record['gpu']=torch.cuda.get_device_name(0)
        if 'L40S' not in record['gpu']:raise ValueError('Registered hardware differs')
        record['historical_kernel_protocol_sha256']=check_frozen_protocol()
        actor=load_policy(ROOT/'runs/stackcube-bc-exploration-pilot-20261028/teacher-input',SOURCE.schema)
        if actor.weights_sha256!=SOURCE_WEIGHTS:raise ValueError('Exact source differs')
        policy=NumpyPolicy(actor,SOURCE.schema)
        # First validate the single impulse interface, not a magnitude sweep.
        calibration=[]
        for enabled in (False,True):
            env=Adapter(enabled);obs=env.reset(list(range(93459000,93459008)))
            if env.env.control_freq!=20 or env.env.sim_freq!=100:raise ValueError('Effective clocks differ')
            masses=env.env.cubeB.mass.cpu().tolist()
            for _ in range(50):env.step([[0.]*8 for _ in range(8)])
            arrays={k:np.asarray(v) for k,v in env.trace.items() if k!='push_calls'}
            np.savez_compressed(output/('calibration-push.npz' if enabled else 'calibration-sham.npz'),**arrays)
            displacement=np.linalg.norm(arrays['b_positions'][-1,:,:2]-arrays['b_positions'][PUSH_STEP,:,:2],axis=-1)
            valid=arrays['b_positions'][-1,:,2]>.005
            calibration.append(dict(enabled=enabled,mass_kg=masses,displacement_m=displacement.tolist(),on_table=valid.tolist(),impulse_calls=env.trace['push_calls']))
            env.close();env=None
        record['calibration']=calibration;persist()
        disp=np.asarray(calibration[1]['displacement_m'])
        if np.median(disp)<.005 or np.max(disp)>.35 or sum(calibration[1]['on_table'])<7:
            raise ValueError('Registered impulse fails physical validity gate; no policy probe and no amplitude search')
        all_arrays={};all_results={}
        for name,enabled in [('sham',False),('push',True)]:
            env=Adapter(enabled);traces=[];results=[]
            for start in range(93460000,93460128,8):
                batch=evaluate_batch(env,policy,list(range(start,start+8)),horizon=50,action_dim=8)
                results.extend(batch)
                if any(r.outcome.failed for r in batch):raise ValueError('Policy assessment execution failed')
                if env.trace['push_calls']!=1:raise ValueError('Disturbance call count differs')
                traces.append({k:(np.asarray(v).transpose(2,0,1) if k=='flags' else np.asarray(v).swapaxes(0,1)) for k,v in env.trace.items() if k!='push_calls'})
            arrays={k:np.concatenate([t[k] for t in traces],axis=0) for k in traces[0]}
            np.savez_compressed(output/(name+'-development.npz'),**arrays)
            from dataclasses import asdict
            (output/(name+'-outcomes.json')).write_text(json.dumps([asdict(r) for r in results],indent=2)+'\n')
            all_arrays[name]=arrays;all_results[name]=results
            record['conditions'][name]=dict(successes=sum(r.outcome.success_at_end for r in results),episodes=len(results),
                execution_failures=sum(r.outcome.failed for r in results),transitions=env.completed,
                destination_on_table_at_end=int(np.sum(arrays['b_positions'][:,-1,2]>.005)),
                stage_ever={key:float(arrays['signals'][:,:,j].any(axis=1).mean()) for j,key in enumerate(('grasp','lift','transport','placement','static','contact_free'))})
            env.close();env=None;persist()
        before=all_arrays['sham']['observations'][:,:PUSH_STEP+1]
        after=all_arrays['push']['observations'][:,:PUSH_STEP+1]
        record['prefix_max_abs_difference']=float(np.max(np.abs(before-after)))
        record['paired_causal_interpretation_valid']=record['prefix_max_abs_difference']<=1e-5
        p=all_arrays['push']['b_positions'];s=all_arrays['sham']['b_positions']
        record['median_push_sham_destination_difference_at_21_m']=float(np.median(np.linalg.norm(p[:,PUSH_STEP+1,:2]-s[:,PUSH_STEP+1,:2],axis=-1)))
        a=record['conditions'];record['decision']=classify(a['sham']['successes'],a['push']['successes'])
        if not record['paired_causal_interpretation_valid']:
            record['decision']='paired_prefix_gate_failed_diagnose_reproducibility'
        record['state']='complete'
    except Exception as exc:
        import traceback
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if env is not None:
            if hasattr(env,'trace'):
                arrays={k:np.asarray(v) for k,v in env.trace.items() if k!='push_calls'}
                np.savez_compressed(output/'partial-on-error.npz',**arrays)
                record['partial_trace']='partial-on-error.npz'
            env.close()
        persist()
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    report=run(p.parse_args().output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','calibration')}))
    raise SystemExit(0 if report['state']=='complete' else 1)
