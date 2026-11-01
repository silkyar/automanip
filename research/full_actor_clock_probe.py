"""Whole-trajectory raw Gaussian policy-gradient noise, including hidden layers.

Uses only existing training samples and data-only actors. No optimizer or robot
work. Does not reconstruct normalized/clipped PPO or its joint Adam step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Normal


def episode_gradients(actor,logstd,observations,actions,residuals):
    params=[*actor.parameters(),logstd]
    mean=actor(observations)
    logprob=Normal(mean,logstd.exp()).log_prob(actions).sum(-1)
    result={}
    for index,(name,residual) in enumerate(residuals.items()):
        loss=(logprob*residual.detach()).sum()
        gradient=torch.autograd.grad(loss,params,retain_graph=index<len(residuals)-1)
        result[name]=torch.cat([g.reshape(-1) for g in gradient]).detach().numpy()
    return result


class FoldCovariance:
    def __init__(self,width):
        self.total=np.zeros((2,width));self.square=np.zeros(2);self.count=np.zeros(2,dtype=int)
    def add(self,fold,gradient):
        if not np.isfinite(gradient).all():raise ValueError('Nonfinite gradient')
        self.total[fold]+=gradient;self.square[fold]+=np.dot(gradient,gradient);self.count[fold]+=1
    def trace(self):
        if (self.count<2).any():raise ValueError('At least2held episodes per fold required')
        terms=(self.square-np.sum(self.total**2,axis=1)/self.count)/(self.count-1)
        if (terms<-1e-8*np.maximum(self.square,1)).any():raise ValueError('Invalid covariance sufficient statistics')
        return float(np.maximum(terms,0).mean())


def run(root,output,plan):
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.stackcube import STACK_TASK
    from research.discovery.amp_v1 import read_actor
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False,parents=True);started=time.monotonic()
    torch.set_num_threads(2)
    result=dict(state='started',cells=[],new_environment_transitions=0,optimizer_calls=0,
                policy_forward_rows=0,backward_calls=0,CPU_versions=dict(torch=torch.__version__,numpy=np.__version__))
    def persist():
        result['wall_seconds']=time.monotonic()-started
        tmp=output/'report.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    try:
        for path,digest in plan['inputs'].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Source/input drift: '+path)
        clock=json.loads((root/plan['clock_report']).read_text());compute=json.loads((root/plan['compute']).read_text())
        if clock['state']!='complete':raise ValueError('Clock labels incomplete')
        for cell in plan['cells']:
            index,checkpoint=cell['index'],cell['checkpoint'];method=['ppo-gae','ppo-finite-mc'][index%2];seed=1+index//2
            folder=root/plan['training_root']/f'trial-{index}/work/runs/scratch-credit-{method}-seed-{seed}'
            path=folder/'actors'/str(checkpoint)
            load_policy(path,PolicySchema(STACK_TASK.fingerprint,48,8))
            actor,_=read_actor(path);actor=actor.double().eval().requires_grad_(True)
            with np.load(folder/f'trace-{checkpoint+51200}.npz',allow_pickle=False) as z:
                observations=torch.from_numpy(z['observations'].copy()).double();actions=torch.from_numpy(z['raw_actions'].copy()).double()
            scalars=compute['trials'][index]['scalars']
            logstd=torch.tensor([next(r['value'] for r in scalars[f'audit/logstd_dim_{a}'] if r['steps']==checkpoint) for a in range(8)],dtype=torch.float64,requires_grad=True)
            with np.load(root/plan['clock_predictions']/f'cell-{index}-{checkpoint}.npz',allow_pickle=False) as z:
                returns=z['returns'].copy().reshape(50,64)
                residuals={name:torch.from_numpy(returns-z[name].reshape(50,64)) for name in plan['models']}
            width=sum(p.numel() for p in actor.parameters())+8
            if width!=146192:raise ValueError('Actor parameter contract differs')
            statistics={name:FoldCovariance(width) for name in residuals}
            for lane in range(64):
                gradients=episode_gradients(actor,logstd,observations[:,lane],actions[:,lane],{name:value[:,lane] for name,value in residuals.items()})
                result['policy_forward_rows']+=50;result['backward_calls']+=len(gradients)
                for name,gradient in gradients.items():statistics[name].add(lane%2,gradient)
            variance={name:stat.trace() for name,stat in statistics.items()}
            row=dict(index=index,checkpoint=checkpoint,method=method,seed=seed,actor_parameters=width,
                     conditional_full_actor_gradient_covariance_trace=variance,
                     clock_to_state_ratio=variance['clock_actor_state']/variance['actor_state'],
                     clock_to_permuted_ratio=variance['clock_actor_state']/variance['permuted_clock_actor_state'],
                     clock_to_actual_value_ratio=variance['clock_actor_state']/variance['raw_value'])
            result['cells'].append(row);persist()
        result['registered_clock_expectation_met']=sum(c['clock_to_state_ratio']<=.75 and c['clock_to_permuted_ratio']<=.75 for c in result['cells'])>=9
        mc=[c for c in result['cells'] if c['method']=='ppo-finite-mc']
        result['material_gain_over_actual_MC_value']=sum(c['clock_to_actual_value_ratio']<=.75 for c in mc)>=4
        result['state']='complete'
        result['scope']='Conditional raw unnormalized mean+logstd policy score, all146192actorparameters and complete50-step sums. Excludes PPO clipping/normalization, critic/global clipping and Adam. Not a task-improvement or efficiency result.'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=run(a.root,a.output,json.loads(Path(a.plan).read_text()));print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}));raise SystemExit(0 if r['state']=='complete' else 1)
