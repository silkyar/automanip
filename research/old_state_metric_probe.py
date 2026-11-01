"""Closed-form readout geometry: current KL versus old-state KL.

An analytical feasibility probe on archived experience, not a full PPO procedure
or a measured policy improvement. Fixed Gaussian scale, actor features/critic.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.discovery.first_update_probe import ProbeArgs,load_production,new_agent,kl_holdout


def constrained_direction(gradient,current,old,old_weight,kappa,ridge):
    metric=current+old_weight*old+ridge*torch.eye(len(current),dtype=current.dtype,device=current.device)
    direction=torch.linalg.solve(metric,gradient.T).T
    energy=(direction*(direction@current)).sum()
    if not torch.isfinite(energy) or energy<=0:raise ValueError('No finite positive direction')
    direction=direction*torch.sqrt(torch.as_tensor(2*kappa,dtype=energy.dtype)/energy)
    return direction,dict(current_KL=float(.5*(direction*(direction@current)).sum()),
        old_KL=float(.5*(direction*(direction@old)).sum()),linear_gain=float((gradient*direction).sum()))


@torch.no_grad()
def features(agent,obs):
    values=obs
    for layer in list(agent.actor_mean.children())[:-1]:values=layer(values)
    return torch.cat((values.double(),torch.ones((len(values),1),dtype=torch.float64)),dim=1)


@torch.no_grad()
def with_direction(initial,direction):
    branch=copy.deepcopy(initial);std=initial.actor_logstd.exp().flatten().double()
    branch.actor_mean[-1].weight.add_((std[:,None]*direction[:,:-1]).float())
    branch.actor_mean[-1].bias.add_((std*direction[:,-1]).float())
    return branch


@torch.no_grad()
def exact_KL(initial,branch,obs):
    value=.5*((branch.actor_mean(obs)-initial.actor_mean(obs))/initial.actor_logstd.exp()).square().sum(-1)
    return dict(mean=float(value.mean()),p95=float(torch.quantile(value,.95)),maximum=float(value.max()))


@torch.no_grad()
def clipped_surrogate(initial,branch,obs,actions,advantages):
    std=initial.actor_logstd.exp()
    old=torch.distributions.Normal(initial.actor_mean(obs),std).log_prob(actions).sum(-1)
    new=torch.distributions.Normal(branch.actor_mean(obs),std).log_prob(actions).sum(-1)
    ratio=(new-old).exp()
    return float((torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages)-advantages).mean())


def run(root,output,plan):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(state='started',rows=[],new_environment_transitions=0,gpu_work=0,optimizer_calls=0,
        linear_solves=0,scope='Mean-policy geometry and sample surrogate only, no task-performance evidence')
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for p,sha in plan['inputs'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Registered source/data changed')
        torch.set_num_threads(2);torch.manual_seed(93491000)
        cls,_,_=load_production(root/plan['production']);args=ProbeArgs()
        episode_ids=json.loads((root/plan['bc_report']).read_text())['train_episode_indices'][:64]
        with np.load(root/plan['prior_demonstrations'],allow_pickle=False) as data:
            far=torch.from_numpy(data['observations'][episode_ids,30:50].copy().reshape(-1,48))
        for seed in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
            with np.load(folder/'update-state.npz',allow_pickle=False) as z:
                initial_state={k[len('initial_'):]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('initial_')}
                inds=z['fitting_indices'].copy();adv=torch.from_numpy(z['advantages'][inds].copy())
            with np.load(folder/'collection.npz',allow_pickle=False) as z:
                observations=torch.from_numpy(z['obs'].copy().reshape(-1,48));actions=torch.from_numpy(z['actions'].copy().reshape(-1,8))[inds]
            initial=new_agent(cls);initial.load_state_dict(initial_state,strict=True)
            holdout,ids=kl_holdout(dict(b_obs=observations),inds,args)
            fitting=observations[inds];x=features(initial,fitting);cur=features(initial,holdout);old=features(initial,far)
            with torch.no_grad():
                innovation=((actions-initial.actor_mean(fitting))/initial.actor_logstd.exp()).double()
            advantage=((adv-adv.mean())/(adv.std()+1e-8))
            gradient=(innovation*advantage.double()[:,None]).T@x/len(x)
            current=cur.T@cur/len(cur);prior=old.T@old/len(old)
            ridge=plan['ridge_relative_trace']*float(current.trace())/len(current)
            row=dict(collection_seed=seed,feature_width=x.shape[1],fitting_rows=len(x),current_holdout_rows=len(cur),prior_late_rows=len(old),ridge=ridge,methods={})
            saved={}
            for name,weight in [('current_only',0.),('old_state_metric',1.)]:
                direction,stats=constrained_direction(gradient,current,prior,weight,plan['current_KL'],ridge)
                result['linear_solves']+=1
                branch=with_direction(initial,direction)
                current_actual=exact_KL(initial,branch,holdout);old_actual=exact_KL(initial,branch,far)
                if abs(current_actual['mean']-plan['current_KL'])>plan['current_KL']*.01:raise ValueError('Analytic/executed readout KL mismatch')
                if abs(old_actual['mean']-stats['old_KL'])>max(1e-6,stats['old_KL']*.01):raise ValueError('Old-state readout KL mismatch')
                stats.update(current_actual=current_actual,old_actual=old_actual,
                             clipped_surrogate_gain=clipped_surrogate(initial,branch,fitting,actions,advantage))
                row['methods'][name]=stats
                saved.update({name+'_'+k:v.detach().numpy() for k,v in branch.state_dict().items()})
            a,b=row['methods']['current_only'],row['methods']['old_state_metric']
            row.update(old_KL_ratio=b['old_KL']/a['old_KL'],linear_gain_ratio=b['linear_gain']/a['linear_gain'])
            row['gate_passed']=row['old_KL_ratio']<=.5 and row['linear_gain_ratio']>=.8 and b['clipped_surrogate_gain']>0
            result['rows'].append(row)
            np.savez_compressed(output/(str(seed)+'-branches.npz'),**saved);persist()
        result['gate_passed']=sum(r['gate_passed'] for r in result['rows'])>=2
        result['decision']='Eligible only for fresh paired forward/reverse task probes' if result['gate_passed'] else 'Reject this registered old-state metric trade-off; no robot follow-up'
        result['state']='complete'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.root,a.output,json.loads(Path(a.plan).read_text()))
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','traceback')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
