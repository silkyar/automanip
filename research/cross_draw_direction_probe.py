"""Does a fitted direction generalize to independently collected score data?

Uses only already archived public training data. This measures reproducibility
of a GAE/score estimator, not true advantages or causal policy improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from research.discovery.first_update_probe import load_production,new_agent,ProbeArgs,kl_holdout
from research.old_state_metric_probe import features


def score_projection(phi,innovation,advantage,direction):
    return advantage*((phi@direction.T)*innovation).sum(-1)


def run(root,output,plan):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(state='started',new_environment_transitions=0,optimizer_calls=0,gpu_work=0,
        rows=[],methods={},scope='Independent-data score-estimator alignment only; not true policy gradients or task returns')
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for p,sha in plan['inputs'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Source/data drift')
        torch.set_num_threads(2);torch.manual_seed(93492000)
        cls,_,_=load_production(root/plan['production']);args=ProbeArgs()
        previous=json.loads((root/plan['metric_report']).read_text());fits={};reference=None
        for seed in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
            with np.load(folder/'update-state.npz',allow_pickle=False) as z:
                state={k[len('initial_'):]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('initial_')}
                ids=z['fitting_indices'].copy();adv=torch.from_numpy(z['advantages'][ids].copy())
            if reference is None:reference=state
            elif any(not torch.equal(reference[k],state[k]) for k in reference):raise ValueError('Collection initial agents differ')
            initial=new_agent(cls);initial.load_state_dict(state,strict=True)
            with np.load(root/('artifacts/cumulative/old-state-metric-cpu-v1/'+str(seed)+'-branches.npz'),allow_pickle=False) as z:
                directions={}
                for method in ('current_only','old_state_metric'):
                    weight=torch.from_numpy(z[method+'_actor_mean.6.weight'].copy())
                    bias=torch.from_numpy(z[method+'_actor_mean.6.bias'].copy())
                    std=state['actor_logstd'].exp().flatten().double()
                    directions[method]=torch.cat(((weight-state['actor_mean.6.weight']).double()/std[:,None],
                        ((bias-state['actor_mean.6.bias']).double()/std)[:,None]),dim=1)
            fits[seed]=dict(ids=ids,mean=adv.mean(),std=adv.std()+1e-8,directions=directions)
        agent=new_agent(cls);agent.load_state_dict(reference,strict=True)
        for evaluated in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(evaluated)
            with np.load(folder/'collection.npz',allow_pickle=False) as z:
                obs=torch.from_numpy(z['obs'].copy().reshape(-1,48));actions=torch.from_numpy(z['actions'].copy().reshape(-1,8))
            with np.load(folder/'update-state.npz',allow_pickle=False) as z:advantages=torch.from_numpy(z['advantages'].copy())
            projection={(seed,method):torch.empty(len(obs),dtype=torch.float64) for seed in fits for method in fits[seed]['directions']}
            with torch.no_grad():
                for start in range(0,len(obs),2048):
                    end=min(start+2048,len(obs));phi=features(agent,obs[start:end])
                    mean=agent.actor_mean[-1](phi[:,:-1].float())
                    innovation=((actions[start:end]-mean)/agent.actor_logstd.exp()).double()
                    for trained,fit in fits.items():
                        adv=((advantages[start:end]-fit['mean'])/fit['std']).double()
                        for method,direction in fit['directions'].items():
                            projection[(trained,method)][start:end]=score_projection(phi,innovation,adv,direction)
            _,holdout_episodes=kl_holdout(dict(b_obs=obs),fits[evaluated]['ids'],args)
            holdout=np.concatenate([holdout_episodes+t*1024 for t in range(50)])
            for (trained,method),values in projection.items():
                if trained==evaluated:
                    actual_fit=float(values[fits[trained]['ids']].mean())
                    expected=next(r for r in previous['rows'] if r['collection_seed']==trained)['methods'][method]['linear_gain']
                    if abs(actual_fit-expected)>1e-4:raise ValueError('Score projection does not reconstruct fitting objective')
                    row=dict(trained_collection=trained,evaluated_collection=evaluated,method=method,
                        evaluation='within_draw_disjoint_episode_labels',mean_gain=float(values[holdout].mean()),
                        fitting_gain=actual_fit,fitting_reconstruction_error=abs(actual_fit-expected))
                else:
                    by_episode=values.reshape(50,1024).mean(0).numpy()
                    np.save(output/(str(trained)+'-on-'+str(evaluated)+'-'+method+'.npy'),by_episode,allow_pickle=False)
                    row=dict(trained_collection=trained,evaluated_collection=evaluated,method=method,
                        evaluation='independent_collection',mean_gain=float(values.mean()),
                        by_time_bin=[float(v) for v in values.reshape(5,10,1024).mean((1,2))])
                result['rows'].append(row)
            persist()
        for method in ('current_only','old_state_metric'):
            cross=[r['mean_gain'] for r in result['rows'] if r['method']==method and r['evaluation']=='independent_collection']
            within=[r['mean_gain'] for r in result['rows'] if r['method']==method and r['evaluation']=='within_draw_disjoint_episode_labels']
            fit=[r['fitting_gain'] for r in result['rows'] if r['method']==method and 'fitting_gain' in r]
            ratio=float(np.mean(cross)/np.mean(fit))
            result['methods'][method]=dict(cross_draw_gains=cross,within_draw_disjoint_gains=within,
                mean_cross_gain=float(np.mean(cross)),cross_to_fitting_gain_ratio=ratio,
                gate_passed=sum(v>0 for v in cross)>=5 and sum(v>0 for v in within)>=2 and ratio>=.1)
        result['gate_passed']=result['methods']['old_state_metric']['gate_passed']
        result['state']='complete';result['decision']='Eligible for already prepared public paired direction rollouts' if result['gate_passed'] else 'Do not run prepared GPU probe yet: fitted signal lacks required independent-data reproduction; investigate estimator variance/targets'
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
