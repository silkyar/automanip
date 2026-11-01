"""Isolate score-gradient aggregation using fixed archived rollout experience.

Both directions use the same features, curvature, original minibatch advantage
normalizer and current KL. Full aggregation processes32x more score rows for
ONE direction; this is not yet a compute-matched learning comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from research.discovery.first_update_probe import load_production,new_agent,ProbeArgs,kl_holdout
from research.old_state_metric_probe import features,constrained_direction


def run(root,output,plan):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(state='started',new_environment_transitions=0,optimizer_calls=0,gpu_work=0,
        score_rows_by_estimator=dict(minibatch=4800,full_rollout=153600),rows=[],cross_draw=[],linear_solves=0)
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for p,sha in plan['inputs'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Source/data drift')
        torch.set_num_threads(2);torch.manual_seed(93493000)
        cls,_,_=load_production(root/plan['production']);args=ProbeArgs();stats={};reference=None
        previous=json.loads((root/plan['metric_report']).read_text())
        for seed in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
            with np.load(folder/'update-state.npz',allow_pickle=False) as z:
                state={k[len('initial_'):]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('initial_')}
                ids=z['fitting_indices'].copy();adv=torch.from_numpy(z['advantages'].copy())
            if reference is None:reference=state
            elif any(not torch.equal(reference[k],state[k]) for k in reference):raise ValueError('Initial Agent differs')
            agent=new_agent(cls);agent.load_state_dict(state,strict=True)
            with np.load(folder/'collection.npz',allow_pickle=False) as z:
                obs=torch.from_numpy(z['obs'].copy().reshape(-1,48));actions=torch.from_numpy(z['actions'].copy().reshape(-1,8))
            phis=[];innovations=[]
            with torch.no_grad():
                for start in range(0,len(obs),2048):
                    end=min(start+2048,len(obs));phi=features(agent,obs[start:end])
                    mean=agent.actor_mean[-1](phi[:,:-1].float())
                    phis.append(phi);innovations.append(((actions[start:end]-mean)/agent.actor_logstd.exp()).double())
            phi=torch.cat(phis);innovation=torch.cat(innovations)
            _,episodes=kl_holdout(dict(b_obs=obs),ids,args)
            holdout=np.concatenate([episodes+t*1024 for t in range(50)])
            current=phi[holdout].T@phi[holdout]/len(holdout)
            ridge=.001*float(current.trace())/len(current)
            mean,std=adv[ids].mean(),adv[ids].std()+1e-8
            normalized=((adv-mean)/std).double()
            g_small=(innovation[ids]*normalized[ids,None]).T@phi[ids]/len(ids)
            g_full=(innovation*normalized[:,None]).T@phi/len(phi)
            raw_score=innovation.T@phi/len(phi);raw_weighted=(innovation*adv.double()[:,None]).T@phi/len(phi)
            row=dict(collection_seed=seed,normalizer_mean=float(mean),normalizer_std=float(std),methods={})
            directions={}
            for name,g in [('minibatch',g_small),('full_rollout',g_full)]:
                direction,info=constrained_direction(g,current,torch.zeros_like(current),0.,.01,ridge)
                result['linear_solves']+=1;directions[name]=direction;row['methods'][name]=info
            expected=next(r for r in previous['rows'] if r['collection_seed']==seed)['methods']['current_only']['linear_gain']
            if abs(row['methods']['minibatch']['linear_gain']-expected)>1e-4:raise ValueError('Original minibatch objective differs')
            stats[seed]=dict(mean=mean,std=std,raw_score=raw_score,raw_weighted=raw_weighted,directions=directions)
            np.savez_compressed(output/(str(seed)+'-statistics.npz'),raw_score=raw_score.numpy(),raw_weighted=raw_weighted.numpy(),
                current=current.numpy(),minibatch_gradient=g_small.numpy(),full_gradient=g_full.numpy(),
                **{name+'_direction':d.numpy() for name,d in directions.items()})
            result['rows'].append(row);persist()
        for fitted,fit in stats.items():
            for evaluated,validation in stats.items():
                if fitted==evaluated:continue
                heldout_gradient=(validation['raw_weighted']-fit['mean'].double()*validation['raw_score'])/fit['std'].double()
                result['cross_draw'].append(dict(fitted=fitted,evaluated=evaluated,
                    **{name:float((heldout_gradient*d).sum()) for name,d in fit['directions'].items()}))
        small=np.asarray([r['minibatch'] for r in result['cross_draw']]);large=np.asarray([r['full_rollout'] for r in result['cross_draw']])
        result['mean_cross_gain']={'minibatch':float(small.mean()),'full_rollout':float(large.mean())}
        result['cross_gain_increase']=float(large.mean()/small.mean()) if small.mean()>0 else None
        fit_gain=np.mean([r['methods']['full_rollout']['linear_gain'] for r in result['rows']])
        result['full_cross_to_fitting_ratio']=float(large.mean()/fit_gain)
        result['gate_passed']=sum(large>0)>=5 and large.mean()>=2*max(small.mean(),0) and result['full_cross_to_fitting_ratio']>=.1
        result['state']='complete';result['decision']='Eligible for separately registered, matched-KL public direction assessment' if result['gate_passed'] else 'Full-rollout aggregation fails registered reliability/material-gain gate; no GPU direction test'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.root,a.output,json.loads(Path(a.plan).read_text()))
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','cross_draw','traceback')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
