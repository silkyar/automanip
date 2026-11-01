"""Matched Monte Carlo targets: constant versus cross-fitted state baseline.

Actor-feature value fitting is established, not a new algorithm. Episode folds
prevent fitting a trajectory's own future rewards into its state baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from research.discovery.first_update_probe import load_production,new_agent
from research.old_state_metric_probe import features,constrained_direction


def discounted_returns(rewards,gamma=.9):
    if rewards.ndim!=2 or not torch.isfinite(rewards).all():raise ValueError('Finite time/episode rewards required')
    result=torch.empty_like(rewards,dtype=torch.float64);future=torch.zeros(rewards.shape[1],dtype=torch.float64)
    for t in reversed(range(len(rewards))):
        future=rewards[t].double()+gamma*future;result[t]=future
    return result


def crossfit_baselines(phi,returns,episodes,ridge_relative=.001):
    if len(phi)!=len(returns) or len(phi)%episodes:raise ValueError('Time-major episode layout required')
    parity=torch.arange(len(phi))%episodes%2
    constant=torch.empty_like(returns);state=torch.empty_like(returns);betas=[];constants=[]
    for held in (0,1):
        fit=parity!=held;test=~fit;x=phi[fit];y=returns[fit]
        covariance=x.T@x/len(x);ridge=ridge_relative*float(covariance.trace())/x.shape[1]
        beta=torch.linalg.solve(covariance+ridge*torch.eye(x.shape[1],dtype=x.dtype),x.T@y/len(x))
        constant[test]=y.mean();state[test]=phi[test]@beta
        betas.append(beta);constants.append(y.mean())
    return dict(constant=constant,state=state,mean_beta=torch.stack(betas).mean(0),mean_constant=torch.stack(constants).mean())


def run(root,output,plan):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(state='started',rows=[],cross_draw=[],new_environment_transitions=0,optimizer_calls=0,gpu_work=0,
        value_linear_solves=0,actor_linear_solves=0,value_fitting_rows=0)
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for p,sha in plan['inputs'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Source/data drift')
        torch.set_num_threads(2);torch.manual_seed(93494000)
        cls,_,_=load_production(root/plan['production']);previous=json.loads((root/plan['aggregation_report']).read_text());folds={};reference=None
        for seed in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
            with np.load(folder/'update-state.npz',allow_pickle=False) as z:
                initial={k[len('initial_'):]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('initial_')}
            if reference is None:reference=initial
            elif any(not torch.equal(reference[k],initial[k]) for k in reference):raise ValueError('Initial Agent differs')
            agent=new_agent(cls);agent.load_state_dict(initial,strict=True)
            with np.load(folder/'collection.npz',allow_pickle=False) as z:
                obs=torch.from_numpy(z['obs'].copy().reshape(-1,48));actions=torch.from_numpy(z['actions'].copy().reshape(-1,8));rewards=torch.from_numpy(z['rewards'].copy())
                if (z['dones'][1:]!=0).any():raise ValueError('Unexpected interior episode boundary')
            returns=discounted_returns(rewards,plan['gamma']).flatten();xs=[];es=[]
            with torch.no_grad():
                for start in range(0,len(obs),2048):
                    end=min(start+2048,len(obs));x=features(agent,obs[start:end]);mu=agent.actor_mean[-1](x[:,:-1].float())
                    xs.append(x);es.append(((actions[start:end]-mu)/agent.actor_logstd.exp()).double())
            phi,innovation=torch.cat(xs),torch.cat(es)
            baseline=crossfit_baselines(phi,returns,1024,plan['ridge_relative'])
            result['value_linear_solves']+=2;result['value_fitting_rows']+=len(phi)
            row0=next(r for r in previous['rows'] if r['collection_seed']==seed);scale=row0['normalizer_std']
            with np.load(root/('artifacts/cumulative/gradient-aggregation-cpu-v1/'+str(seed)+'-statistics.npz'),allow_pickle=False) as z:current=torch.from_numpy(z['current'].copy())
            ridge=.001*float(current.trace())/len(current)
            weights=innovation.square().sum(-1)*phi.square().sum(-1)
            row=dict(collection_seed=seed,models={});directions={}
            for method in ('constant','state'):
                residual=(returns-baseline[method])/scale
                gradient=(innovation*residual[:,None]).T@phi/len(phi)
                direction,stats=constrained_direction(gradient,current,torch.zeros_like(current),0.,.01,ridge)
                result['actor_linear_solves']+=1;directions[method]=direction
                stats.update(crossfit_value_MSE=float(((returns-baseline[method])**2).mean()),
                    crossfit_score_second_moment=float((weights*residual.square()).mean()))
                row['models'][method]=stats
            row['state_to_constant_score_moment_ratio']=row['models']['state']['crossfit_score_second_moment']/row['models']['constant']['crossfit_score_second_moment']
            folds[seed]=dict(phi=phi,innovation=innovation,returns=returns,scale=scale,
                beta=baseline['mean_beta'],constant=baseline['mean_constant'],directions=directions)
            np.savez_compressed(output/(str(seed)+'-baseline-directions.npz'),state_beta=baseline['mean_beta'].numpy(),constant=baseline['mean_constant'].numpy(),**{k+'_direction':v.numpy() for k,v in directions.items()})
            result['rows'].append(row);persist()
        for fitted,a in folds.items():
            for evaluated,b in folds.items():
                if fitted==evaluated:continue
                row=dict(fitted=fitted,evaluated=evaluated)
                for method in ('constant','state'):
                    prediction=a['constant'] if method=='constant' else b['phi']@a['beta']
                    residual=(b['returns']-prediction)/a['scale']
                    score=((b['phi']@a['directions'][method].T)*b['innovation']).sum(-1)*residual
                    row[method]=float(score.mean())
                result['cross_draw'].append(row)
        result['methods']={}
        for method in ('constant','state'):
            cross=np.asarray([r[method] for r in result['cross_draw']]);fit=np.mean([r['models'][method]['linear_gain'] for r in result['rows']])
            result['methods'][method]=dict(mean_cross_gain=float(cross.mean()),positive_cross_cells=int(sum(cross>0)),cross_to_fitting_ratio=float(cross.mean()/fit))
        a,b=result['methods']['constant'],result['methods']['state']
        result['gate_passed']=b['positive_cross_cells']>=5 and b['cross_to_fitting_ratio']>=.1 and b['mean_cross_gain']>=2*max(a['mean_cross_gain'],0) and sum(r['state_to_constant_score_moment_ratio']<=.8 for r in result['rows'])>=2
        result['state']='complete';result['decision']='Eligible for fresh public direction tests with matched MC targets' if result['gate_passed'] else 'Reject this ordinary value-baseline mechanism under the registered reliability gate; no robot follow-up'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        import resource,sys
        result['peak_process_RSS_MB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.root,a.output,json.loads(Path(a.plan).read_text()))
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','cross_draw','traceback')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
