"""Same full-rollout gradient, identity versus inverse-covariance conditioning.

Pure cached-statistic analysis; no new fitting rows, actor updates or simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def normalize_direction(direction,current,kappa=.01):
    energy=(direction*(direction@current)).sum()
    if not torch.isfinite(energy) or energy<=0:raise ValueError('Invalid direction energy')
    return direction*torch.sqrt(torch.as_tensor(2*kappa,dtype=energy.dtype)/energy)


def run(root,output,plan):
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    result=dict(state='started',new_environment_transitions=0,optimizer_calls=0,gpu_work=0,rows=[],cross_draw=[],methods={})
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for p,sha in plan['inputs'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:raise ValueError('Registered data/source changed')
        torch.set_num_threads(2)
        previous=json.loads((root/plan['aggregation_report']).read_text());stats={}
        for seed in plan['collection_seeds']:
            row0=next(r for r in previous['rows'] if r['collection_seed']==seed)
            with np.load(root/('artifacts/cumulative/gradient-aggregation-cpu-v1/'+str(seed)+'-statistics.npz'),allow_pickle=False) as z:
                current=torch.from_numpy(z['current'].copy());g=torch.from_numpy(z['full_gradient'].copy())
                inverse=torch.from_numpy(z['full_rollout_direction'].copy());score=torch.from_numpy(z['raw_score'].copy());weighted=torch.from_numpy(z['raw_weighted'].copy())
            identity=normalize_direction(g,current)
            eigenvalues,basis=torch.linalg.eigh(current)
            ridge=.001*float(current.trace())/len(current);low=eigenvalues<=ridge
            row=dict(collection_seed=seed,low_eigenvalue_dimensions=int(low.sum()),ridge=ridge,methods={})
            for name,d in [('inverse_covariance',inverse),('identity',identity)]:
                coordinates=d@basis
                energy=(coordinates.square()*eigenvalues).sum(0)
                info=dict(current_KL=float(.5*(d*(d@current)).sum()),linear_fitting_gain=float((g*d).sum()),
                    current_KL_fraction_in_low_eigenvalues=float(energy[low].sum()/energy.sum()))
                if abs(info['current_KL']-.01)>.0001:raise ValueError('Mean KL differs')
                row['methods'][name]=info
            stats[seed]=dict(current=current,gradient=g,score=score,weighted=weighted,mean=row0['normalizer_mean'],std=row0['normalizer_std'],
                directions=dict(inverse_covariance=inverse,identity=identity),basis=basis,low=low)
            np.savez_compressed(output/(str(seed)+'-directions.npz'),inverse_covariance=inverse.numpy(),identity=identity.numpy())
            result['rows'].append(row)
        for fitted,a in stats.items():
            for evaluated,b in stats.items():
                if fitted==evaluated:continue
                held=(b['weighted']-a['mean']*b['score'])/a['std']
                row=dict(fitted=fitted,evaluated=evaluated)
                for method,d in a['directions'].items():
                    row[method]=float((held*d).sum())
                    split=(held@a['basis'])*(d@a['basis'])
                    row[method+'_low_eigenvalue_gain']=float(split[:,a['low']].sum())
                result['cross_draw'].append(row)
        for method in ('inverse_covariance','identity'):
            values=np.asarray([r[method] for r in result['cross_draw']])
            fit=np.mean([r['methods'][method]['linear_fitting_gain'] for r in result['rows']])
            result['methods'][method]=dict(mean_cross_gain=float(values.mean()),cross_to_fitting_ratio=float(values.mean()/fit),positive_cross_cells=int(sum(values>0)))
        a,b=result['methods']['inverse_covariance'],result['methods']['identity']
        result['identity_cross_gain_increase']=b['mean_cross_gain']/a['mean_cross_gain'] if a['mean_cross_gain']>0 else None
        result['gate_passed']=b['positive_cross_cells']>=5 and b['cross_to_fitting_ratio']>=.1 and b['mean_cross_gain']>=2*max(a['mean_cross_gain'],0)
        result['state']='complete';result['decision']='Eligible for separately registered matched-KL public forward/reverse probe; no learning claim' if result['gate_passed'] else 'Preconditioning contrast fails reliability gate; inspect estimator/targets before new robot work'
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
