"""Registered paired public release diagnostic, never a learned-policy score."""
import argparse
import json
from pathlib import Path
import numpy as np

BRANCHES=['no_update','scheduled_open','targeted_open','targeted_open_zero_arm']
CONTRASTS=[('targeted_open','no_update'),('targeted_open_zero_arm','no_update'),('targeted_open','scheduled_open'),('targeted_open_zero_arm','targeted_open')]


def paired_interval(differences):
    d=np.asarray(differences,dtype=float)
    if d.shape!=(2,512) or not np.isfinite(d).all():raise ValueError('Two fixed sources,512paired episodes required')
    blocks=d.reshape(2,8,64).mean(2)
    draws=np.random.default_rng(95209999).integers(0,8,size=(10000,2,8))
    samples=blocks[np.arange(2)[None,:,None],draws].mean((1,2))
    return dict(mean_difference=float(d.mean()),interval_98_75=np.quantile(samples,[.00625,.99375]).tolist(),resampling='8whole64episode vector batches within each of2fixed policies; no training-seed population inference')


def analyze(reports):
    if len(reports)!=2 or any(r['state']!='complete' for r in reports):raise ValueError('Both completed reports required; do not turn execution failures into scores')
    records={}
    for r in reports:
        if r['new_simulator_transitions_completed']!=204800 or r['new_simulator_transitions_reserved']!=204800:raise ValueError('Unexpected actual physics work')
        if r['optimizer_calls'] or r['confirmation_queries'] or r['final_queries']:raise ValueError('Unexpected learner or protected query')
        for row in r['records']:
            key=row['collection_seed'],row['mode'],row['branch']
            if key in records or key[0] not in [2,3] or key[1] not in ['deterministic','stochastic'] or key[2] not in BRANCHES:raise ValueError('Wrong or duplicate branch')
            if row['episodes']!=512 or row['execution_failures'] or len(row['episode_successes'])!=512:raise ValueError('Incomplete assigned episodes')
            controls=row['interventions']
            if len(controls)!=8 or any(len(c['applied'])!=64 or any(n!=(0 if key[2]=='no_update' else 5) for n in c['applied']) or c['max_prefix_difference']>1e-5 for c in controls):raise ValueError('Unmatched command budget or invalid paired prefix')
            records[key]=row
    if len(records)!=16:raise ValueError('All registered branches and modes required')
    result=dict(scope='Manual causal action interventions on two fixed failed8.192Mpolicies; not training or a benchmark improvement',new_simulator_transitions=409600,optimizer_calls=0,worker_seconds=[r['wall_seconds'] for r in reports],modes={})
    for mode in ['deterministic','stochastic']:
        values={b:np.asarray([records[s,mode,b]['episode_successes'] for s in [2,3]],dtype=float) for b in BRANCHES}
        estimates={a+'-minus-'+b:paired_interval(values[a]-values[b]) for a,b in CONTRASTS}
        result['modes'][mode]=dict(successes={b:[int(x.sum()) for x in values[b]] for b in BRANCHES},contrasts=estimates)
    main=result['modes']['stochastic']['contrasts']
    for branch in ['targeted_open','targeted_open_zero_arm']:
        contrast=main[branch+'-minus-no_update']
        result[branch+'_material_release_screen']=contrast['mean_difference']>=.20 and contrast['interval_98_75'][0]>0
    timing=main['targeted_open-minus-scheduled_open']
    result['timing_screen']=timing['mean_difference']>=.10 and timing['interval_98_75'][0]>0
    result['interpretation']='A positive screen supports a missing corrective action sequence in these policies only. A negative result rejects the specified five-command intervention as a sufficient repair. No automatic training or controller promotion.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    reports=[json.loads((a.archive/f'remote/collection-{i}/run/report.json').read_text()) for i in range(2)]
    result=analyze(reports)
    stages=[]
    for i,seed in enumerate([2,3]):
        for mode in ['deterministic','stochastic']:
            for branch in BRANCHES:
                with np.load(a.archive/f'remote/collection-{i}/run/{seed}/{mode}/{branch}.npz',allow_pickle=False) as data:
                    flags=data['stage_flags'].astype(bool)
                    if flags.shape!=(50,512,7):raise ValueError('Stage trace layout differs')
                    stages.append(dict(seed=seed,mode=mode,branch=branch,ever_stage={k:int(flags[:,:,j].any(0).sum()) for j,k in enumerate(['grasp','lift','transport','on_top','static','contact_free','success'])},final_success=int(flags[-1,:,6].sum()),last10_consecutive_success=int(flags[-10:,:,6].all(0).sum())))
    result['stage_diagnostics']=stages
    with (a.archive/'analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='stage_diagnostics'},indent=2))


if __name__=='__main__':main()
