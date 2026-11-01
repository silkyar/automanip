"""Frozen learned-selector prototype under randomized public group assignment."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.learned_correction_policy import BRANCHES

CONTRASTS=[('learned_supported','base'),('learned_supported','learned_raw'),('learned_supported','random_switch'),('learned_raw','random_switch')]


def interval(blocks):
    d=np.asarray(blocks,dtype=float)
    if d.shape!=(2,8) or not np.isfinite(d).all():raise ValueError('8blocksperfixedsourcerequired')
    sample=np.random.default_rng(95709999).integers(0,8,(10000,2,8));draw=d[np.arange(2)[None,:,None],sample].mean((1,2))
    return dict(mean_gain=float(d.mean()),interval_98_75=np.quantile(draw,[.00625,.99375]).tolist())


def analyze(reports):
    if len(reports)!=2 or any(r['state']!='complete' for r in reports):raise ValueError('Two completedprototypereportsrequired')
    values={};counts={};grasp={}
    for i,seed in enumerate([2,3]):
        report=reports[i]
        if report['new_simulator_transitions_reserved']!=25600 or report['new_simulator_transitions_completed']!=25600 or report['optimizer_calls'] or report['confirmation_queries'] or report['final_queries']:raise ValueError('Unexpectedphysicsorprotectedqueries')
        for row in report['records']:
            key=seed,row['branch']
            if key in values or row['branch'] not in BRANCHES or row['collection_seed']!=seed or row['episodes']!=128 or row['execution_failures'] or len(row['batch_success_rates'])!=8:raise ValueError('Invalid prototypearm')
            values[key]=np.asarray(row['batch_success_rates']);counts[key]=row['successes'];grasp[key]=sum(b['grasp'] for b in row['batch_stage_counts'])
    if len(values)!=8:raise ValueError('All8armsrequired')
    contrasts={a+'-minus-'+b:interval([values[s,a]-values[s,b] for s in [2,3]]) for a,b in CONTRASTS}
    screens={b:contrasts['learned_supported-minus-'+b]['mean_gain']>=t and contrasts['learned_supported-minus-'+b]['interval_98_75'][0]>0 for b,t in [('base',.20),('learned_raw',.10),('random_switch',.10)]}
    retention=all(grasp[s,'learned_supported']/128>=.95 and (grasp[s,'learned_supported']-grasp[s,'base'])/128>=-.05 for s in [2,3])
    return dict(format='automanip-learned-correction-prototype-analysis-v1',successes={b:[counts[s,b] for s in [2,3]] for b in BRANCHES},ever_grasp={b:[grasp[s,b] for s in [2,3]] for b in BRANCHES},contrasts=contrasts,material_screens=screens,grasp_retention_point_screen=retention,registered_all_screens_passed=all(screens.values()) and retention,new_public_transitions=51200,optimizer_calls=0,worker_seconds=[r['wall_seconds'] for r in reports],scope='Assistedlearnedselectorprototypewithfixedcorrectiveactionprior; twofixed8.192Mparents. No fromscratch,fullretention,policyartifactpromotion,privateconfirmationor10xsamplereductionclaim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    reports=[json.loads((a.archive/f'remote/collection-{i}/run/report.json').read_text()) for i in range(2)]
    result=analyze(reports)
    with (a.archive/'analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
