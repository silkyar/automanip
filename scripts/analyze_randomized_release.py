"""Blocked randomized intervention analysis, no exact trajectory pairing."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.randomized_release_policy import BRANCHES
from scripts.analyze_release_control import CONTRASTS


def interval(block_differences):
    d=np.asarray(block_differences,dtype=float)
    if d.shape!=(2,32) or not np.isfinite(d).all():raise ValueError('32vector blocks from each fixed source required')
    draws=np.random.default_rng(95409999).integers(0,32,size=(10000,2,32))
    samples=d[np.arange(2)[None,:,None],draws].mean((1,2))
    return dict(mean_difference=float(d.mean()),interval_98_75=np.quantile(samples,[.00625,.99375]).tolist(),scope='Randomized16episodes per arm within64lane blocks; resample32whole blocks within each of2fixed policies, not training-seed inference')


def analyze(reports):
    if len(reports)!=2 or any(r['state']!='complete' for r in reports):raise ValueError('Two completed randomized reports required')
    values={};counts={}
    for i,seed in enumerate([2,3]):
        report=reports[i]
        if report['new_simulator_transitions_reserved']!=102400 or report['new_simulator_transitions_completed']!=102400 or report['optimizer_calls']:raise ValueError('Work accounting differs')
        for row in report['records']:
            key=seed,row['branch']
            if row['collection_seed']!=seed or key in values or row['branch'] not in BRANCHES or row['episodes']!=512 or row['execution_failures'] or len(row['batch_success_rates'])!=32:raise ValueError('Wrong or incomplete randomized arm')
            values[key]=row['batch_success_rates'];counts[key]=row['successes']
    if len(values)!=8:raise ValueError('All8arms required')
    comparisons={a+'-minus-'+b:interval([[x-y for x,y in zip(values[s,a],values[s,b])] for s in [2,3]]) for a,b in CONTRASTS}
    return dict(format='automanip-randomized-release-analysis-v1',new_simulator_transitions=204800,optimizer_calls=0,successes={b:[counts[s,b] for s in [2,3]] for b in BRANCHES},comparisons=comparisons,material_release_screen={b:comparisons[b+'-minus-no_update']['mean_difference']>=.20 and comparisons[b+'-minus-no_update']['interval_98_75'][0]>0 for b in ['targeted_open','targeted_open_zero_arm']},timing_screen=comparisons['targeted_open-minus-scheduled_open']['mean_difference']>=.10 and comparisons['targeted_open-minus-scheduled_open']['interval_98_75'][0]>0,worker_seconds=[r['wall_seconds'] for r in reports],scope='Manually specified interventions on two frozen failed policies; randomized group effects, not paired episode effects, learned skill, training-seed generalization or sample-efficiency gain')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    reports=[json.loads((a.archive/f'remote/collection-{i}/run/report.json').read_text()) for i in range(2)]
    result=analyze(reports)
    with (a.archive/'analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
