"""Registered2x2post-opening intervention effects, conditional on fixed actors."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.release_maintenance_policy import BRANCHES
from scripts.analyze_randomized_release import interval

CONTRASTS={'keep_open-minus-resume_actor':[-1,1,0,0],
           'zero_arm-minus-resume_actor':[-1,0,1,0],
           'keep_open_zero_arm-minus-resume_actor':[-1,0,0,1],
           'factorial_interaction':[1,-1,-1,1]}


def analyze(reports):
    if len(reports)!=2 or any(r['state']!='complete' for r in reports):raise ValueError('Two complete maintenance reports required')
    values={};counts={}
    for i,seed in enumerate([2,3]):
        report=reports[i]
        if report['new_simulator_transitions_reserved']!=102400 or report['new_simulator_transitions_completed']!=102400 or report['optimizer_calls']:raise ValueError('Work accounting differs')
        for row in report['records']:
            key=seed,row['branch']
            if row['collection_seed']!=seed or key in values or row['branch'] not in BRANCHES or row['episodes']!=512 or row['execution_failures'] or len(row['batch_success_rates'])!=32:raise ValueError('Wrong or incomplete maintenance arm')
            values[key]=np.asarray(row['batch_success_rates']);counts[key]=row['successes']
    if len(values)!=8:raise ValueError('All8arms required')
    comparisons={name:interval([sum(c*values[s,b] for c,b in zip(coeff,BRANCHES)) for s in [2,3]]) for name,coeff in CONTRASTS.items()}
    screen={b:comparisons[b+'-minus-resume_actor']['mean_difference']>=.20 and comparisons[b+'-minus-resume_actor']['interval_98_75'][0]>0 for b in BRANCHES[1:]}
    return dict(format='automanip-release-maintenance-analysis-v1',new_simulator_transitions=204800,optimizer_calls=0,successes={b:[counts[s,b] for s in [2,3]] for b in BRANCHES},comparisons=comparisons,material_maintenance_screen=screen,worker_seconds=[r['wall_seconds'] for r in reports],scope='Hand-specified post-opening2x2control intervention on2fixedfailedpolicies. No training, new learned skill, population-of-training-seeds or sample-efficiency claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    reports=[json.loads((a.archive/f'remote/collection-{i}/run/report.json').read_text()) for i in range(2)]
    result=analyze(reports)
    with (a.archive/'analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
