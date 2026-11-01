"""Fresh physical assessment of learned waiting versus a learned fixed time."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.continuation_correction_policy import BRANCHES
from scripts.analyze_learned_correction import interval

CONTRASTS=[('state_continuation','learned_fixed_time'),('state_continuation','fixed_threshold'),('state_continuation','time_continuation'),('time_continuation','learned_fixed_time')]


def analyze(reports):
    if len(reports)!=2 or any(r['state']!='complete' for r in reports):raise ValueError('Two completed waitingreportsrequired')
    values={};counts={};grasps={}
    for i,seed in enumerate([2,3]):
        report=reports[i]
        if report['new_simulator_transitions_reserved']!=25600 or report['new_simulator_transitions_completed']!=25600 or report['optimizer_calls'] or report['confirmation_queries'] or report['final_queries']:raise ValueError('Work/protectedquerymismatch')
        for row in report['records']:
            key=seed,row['branch']
            if key in values or row['branch'] not in BRANCHES or row['collection_seed']!=seed or row['episodes']!=128 or row['execution_failures'] or len(row['batch_success_rates'])!=8:raise ValueError('Invalid waitingarm')
            values[key]=np.asarray(row['batch_success_rates']);counts[key]=row['successes'];grasps[key]=sum(r['grasp'] for r in row['batch_stage_counts'])
    if len(values)!=8:raise ValueError('All8armsrequired')
    comparisons={a+'-minus-'+b:interval([values[s,a]-values[s,b] for s in [2,3]]) for a,b in CONTRASTS}
    screens={b:comparisons['state_continuation-minus-'+b]['mean_gain']>=threshold and comparisons['state_continuation-minus-'+b]['interval_98_75'][0]>0 for b,threshold in [('learned_fixed_time',.10),('fixed_threshold',.10),('time_continuation',.05)]}
    grasp_screen=all(grasps[s,'state_continuation']/128>=.95 for s in [2,3])
    return dict(format='automanip-continuation-correction-analysis-v1',state='complete',successes={b:[counts[s,b] for s in [2,3]] for b in BRANCHES},ever_grasp={b:[grasps[s,b] for s in [2,3]] for b in BRANCHES},contrasts=comparisons,material_screens=screens,state_grasp_point_screen=grasp_screen,registered_overall_screen_passed=all(screens.values()) and grasp_screen,new_public_transitions=51200,optimizer_calls=0,worker_seconds=[r['wall_seconds'] for r in reports],scope='Assistedclosednumericstoppingprototype,twofixedparents. Samepayoffmodel/priors/supportforlearnedrules; learnedfixedtimefromtraininglabels. ClockawarecompositenotahistoricalMLPbenchmarkcandidate;nofromscratchor10xclaim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    result=analyze([json.loads((a.archive/f'remote/collection-{i}/run/report.json').read_text()) for i in range(2)])
    with (a.archive/'analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
