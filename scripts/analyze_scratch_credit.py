"""Registered early learning-curve comparison; data-only, no cloud imports."""
import argparse
import json
from pathlib import Path


GRID=[0,512000,1024000,2048000,4096000,6144000,8192000,10240000]
METHODS=['ppo-gae','ppo-finite-mc']


def normalized_auc(points):
    if len(points)<2 or points[0][0]!=0 or any(b[0]<=a[0] for a,b in zip(points,points[1:])):
        raise ValueError('Increasing curve from0 required')
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))/points[-1][0]


def sustained_crossing(points,threshold):
    for left,right in zip(points,points[1:]):
        if left[1]>=threshold and right[1]>=threshold:
            return dict(observed_first_step=left[0],confirmed_at=right[0],censored=False)
    return dict(observed_first_step=None,confirmed_at=None,censored=True,last_observed_step=points[-1][0],
                interpretation='No two consecutive observed checkpoints crossed threshold; no interpolated crossing claim')


def analyze(reports,receipts,*,methods=METHODS):
    if len(methods)!=2 or len(set(methods))!=2:raise ValueError('Two distinct methods required')
    if len(reports)!=6 or len(receipts)!=6:raise ValueError('All6registered trial records required')
    curves={};work={}
    for report,receipt in zip(reports,receipts):
        if report['state']!='complete' or receipt['state']!='complete':raise ValueError('Execution incomplete; preserve error, do not score it as0success')
        seed,method=report['seed'],report['method'];key=(seed,method)
        if seed not in [1,2,3] or method not in methods or key in curves:raise ValueError('Wrong or duplicate seed/method')
        if receipt['seed']!=seed or receipt['method']!=method:raise ValueError('Training/scoring identity mismatch')
        if report['final_queries'] or report['confirmation_queries']:raise ValueError('Protected queries unexpectedly consumed')
        if receipt['actual_work']['environment_completed']>10240000:raise ValueError('Interaction ceiling exceeded')
        points={}
        for point in report['points']:
            if point['episodes']!=128 or point['failures'] or not 0<=point['successes']<=128:raise ValueError('Invalid rollout assessment')
            if point['steps'] in points:raise ValueError('Duplicate checkpoint')
            if point['steps']>receipt['actual_work']['environment_completed']:raise ValueError('Checkpoint exceeds real completed work')
            points[point['steps']]=point
        if 0 not in points:raise ValueError('Initial policy missing')
        curves[key]=points;work[key]=receipt
    pairs=[]
    for seed in [1,2,3]:
        a,b=[curves[seed,m] for m in methods]
        common=[s for s in GRID if s in a and s in b]
        if len(common)<2:raise ValueError('Insufficient common learning-curve points')
        # A missing middle checkpoint must not be silently interpolated away.
        if common!=[s for s in GRID if s<=common[-1]]:raise ValueError('Incomplete common checkpoint prefix')
        result=dict(seed=seed,common_steps=common,methods={})
        for method,points in zip(methods,[a,b]):
            values=[(step,points[step]['successes']/128) for step in common]
            receipt=work[seed,method]
            result['methods'][method]=dict(success_curve=values,success_auc=normalized_auc(values),
                T80=sustained_crossing(values,.8),T90=sustained_crossing(values,.9),
                actual_training_transitions=receipt['actual_work']['environment_completed'],
                optimizer_calls=receipt['actual_work']['optimizer_completed'],
                trainer_process_seconds=receipt['trainer_process_seconds'],wrapper_seconds=receipt['wrapper_seconds'])
        control,candidate=[result['methods'][m] for m in methods]
        result['paired_auc_gain']=candidate['success_auc']-control['success_auc']
        result['paired_final_success_gain']=candidate['success_curve'][-1][1]-control['success_curve'][-1][1]
        t0,t1=control['T80'],candidate['T80']
        result['observed_T80_speedup']=None if t0['censored'] or t1['censored'] or not t1['observed_first_step'] else t0['observed_first_step']/t1['observed_first_step']
        result['candidate_uses_no_more_optimizer_calls']=candidate['optimizer_calls']<=control['optimizer_calls']
        pairs.append(result)
    mean_auc=sum(p['paired_auc_gain'] for p in pairs)/3
    # Strong evidence cannot be manufactured from censored curves or incomplete
    # learner compute accounting. This file reports the screen without promotion.
    crossed=all(p['observed_T80_speedup'] is not None and p['observed_T80_speedup']>=2 for p in pairs)
    return dict(format='automanip-scratch-credit-analysis-v1',paired_seeds=pairs,
                mean_paired_auc_gain=mean_auc,
                twofold_observed_T80_screen=bool(crossed),
                final_success_point_preserved_all_seeds=all(p['paired_final_success_gain']>=0 for p in pairs),
                no_more_optimizer_calls_all_seeds=all(p['candidate_uses_no_more_optimizer_calls'] for p in pairs),
                material_compute_matched_result_established=False,
                compute_requirement='Also compare measured learner time from retained TensorBoard events; equal ceilings and optimizer counts alone do not establish equal total compute.',
                scope='Fresh random initialization,3paired training seeds on one frozen task. Public development curves, not independent confirmation or held-out task validation. Negative results and censored crossings retained.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports=[json.loads((a.scores/f'trial-{i}/report.json').read_text()) for i in range(6)]
    receipts=[json.loads((a.training/f'trial-{i}/receipt.json').read_text()) for i in range(6)]
    result=analyze(reports,receipts)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='paired_seeds'},indent=2))


if __name__=='__main__':main()
