"""Held-batch outcome prediction from training-only randomized switch data.

This does not deploy a gate or estimate its closed-loop success. Privileged
velocity/stage columns never enter model features.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy


def partition(batch):
    if type(batch) is not int or not 0<=batch<32:raise ValueError('Closed32batchpartition')
    return ['test','calibration','fit','fit'][batch%4]


def fit_head(design,target):
    gram=design.T@design;ridge=.001*np.trace(gram)/len(gram)
    return np.linalg.solve(gram+ridge*np.eye(len(gram)),design.T@target)


def interval(blocks):
    blocks=np.asarray(blocks,dtype=float)
    if blocks.shape!=(8,) or not np.isfinite(blocks).all():raise ValueError('8wholeheldoutbatches required')
    draw=np.random.default_rng(95609999).integers(0,8,(10000,8))
    return dict(mean_gain=float(blocks.mean()),interval_98_75=np.quantile(blocks[draw].mean(1),[.00625,.99375]).tolist())


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();root=a.archive
    plan=json.loads((root/'provenance/inputs/plan.json').read_text());rows=[];fits=0;forwards=0
    for index,source in enumerate(plan['sources']):
        report=json.loads((root/f'remote/source-{index}/run/report.json').read_text())
        if report['state']!='complete' or report['partition']!='training' or report['environment_completed']!=102400 or report['optimizer_calls']:raise ValueError('Incomplete training corpus')
        schema=PolicySchema(plan['task_fingerprint'],48,8)
        policy=NumpyPolicy(load_policy(root/'provenance/inputs'/source['policy'],schema),schema)
        records=[]
        for batch in range(32):
            with np.load(root/f'remote/source-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                x=z['state_at_decision'].copy();t=z['treatment'].astype(int);y=z['outcome'].astype(float);clock=z['decision'].copy()
                np.testing.assert_array_equal(x,z['observations'][clock,np.arange(64)])
                np.testing.assert_array_equal(z['applied'],np.where(t==1,50-clock,0))
                if sorted(t.tolist())!=[0]*32+[1]*32:raise ValueError('Unbalancedrandomization')
                for weight,bias in policy.layers[:-1]:x=np.tanh(x@weight.T+bias)
                forwards+=64;design=np.column_stack([x,np.ones(64)])
                records.append(dict(batch=batch,part=partition(batch),design=design,treatment=t,target=y,clock=clock))
        def merged(part,key):return np.concatenate([r[key] for r in records if r['part']==part])
        train_x=merged('fit','design');train_y=merged('fit','target');train_a=merged('fit','treatment');train_clock=merged('fit','clock')
        coefficients=[];false_coefficients=[];constant=[];time_predictions=[]
        rng=np.random.default_rng(95629999+index)
        for arm in [0,1]:
            mask=train_a==arm;x,y=train_x[mask],train_y[mask]
            coefficients.append(fit_head(x,y));false_coefficients.append(fit_head(x,rng.permutation(y)));fits+=2
            constant.append((y.sum()+1)/(len(y)+2));bins=np.minimum((train_clock[mask]-10)//5,6)
            time_predictions.append(np.array([(y[bins==b].sum()+1)/((bins==b).sum()+2) for b in range(7)]))
        coefficients=np.stack(coefficients);false_coefficients=np.stack(false_coefficients)
        predictions=[];differences={'versus_constant':[],'versus_time':[]};scores={k:[] for k in ['feature','constant','time','shuffled']}
        for record in records:
            x,y,arm=record['design'],record['target'],record['treatment'];n=len(y)
            feature=np.clip((x@coefficients.T)[np.arange(n),arm],0,1)
            shuffled=np.clip((x@false_coefficients.T)[np.arange(n),arm],0,1)
            null=np.asarray(constant)[arm];clock=np.array([time_predictions[t][min((int(c)-10)//5,6)] for t,c in zip(arm,record['clock'])])
            errors={k:(pred-y)**2 for k,pred in [('feature',feature),('constant',null),('time',clock),('shuffled',shuffled)]}
            if record['part']=='test':
                for k in scores:scores[k].extend(errors[k].tolist())
                differences['versus_constant'].append(float((errors['constant']-errors['feature']).mean()));differences['versus_time'].append(float((errors['time']-errors['feature']).mean()))
            predictions.append(dict(batch=record['batch'],partition=record['part'],outcome=y.tolist(),treatment=arm.tolist(),feature_score=feature.tolist(),constant_score=null.tolist(),time_score=clock.tolist(),shuffled_score=shuffled.tolist()))
        contrasts={k:interval(v) for k,v in differences.items()}
        rows.append(dict(seed=source['training_seed'],new_training_episodes=2048,fit_episodes=1024,calibration_episodes=512,test_episodes=512,test_brier={k:float(np.mean(v)) for k,v in scores.items()},contrasts=contrasts,material_predictive_screen=all(v['mean_gain']>=.02 and v['interval_98_75'][0]>0 for v in contrasts.values()),predictions=predictions))
        np.savez_compressed(root/f'outcome-models-source-{index}.npz',coefficients=coefficients,shuffled_coefficients=false_coefficients,constant=np.asarray(constant),time=np.asarray(time_predictions))
    result=dict(state='complete',rows=rows,linear_solves=fits,source_actor_feature_forward_rows=forwards,new_training_transitions=204800,elapsed_seconds=time.monotonic()-started,model_compute='CPU closed-form linear solves; fitting row counts1024*2models*2sources=4096; no policyoptimizer',scope='Training-only factual outcome prediction on heldwholevectorbatches. Scores are not calibrated closed-loop switch probabilities; randomdecisionstate distribution differs from first-triggerdeployment. No policyartifact, frozenbenchmarkquery or sampleefficiencyclaim.')
    with (root/'outcome-analysis-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='predictions'}))


if __name__=='__main__':main()
