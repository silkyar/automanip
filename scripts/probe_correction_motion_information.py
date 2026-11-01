"""Fixed, counted outcome-model diagnosis from an existing randomized corpus."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from research.correction_motion_information import contrast, extract, fit_map, ridge, shuffle_motion, time_basis


ROOT=Path('artifacts/cumulative/random-switch-training-20261030')
OUTPUT=Path('artifacts/cumulative/correction-motion-information-CPU-20261031')
DRAW_SEEDS=(97110000,97120000)
METHODS=('current256','current512','oracle','history','shuffled_oracle')


def load_source(index):
    records=[]
    hashes={}
    report_path=ROOT/f'remote/source-{index}/run/report.json'
    report=json.loads(report_path.read_text())
    if report['state']!='complete' or report['environment_completed']!=102400 or report['optimizer_calls']:
        raise ValueError('Original training collection incomplete')
    for batch in range(32):
        if batch%4==1:
            continue  # Original calibration batches are not needed for this test.
        path=ROOT/f'remote/source-{index}/run/batch-{batch}.npz'
        with np.load(path,allow_pickle=False) as z:
            clock=z['decision'].copy()
            treatment=z['treatment'].copy()
            if sorted(treatment.tolist())!=[0]*32+[1]*32:
                raise ValueError('Randomized assignment layout differs')
            features=extract(z['observations'],z['diagnostic_cube_linear_velocity'],z['diagnostic_cube_angular_velocity'],clock)
            if not np.array_equal(z['state_at_decision'],features['current'][:,:48]):
                raise ValueError('Pre-action current-state alignment differs')
            if not np.array_equal(z['applied'],np.where(treatment==1,50-clock,0)):
                raise ValueError('Assigned intervention timing differs')
            target=z['outcome'].copy()
            if target.shape!=(64,) or not np.isin(target,[0,1]).all():
                raise ValueError('Outcome labels differ')
            mask=treatment==1
            record={k:v[mask] for k,v in features.items()}
            record.update(target=target[mask].astype(float),clock=clock[mask],
                batch=np.full(32,batch),lane=np.flatnonzero(mask))
            records.append(record)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    merged={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    merged['fit']=np.isin(merged['batch']%4,[2,3])
    merged['test']=merged['batch']%4==0
    if (merged['fit'].sum(),merged['test'].sum())!=(512,256):
        raise ValueError('Whole-batch split differs')
    return merged,hashes


def probe():
    started=time.process_time()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    plan_path=Path('cumulative/experiments/correction-motion-information-CPU-v1.json')
    plan=json.loads(plan_path.read_text())
    if not plan['execution_authorized']:
        raise PermissionError('CPU study not admitted')
    solves=0
    rows=[]
    source_hashes={}
    for index,parent in enumerate((2,3)):
        data,hashes=load_source(index)
        source_hashes.update(hashes)
        fit,test=data['fit'],data['test']
        y=data['target']
        clock_design=np.column_stack([np.ones(len(y)),time_basis(data['clock'])])
        time_coefficient=ridge(clock_design[fit],y[fit]);solves+=1
        time_prediction=np.clip(clock_design@time_coefficient,0,1)
        permuted=shuffle_motion(data['oracle'],data['batch'],data['clock'],97130000+parent)
        for draw,seed in enumerate(DRAW_SEEDS):
            arrays={k:data[k] for k in ('target','clock','batch','lane','fit','test')}
            arrays['prediction_time']=time_prediction
            arrays['coefficient_time']=time_coefficient
            scores={'time':float(np.square(time_prediction[test]-y[test]).mean())}
            for method in METHODS:
                extra={'oracle':data['oracle'],'history':data['history'],'shuffled_oracle':permuted}.get(method)
                x=data['current'] if extra is None else np.column_stack([data['current'],extra])
                phi,parameters=fit_map(x[fit],x,seed,512 if method=='current512' else 256)
                design=np.column_stack([phi,clock_design])
                coefficient=ridge(design[fit],y[fit]);solves+=1
                prediction=np.clip(design@coefficient,0,1)
                arrays['prediction_'+method]=prediction
                arrays['coefficient_'+method]=coefficient
                for key,value in parameters.items():arrays[method+'_'+key]=value
                scores[method]=float(np.square(prediction[test]-y[test]).mean())
            # Save completed fits/predictions before aggregation; no reporting
            # failure should require repeating a regression or changing its data.
            path=OUTPUT/f'parent-{parent}-draw-{draw}.npz'
            np.savez_compressed(path,**arrays)
            oracle_error=np.square(arrays['prediction_oracle'][test]-y[test])
            contrasts={}
            for control in ('current512','shuffled_oracle'):
                control_error=np.square(arrays['prediction_'+control][test]-y[test])
                c=contrast(control_error,oracle_error,data['batch'][test])
                c['relative_brier_reduction']=c['mean_brier_gain']/float(control_error.mean()) if control_error.mean() else 0.
                contrasts[control]=c
            passed=all(c['mean_brier_gain']>=.02 and c['relative_brier_reduction']>=.20
                       and c['interval_99_375'][0]>0 for c in contrasts.values())
            history_contrast=contrast(np.square(arrays['prediction_current512'][test]-y[test]),
                                      np.square(arrays['prediction_history'][test]-y[test]),data['batch'][test])
            gap=scores['current512']-scores['oracle']
            rows.append(dict(parent=parent,draw=draw,feature_seed=seed,
                fit_rows=512,test_rows=256,fit_successes=int(y[fit].sum()),test_successes=int(y[test].sum()),
                test_brier=scores,oracle_minus_controls=contrasts,oracle_screen_passed=passed,
                history_improvement=history_contrast,
                history_fraction_of_oracle_gap=(scores['current512']-scores['history'])/gap if gap>0 else None,
                numeric_archive=str(path),numeric_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if solves!=22:
        raise ValueError('Unexpected regression count')
    result=dict(format='automanip-correction-motion-information-CPU-v1',state='complete',
        oracle_screen_passed=all(r['oracle_screen_passed'] for r in rows),rows=rows,
        linear_solves=solves,fit_rows=11264,random_feature_forward_rows=15360,
        CPU_seconds=time.process_time()-started,new_environment_transitions=0,
        reused_original_training_transitions=204800,simulator_queries=0,optimizer_calls=0,
        policy_exports=0,robot_assessments=0,source_sha256=source_hashes,
        protocol_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        scope='Prediction on a reused acquired-policy intervention corpus with diagnostic decision time. Privileged velocities are diagnostic only; no policy, causal velocity intervention, fresh generalization or sample-efficiency claim.')
    result_path=Path('cumulative/results/correction-motion-information-CPU-20261031.json')
    with result_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    provenance=OUTPUT/'provenance';provenance.mkdir()
    for source in [plan_path,Path('research/correction_motion_information.py'),Path(__file__),Path('tests/test_correction_motion_information.py')]:
        target=provenance/source.name
        target.write_bytes(source.read_bytes())
    return result


if __name__=='__main__':
    result=probe()
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','rows')},indent=2))
    for row in result['rows']:print(json.dumps({k:v for k,v in row.items() if k not in ('numeric_archive','numeric_sha256')},indent=2))
