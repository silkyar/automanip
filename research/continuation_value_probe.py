"""Learn option exercise and continuation values from existing training only."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy
from research.correction_support_probe import features
from research.least_squares_stopping import fit_continuation,evaluate_paths


def exercise_features(phi,time_index):
    phi=np.asarray(phi,dtype=float);t=np.asarray(time_index,dtype=float)
    if phi.shape[-1]!=257 or t.shape!=phi.shape[:-1] or not np.isfinite(phi).all() or not np.isfinite(t).all() or (t<0).any() or (t>=50).any():raise ValueError('Current257featuresandvalidclockrequired')
    remaining=(50-t)/50
    return np.concatenate([phi,remaining[...,None],remaining[...,None]**2],axis=-1)


def main():
    started=time.monotonic();root=Path('artifacts/cumulative/random-switch-training-20261030');plan_path=Path('cumulative/experiments/continuation-value-cpu-v1.json');registration=json.loads(plan_path.read_text())
    for path,digest in registration['inputs'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Training/helperpin drift')
    plan=json.loads((root/'provenance/inputs/plan.json').read_text());output=Path('artifacts/cumulative/continuation-value-cpu-v1');output.mkdir(exist_ok=False);rows=[];forwards=0;solves=0
    projection=np.linalg.qr(np.random.default_rng(registration['projection_seed']).standard_normal((256,16)))[0]
    for index,source in enumerate(plan['sources']):
        schema=PolicySchema(plan['task_fingerprint'],48,8);actor=NumpyPolicy(load_policy(root/'provenance/inputs'/source['policy'],schema),schema)
        decision_x=[];decision_y=[];decision_part=[];base_obs=[];base_terminal=[];base_part=[]
        for batch in range(32):
            with np.load(root/f'remote/source-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                part=['development','calibration','fit','fit'][batch%4]
                hold=z['treatment']==1;phi=features(actor,z['state_at_decision'][hold]);forwards+=len(phi)
                decision_x.append(exercise_features(phi,z['decision'][hold]));decision_y.append(z['outcome'][hold].astype(float));decision_part.extend([part]*int(hold.sum()))
                baseline=~hold;base_obs.append(z['observations'][:50,baseline].copy());base_terminal.append(z['outcome'][baseline].astype(float));base_part.extend([part]*int(baseline.sum()))
        x,y=np.concatenate(decision_x),np.concatenate(decision_y);part=np.asarray(decision_part);fit=part=='fit';cal=part=='calibration';dev=part=='development'
        gram=x[fit].T@x[fit];precision=gram+.001*np.trace(gram)/259*np.eye(259)
        coefficient=np.linalg.solve(precision,x[fit].T@y[fit]);solves+=1
        metric=np.linalg.solve(precision,np.eye(259));metric=(metric+metric.T)/2
        leverage=np.sum((x@metric)*x,axis=1);cutoff=float(np.quantile(leverage[cal],.95))
        dev_score=np.clip(x[dev]@coefficient,0,1)
        observations=np.concatenate(base_obs,axis=1);terminal=np.concatenate(base_terminal);base_part=np.asarray(base_part)
        phi=features(actor,observations.reshape(-1,48)).reshape(50,len(terminal),257);forwards+=50*len(terminal)
        times=np.broadcast_to(np.arange(50)[:,None],phi.shape[:2]);design=exercise_features(phi,times)
        payoff=np.clip(design@coefficient,0,1);eligible=np.sum((design@metric)*design,axis=-1)<=cutoff
        state_features=np.concatenate([phi[:,:,:256]@projection,np.ones((*phi.shape[:2],1))],axis=-1)
        time_features=np.ones((*phi.shape[:2],1))
        base_fit=base_part=='fit';base_dev=base_part=='development';models={};model_statistics={}
        for name,representation in [('time',time_features),('state',state_features)]:
            model=fit_continuation(representation[:,base_fit],payoff[:,base_fit],eligible[:,base_fit],terminal[base_fit]);solves+=model['linear_solves'];models[name]=model
            estimated,timing=evaluate_paths(representation[:,base_dev],payoff[:,base_dev],eligible[:,base_dev],terminal[base_dev],model)
            model_statistics[name]=dict(fit_paths=int(base_fit.sum()),development_paths=int(base_dev.sum()),DEVELOPMENT_MODEL_IMPLIED_payoff=float(estimated.mean()),development_trigger_count=int((timing>=0).sum()),development_median_trigger=None if not (timing>=0).any() else float(np.median(timing[timing>=0])))
        np.savez_compressed(output/f'models-source-{index}.npz',exercise_coefficient=coefficient,support_metric=metric,support_cutoff=np.asarray(cutoff),projection=projection,time_continuation=models['time']['coefficients'],state_continuation=models['state']['coefficients'])
        rows.append(dict(seed=source['training_seed'],fit_exercise_episodes=int(fit.sum()),internal_calibration_exercise_episodes=int(cal.sum()),development_exercise_episodes=int(dev.sum()),development_Brier=float(np.mean((dev_score-y[dev])**2)),support_cutoff=cutoff,initial_supported_baseline_states=int(eligible[0].sum()),continuation_models=model_statistics))
    result=dict(state='complete',rows=rows,source_actor_feature_forward_rows=forwards,linear_solves=solves,new_robot_transitions=0,optimizer_calls=0,elapsed_seconds=time.monotonic()-started,scope='Existingtrainingdataonly. Developmentmodel-impliedcashflowsareNOTactualrobotreturnsanddonotestablishpolicyimprovement. Freshindependentphysicsstillrequired.')
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'stopping-helper.py').write_bytes(Path('research/least_squares_stopping.py').read_bytes());(output/'plan.json').write_bytes(plan_path.read_bytes());(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
