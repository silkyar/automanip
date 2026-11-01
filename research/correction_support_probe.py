"""Action-conditional support diagnostic; no outcomes used in cutoff selection."""
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy


def features(actor,obs):
    x=np.asarray(obs,dtype=float)
    for w,b in actor.layers[:-1]:x=np.tanh(x@w.T+b)
    return np.column_stack([x,np.ones(len(x))])


def main():
    started=time.monotonic();root=Path('artifacts/cumulative/random-switch-training-20261030');plan=json.loads((root/'provenance/inputs/plan.json').read_text());rows=[];forwards=0
    output=Path('artifacts/cumulative/correction-support-cpu-v1');output.mkdir(exist_ok=False)
    for index,source in enumerate(plan['sources']):
        schema=PolicySchema(plan['task_fingerprint'],48,8);actor=NumpyPolicy(load_policy(root/'provenance/inputs'/source['policy'],schema),schema)
        fit=[];cal=[];initial=[]
        for batch in range(32):
            with np.load(root/f'remote/source-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                initial.append(features(actor,z['observations'][0]));forwards+=64
                if batch%4 in [1,2,3]:
                    selected=z['treatment']==1;x=features(actor,z['state_at_decision'][selected]);forwards+=len(x)
                    if batch%4==1:cal.append(x)
                    else:fit.append(x)
        fit,cal,initial=map(np.concatenate,[fit,cal,initial]);gram=fit.T@fit;ridge=.001*np.trace(gram)/257;precision=gram+ridge*np.eye(257)
        rhs=np.concatenate([cal,initial]);solved=np.linalg.solve(precision,rhs.T).T;leverage=np.sum(rhs*solved,axis=1)
        lc,li=leverage[:len(cal)],leverage[len(cal):];cutoff=float(np.quantile(lc,.95))
        coefficients=np.load(root/f'outcome-models-source-{index}.npz',allow_pickle=False)['coefficients'];score=np.clip(initial@coefficients.T,0,1);raw=(score[:,1]>=.8)&(score[:,1]-score[:,0]>=.1)
        rows.append(dict(seed=source['training_seed'],fit_corrective_episodes=len(fit),calibration_corrective_episodes=len(cal),reset_states=len(initial),calibration_leverage_quantiles=np.quantile(lc,[0,.5,.95,1]).tolist(),reset_leverage_quantiles=np.quantile(li,[0,.5,.95,1]).tolist(),cutoff=cutoff,initial_raw_gate_count=int(raw.sum()),initial_supported_gate_count=int((raw&(li<=cutoff)).sum())))
        np.savez_compressed(output/f'support-source-{index}.npz',precision=precision,cutoff=np.asarray(cutoff),calibration_leverage=lc,reset_leverage=li)
    result=dict(state='complete',rows=rows,source_actor_forward_rows=forwards,linear_system_solves=2,physics_calls=0,optimizer_calls=0,elapsed_seconds=time.monotonic()-started,scope='Unlabeledsupportheuristiconly; notareliableprobabilityboundorclosedlooppolicytest')
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'plan.json').write_bytes(Path('cumulative/experiments/correction-support-cpu-v1.json').read_bytes());(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
