"""Read-only Gaussian actuator-support diagnostic on existing training states."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def support_statistics(mean,std,mask):
    if not mask.any():return dict(states=0)
    mean=np.asarray(mean)[mask];std=float(std)
    if not np.isfinite(mean).all() or not np.isfinite(std) or std<=0:raise ValueError('Finite Gaussian parameters required')
    result=dict(states=len(mean),raw_mean_quantiles=np.quantile(mean,[0,.25,.5,.75,1]).tolist(),std=std)
    for threshold,name in [(-1.,'escape_lower_clamp'),(0.,'positive_normalized_command'),(.28,'target_finger_position_above_022m')]:
        logp=torch.special.log_ndtr(torch.from_numpy((mean-threshold)/std)).numpy()
        maximum=float(logp.max());logmean=maximum+float(np.log(np.exp(logp-maximum).mean()))
        result[name]=dict(log10_mean_probability=logmean/np.log(10),log10_median_probability=float(np.median(logp))/np.log(10),
                          median_probability=float(np.exp(np.median(logp))))
    return result


def run(root,output,plan):
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.numpy_inference import NumpyPolicy
    from automanip.stackcube import STACK_TASK
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False,parents=True);started=time.monotonic()
    result=dict(state='started',cells=[],new_environment_transitions=0,optimizer_calls=0,actor_forward_rows=0)
    for path,digest in plan['inputs'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Source/input drift')
    compute=json.loads((root/plan['compute']).read_text())
    for cell in plan['cells']:
        index,checkpoint=cell['index'],cell['checkpoint'];method=['ppo-gae','ppo-finite-mc'][index%2];seed=1+index//2
        folder=root/plan['training_root']/f'trial-{index}/work/runs/scratch-credit-{method}-seed-{seed}'
        with np.load(folder/f'trace-{checkpoint+51200}.npz',allow_pickle=False) as z:obs=z['observations'].copy().reshape(-1,48)
        schema=PolicySchema(STACK_TASK.fingerprint,48,8);policy=NumpyPolicy(load_policy(folder/'actors'/str(checkpoint),schema),schema)
        x=obs.astype(float)
        for i,(weight,bias) in enumerate(policy.layers):
            x=x@weight.T+bias
            if i<len(policy.layers)-1:x=np.tanh(x)
        result['actor_forward_rows']+=len(obs)
        scalars=compute['trials'][index]['scalars'];std=np.exp(next(r['value'] for r in scalars['audit/logstd_dim_7'] if r['steps']==checkpoint))
        a,b=obs[:,25:28],obs[:,32:35]
        top=(np.linalg.norm(a[:,:2]-b[:,:2],axis=1)<=np.sqrt(2)*.02+.005)&(np.abs(a[:,2]-b[:,2]-.04)<=.005)
        narrow=obs[:,7:9].sum(1)<=.042
        result['cells'].append(dict(index=index,method=method,seed=seed,checkpoint=checkpoint,
            all_states=support_statistics(x[:,7],std,np.ones(len(obs),dtype=bool)),
            on_top_geometry=support_statistics(x[:,7],std,top),
            on_top_and_narrow_gripper=support_statistics(x[:,7],std,top&narrow),
            scope='Pre-action observations only, no terminal obs. Narrow width is not a measured contact/grasp flag. Command probability is not a probability of successful release.'))
    selected=[c for c in result['cells'] if c['index'] in [2,4] and c['checkpoint']==8192000]
    result['registered_saturation_expectation_met']=all(c['on_top_and_narrow_gripper']['states']>=20 and c['on_top_and_narrow_gripper']['escape_lower_clamp']['median_probability']<.001 for c in selected)
    result.update(state='complete',wall_seconds=time.monotonic()-started)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=run(a.root,a.output,json.loads(Path(a.plan).read_text()));print(json.dumps({k:v for k,v in r.items() if k!='cells'}))
