"""Describe frozen actor commands at achieved-maintenance states; no learning."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy
from automanip.stackcube import STACK_TASK
from research.finite_clock_probe import raw_features


def clipped_moments(mean,std):
    m=torch.as_tensor(mean,dtype=torch.float64);s=torch.as_tensor(std,dtype=torch.float64)
    low=(-1-m)/s;high=(1-m)/s
    cdf=lambda x:torch.special.ndtr(x)
    pdf=lambda x:torch.exp(-x*x/2)/np.sqrt(2*np.pi)
    left=cdf(low);right=cdf(-high);inside=1-left-right
    first=m*inside+s*(pdf(low)-pdf(high))-left+right
    second=(m*m+s*s)*inside+s*((m-1)*pdf(low)-(m+1)*pdf(high))+left+right
    return first.numpy(),second.numpy()


def main():
    started=time.monotonic();torch.set_num_threads(2)
    root=Path('artifacts/cumulative/release-maintenance-20261030');plan=json.loads((root/'provenance/inputs/plan.json').read_text());rows=[];forwards=0
    for index,seed in enumerate([2,3]):
        selected=[]
        for batch in range(32):
            with np.load(root/f'remote/collection-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                time_index=np.arange(1,50)[:,None]
                mask=(z['assignment'][None,:]==3)&(time_index>=z['trigger'][None,:]+5)&z['stage_flags'][:49,:,6].astype(bool)
                selected.append(z['observations'][1:50][mask])
        observations=np.concatenate(selected)
        if not len(observations):raise ValueError('No achieved-maintenance states')
        schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        actor=NumpyPolicy(load_policy(root/'provenance/inputs'/plan['policies'][str(seed)],schema),schema)
        _,mean=raw_features(actor,observations);forwards+=len(observations)
        std=np.asarray(plan['gaussian_std'][str(seed)])
        expected,second=clipped_moments(mean,std)
        variance=np.maximum(second-expected**2,0.)
        closing_probability=torch.special.ndtr(torch.from_numpy((0-mean[:,7])/std[7])).numpy()
        opening_probability=torch.special.ndtr(torch.from_numpy((mean[:,7]-.28)/std[7])).numpy()
        rows.append(dict(seed=seed,selected_pre_action_states=len(observations),deterministic_arm_command_RMS=float(np.sqrt(np.mean(np.clip(mean[:,:7],-1,1)**2))),mean_squared_expected_arm_command=float(np.mean(expected[:,:7]**2)),mean_arm_command_variance=float(np.mean(variance[:,:7])),expected_motion_fraction=float(np.mean(expected[:,:7]**2)/np.mean(second[:,:7])),mean_expected_gripper_command=float(expected[:,7].mean()),median_probability_original_gripper_closes=float(np.median(closing_probability)),median_probability_original_gripper_opens_above_point28=float(np.median(opening_probability))))
    result=dict(state='complete',rows=rows,policy_forward_rows=forwards,optimizer_calls=0,fitting_operations=0,physics_calls=0,elapsed_seconds=time.monotonic()-started,scope='Conditional descriptive inference at public diagnostic states; no physical counterfactual outcome or learnedpolicy claim; diagnostic data never learnerinput')
    output=Path('artifacts/cumulative/maintenance-command-audit-v1');output.mkdir(exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'plan.json').write_bytes(Path('cumulative/experiments/maintenance-command-audit-v1.json').read_bytes());(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
