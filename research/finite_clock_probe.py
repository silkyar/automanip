"""Remaining-time baseline diagnostic on existing training subsamples only.

No policy, critic or simulator is trained. Fitted linear control variates are
cross-fitted by whole episodes. Their score moments are not PPO performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.crossfit_value_probe import crossfit_baselines, discounted_returns


def horizon_mass(horizon,gamma=.9):
    if type(horizon) is not int or horizon<1 or not 0<gamma<1:raise ValueError('Positive horizon and discount required')
    return 1-gamma**np.arange(horizon,0,-1,dtype=float)


def summarize_geometry(observations,rewards):
    if observations.shape!=(50,64,48) or rewards.shape!=(50,64):raise ValueError('Closed blind training trace shape')
    tcp=observations[:,:,18:21];a=observations[:,:,25:28];b=observations[:,:,32:35]
    error=max(np.abs(a-tcp-observations[:,:,39:42]).max(),np.abs(b-tcp-observations[:,:,42:45]).max(),np.abs(b-a-observations[:,:,45:48]).max())
    if error>1e-5:raise ValueError('Observation field layout/redundancy differs')
    distance=np.linalg.norm(a[:,:,:2]-b[:,:,:2],axis=-1)
    lift=a[:,:,2]>=a[0,:,2]+.03
    transport=lift & ((distance[0]-distance)>=.03)
    top=(distance<=np.sqrt(2)*.02+.005)&(np.abs(a[:,:,2]-b[:,:,2]-.04)<=.005)
    # Geometric states stop BEFORE the last action; no fabricated terminal obs.
    return dict(observation_redundancy_max_error=float(error),
                ever_lift_before_last_action=float(lift.any(0).mean()),
                ever_transport_before_last_action=float(transport.any(0).mean()),
                ever_on_top_geometry_before_last_action=float(top.any(0).mean()),
                final_pre_action_on_top_geometry=float(top[-1].mean()),
                mean_return=float(rewards.sum(0).mean()),
                ever_grasp_reward_branch=float(((rewards>=.5-1e-6)&(rewards<=.625+1e-6)).any(0).mean()),
                ever_on_top_or_success_reward_branch=float((rewards>.7).any(0).mean()),
                ever_stock_success_reward=float((rewards>=.999999).any(0).mean()),
                scope='Geometric pre-action states and unchanged reward branches; no inferred contact/grasp flag on on-top frames, no inferred true static flag, not independent evaluation.')


def raw_features(policy,observations):
    x=np.asarray(observations,dtype=float)
    for weight,bias in policy.layers[:-1]:x=np.tanh(x@weight.T+bias)
    phi=np.concatenate([x,np.ones((len(x),1))],axis=1)
    weight,bias=policy.layers[-1]
    return phi,x@weight.T+bias


def run(root,output,plan):
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.numpy_inference import NumpyPolicy
    from automanip.stackcube import STACK_TASK
    root=Path(root);output=Path(output);output.mkdir(exist_ok=False,parents=True)
    started=time.monotonic();torch.set_num_threads(2)
    result=dict(state='started',cells=[],new_environment_transitions=0,optimizer_calls=0,
                baseline_linear_solves=0,baseline_fitting_rows=0,actor_forward_rows=0,
                CPU_versions=dict(numpy=np.__version__,torch=torch.__version__))
    def persist():
        result['wall_seconds']=time.monotonic()-started
        p=output/'report.tmp';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(output/'report.json')
    try:
        for path,digest in plan['inputs'].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Registered data/source drift: '+path)
        compute=json.loads((root/plan['compute']).read_text())
        for cell in plan['cells']:
            index=cell['index'];checkpoint=cell['checkpoint'];seed=1+index//2
            method=['ppo-gae','ppo-finite-mc'][index%2]
            folder=root/plan['training_root']/f'trial-{index}/work/runs/scratch-credit-{method}-seed-{seed}'
            with np.load(folder/f'trace-{checkpoint+51200}.npz',allow_pickle=False) as z:
                observations=z['observations'].copy();actions=z['raw_actions'].copy();rewards=z['rewards'].copy();values=z['values'].copy()
                if (z['dones'][1:]!=0).any() or not (z['next_done']==1).all():raise ValueError('Aligned full episodes required')
            schema=PolicySchema(STACK_TASK.fingerprint,48,8)
            policy=NumpyPolicy(load_policy(folder/'actors'/str(checkpoint),schema),schema)
            phi,mean=raw_features(policy,observations.reshape(-1,48));result['actor_forward_rows']+=len(phi)
            scalars=compute['trials'][index]['scalars']
            std=np.exp([next(r['value'] for r in scalars[f'audit/logstd_dim_{a}'] if r['steps']==checkpoint) for a in range(8)])
            epsilon=(actions.reshape(-1,8)-mean)/std
            noise_mean=epsilon.mean(0);noise_std=epsilon.std(0)
            if np.abs(noise_mean).max()>.15 or (noise_std<.8).any() or (noise_std>1.2).any():raise ValueError('Checkpoint/noise alignment failed')
            weight=((epsilon/std)**2).sum(1)*((phi**2).sum(1))
            returns=discounted_returns(torch.from_numpy(rewards),.9).flatten()
            mass=np.repeat(horizon_mass(50),64)
            permutation=np.random.default_rng(plan['clock_permutation_seed']).permutation(50)
            false_mass=np.repeat(horizon_mass(50)[permutation],64)
            calibration=np.column_stack([values.reshape(-1),np.ones(len(phi))])
            designs={'calibrated_value':calibration,'clock_calibrated_value':calibration*mass[:,None],
                     'actor_state':phi,'clock_actor_state':phi*mass[:,None],
                     'permuted_clock_actor_state':phi*false_mass[:,None],
                     'permuted_clock_calibrated_value':calibration*false_mass[:,None]}
            predictions={'raw_value':values.reshape(-1).astype(float)}
            row=dict(index=index,method=method,seed=seed,checkpoint=checkpoint,episodes=64,
                     noise_mean=noise_mean.tolist(),noise_std=noise_std.tolist(),geometry=summarize_geometry(observations,rewards),models={})
            for name,design in designs.items():
                fitted=crossfit_baselines(torch.from_numpy(design),returns,64,.001)
                predictions[name]=fitted['state'].numpy()
                result['baseline_linear_solves']+=2;result['baseline_fitting_rows']+=len(phi)
            gradients={}
            for name,prediction in predictions.items():
                residual=returns.numpy()-prediction
                mean_gradient=np.einsum('tna,tnd,tn->nad',(epsilon/std).reshape(50,64,8),phi.reshape(50,64,-1),residual.reshape(50,64),optimize=True)
                std_gradient=np.einsum('tna,tn->na',(epsilon**2-1).reshape(50,64,8),residual.reshape(50,64),optimize=True)
                gradient=np.column_stack([mean_gradient.reshape(64,-1),std_gradient])
                # Conditional on each opposite-fold fitted baseline, held lanes
                # are independent trajectories. Sum time BEFORE covariance.
                trajectory_variance=np.mean([gradient[fold::2].var(0,ddof=1).sum() for fold in [0,1]])
                gradients['gradient_'+name]=gradient
                row['models'][name]=dict(MSE=float(np.mean(residual**2)),
                    mean_readout_score_second_moment=float(np.mean(weight*residual**2)),
                    late10_score_second_moment=float(np.mean((weight*residual**2).reshape(50,64)[40:])),
                    within_fold_trajectory_readout_logstd_gradient_variance=float(trajectory_variance))
            for a,b,label in [('clock_actor_state','actor_state','actor_features'),('clock_calibrated_value','calibrated_value','actual_value_calibration')]:
                row[label+'_clock_moment_ratio']=row['models'][a]['mean_readout_score_second_moment']/row['models'][b]['mean_readout_score_second_moment']
            variance_key='within_fold_trajectory_readout_logstd_gradient_variance'
            row['clock_to_state_trajectory_variance_ratio']=row['models']['clock_actor_state'][variance_key]/row['models']['actor_state'][variance_key]
            row['clock_to_permuted_trajectory_variance_ratio']=row['models']['clock_actor_state'][variance_key]/row['models']['permuted_clock_actor_state'][variance_key]
            np.savez_compressed(output/f'cell-{index}-{checkpoint}.npz',returns=returns.numpy(),score_weight=weight,**predictions,**gradients)
            result['cells'].append(row);persist()
        ratios=[c['actor_features_clock_moment_ratio'] for c in result['cells']]
        result['registered_clock_expectation_met']=sum(r<=.75 for r in ratios)>=9
        result['mean_clock_moment_ratio']=float(np.mean(ratios))
        result['registered_temporal_clock_expectation_met']=sum(c['clock_to_state_trajectory_variance_ratio']<=.75 and c['clock_to_permuted_trajectory_variance_ratio']<=.75 for c in result['cells'])>=9
        result['state']='complete'
        result['interpretation']='Conditional baseline information/partial-score variance diagnostic only; no production PPO gradient reconstruction, independent-return gain, or automatic GPU admission.'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=run(a.root,a.output,json.loads(Path(a.plan).read_text()));print(json.dumps({k:v for k,v in r.items() if k not in ['cells','traceback']}));raise SystemExit(0 if r['state']=='complete' else 1)
