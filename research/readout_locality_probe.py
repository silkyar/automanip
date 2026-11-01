"""CPU mechanism gate on archived first-update data; no simulator or scoring.

Tests whether head-only updates preserve old late-episode behavior beyond what
is explained by a smaller update on the current visitation distribution.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.discovery.first_update_probe import (ProbeArgs,load_production,new_agent,
    check_reconstruction,reconstructed_step,optimizer,kl_holdout,interpolate)
from research.readout_reuse import configure_actor


def recovered_shuffle(state,expected,batch_size):
    for position in [624,*range(624)]:
        candidate=('MT19937',state,position,0,0.)
        np.random.set_state(candidate);indices=np.arange(batch_size);np.random.shuffle(indices)
        if np.array_equal(indices[:len(expected)],expected):return candidate,position+2 if position!=624 else 1
    raise ValueError('Saved NumPy state cannot reconstruct original fitting indices')


@torch.no_grad()
def kl_stats(start,end,states):
    before=start.actor_mean(states);after=end.actor_mean(states)
    values=.5*((after-before)/start.actor_logstd.exp()).square().sum(-1)
    return dict(mean=float(values.mean()),p95=float(torch.quantile(values,.95)),maximum=float(values.max()),
        clipped_action_mse=float((after.clamp(-1,1)-before.clamp(-1,1)).square().mean()))


@torch.no_grad()
def matched_branch(start,end,states,target):
    before=start.actor_mean(states);scale=start.actor_logstd.exp();lo,hi=0.,1.
    for _ in range(24):
        mid=(lo+hi)/2;branch=interpolate(start,end,mid)
        value=float((.5*((branch.actor_mean(states)-before)/scale).square().sum(-1)).mean())
        if value>target:hi=mid
        else:lo=mid
    branch=interpolate(start,end,(lo+hi)/2)
    return branch,(lo+hi)/2


def run(root,output,plan):
    started=time.monotonic();root=Path(root);output=Path(output);output.mkdir(exist_ok=False)
    result=dict(state='started',new_environment_transitions=0,gpu_work=0,optimizer_calls=0,optimizer_calls_reserved=0,rows=[],
        scope='Archived assisted first-update geometry only; neither robot retention nor learning efficiency')
    work=dict(actor_forward_rows=0,actor_forward_calls=0,critic_forward_rows=0,critic_forward_calls=0)
    result['forward_work']=work
    def persist():
        result['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    try:
        for name,digest in plan['inputs'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Registered input differs: '+name)
        torch.set_num_threads(2);torch.manual_seed(93490000)
        cls,gae,update=load_production(root/plan['production'])
        args=ProbeArgs()
        demo_indices=json.loads((root/plan['bc_report']).read_text())['train_episode_indices'][:64]
        with np.load(root/plan['prior_demonstrations'],allow_pickle=False) as saved:
            far=torch.from_numpy(saved['observations'][demo_indices,30:50].copy().reshape(-1,48))
        result['prior_reference']=dict(training_episode_indices=demo_indices,time_indices=[30,50],rows=len(far),use='Diagnostics only; no gradients or actor fitting on these states')
        for seed in plan['collection_seeds']:
            folder=root/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
            with np.load(folder/'update-state.npz',allow_pickle=False) as saved:
                initial_state={k[len('initial_'):]:torch.from_numpy(saved[k].copy()) for k in saved.files if k.startswith('initial_')}
                historical={k[len('joint_update_'):]:saved[k].copy() for k in saved.files if k.startswith('joint_update_')}
                expected_indices=saved['fitting_indices'].copy();old_adv=saved['advantages'].copy();old_returns=saved['returns'].copy()
            initial=new_agent(cls);initial.load_state_dict(initial_state,strict=True)
            def actor_hook(module,inputs,outputs):
                work['actor_forward_rows']+=len(inputs[0]);work['actor_forward_calls']+=1
            def critic_hook(module,inputs,outputs):
                work['critic_forward_rows']+=len(inputs[0]);work['critic_forward_calls']+=1
            initial.actor_mean.register_forward_hook(actor_hook)
            initial.critic.register_forward_hook(critic_hook)
            if any(not torch.isfinite(v).all() for v in initial_state.values()):raise ValueError('Nonfinite initial parameters')
            with np.load(folder/'collection.npz',allow_pickle=False) as saved:
                data={k:torch.from_numpy(saved[k].copy()) for k in ('obs','actions','rewards','values','logprobs','dones','final_values','next_obs','next_done')}
            with np.load(folder/'rng-before-update.npz',allow_pickle=False) as saved:
                rng,searches=recovered_shuffle(saved['numpy_state'].copy(),expected_indices,args.batch_size)
            np.random.set_state(rng)
            result['optimizer_calls_reserved']+=2;persist()
            full,batch,stats,parity=check_reconstruction(initial,args,data,gae,update)
            result['optimizer_calls']+=2
            if not np.array_equal(stats['mb_indices'],expected_indices):raise ValueError('First minibatch differs')
            adv_error=float(np.max(np.abs(batch['b_advantages'].numpy()-old_adv)))
            returns_error=float(np.max(np.abs(batch['b_returns'].numpy()-old_returns)))
            historical_error=max(float(np.max(np.abs(v.detach().numpy()-historical[k]))) for k,v in full.state_dict().items())
            row=dict(collection_seed=seed,production_CPU_parity=parity,numpy_shuffle_searches=searches,
                     saved_advantage_max_error=adv_error,saved_return_max_error=returns_error,
                     archived_GPU_parameter_max_error=historical_error)
            result['rows'].append(row);persist()
            if max(adv_error,returns_error)>plan['saved_target_tolerance'] or historical_error>plan['archived_GPU_parameter_tolerance']:
                raise ValueError('CPU/archived-production reconstruction tolerance failed; no mechanism conclusion')
            head=copy.deepcopy(initial);row['readout']=configure_actor(head.actor_mean,'readout')
            np.random.set_state(rng);result['optimizer_calls_reserved']+=1;persist()
            head_stats=reconstructed_step(head,optimizer(head),args,batch);result['optimizer_calls']+=1
            if not np.array_equal(head_stats['mb_indices'],expected_indices):raise ValueError('Readout minibatch differs')
            states,episodes=kl_holdout(batch,expected_indices,args)
            full_kl=kl_stats(initial,full,states);head_kl=kl_stats(initial,head,states)
            target=min(plan['current_KL_cap'],full_kl['mean'],head_kl['mean'])
            if target<1e-6:raise ValueError('No material actor displacement to compare')
            full_branch,full_fraction=matched_branch(initial,full,states,target)
            head_branch,head_fraction=matched_branch(initial,head,states,target)
            full_current=kl_stats(initial,full_branch,states);head_current=kl_stats(initial,head_branch,states)
            if max(abs(full_current['mean']-target),abs(head_current['mean']-target))>target*.01:raise ValueError('Current-distribution KL not matched within 1%')
            full_far=kl_stats(initial,full_branch,far);head_far=kl_stats(initial,head_branch,far)
            ratio=head_far['mean']/max(full_far['mean'],1e-12)
            row.update(current_KL_target=target,current_holdout_episodes=episodes.tolist(),
                raw_full_KL=full_kl,raw_readout_KL=head_kl,
                full_fraction=full_fraction,readout_fraction=head_fraction,
                full_current_KL=full_current,readout_current_KL=head_current,
                full_prior_late_KL=full_far,readout_prior_late_KL=head_far,
                prior_late_KL_ratio=ratio,per_collection_locality_gate=ratio<=.8)
            np.savez_compressed(output/(str(seed)+'-branches.npz'),
                **{'full_'+k:v.detach().numpy() for k,v in full_branch.state_dict().items()},
                **{'readout_'+k:v.detach().numpy() for k,v in head_branch.state_dict().items()})
            persist()
        ratios=[r['prior_late_KL_ratio'] for r in result['rows']]
        result['mean_prior_late_KL_ratio']=float(np.mean(ratios))
        result['gate_passed']=sum(v<=.8 for v in ratios)>=2 and np.mean(ratios)<=.8
        result['state']='complete';result['decision']='Eligible only for a fresh public paired direction probe' if result['gate_passed'] else 'Reject hidden-feature freezing as a locality mechanism under this registered test; no robot follow-up'
    except Exception as exc:
        import traceback
        result.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.root,a.output,json.loads(Path(a.plan).read_text()))
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','traceback','prior_reference')}))
    raise SystemExit(0 if r['state']=='complete' else 1)
