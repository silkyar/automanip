"""Read numeric archived snapshots; no fitting, simulator or evaluator access.

Post-hoc diagnostics are not targets for tuning or measurements of true robot Q.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.discovery.replay_learner import Actor,Critic


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(module,arrays,prefix):
    state=module.state_dict()
    for key,tensor in state.items():
        value=arrays[prefix+'.'+key]
        if tuple(value.shape)!=tuple(tensor.shape):raise ValueError('Snapshot shape differs')
        state[key]=torch.from_numpy(value.copy()).to(tensor.dtype)
    module.load_state_dict(state)
    return module


def main():
    started=time.monotonic();torch.set_num_threads(1)
    archive=ROOT/'runs/stackcube-replay-moments-screen-20261028'
    summary=json.loads((archive/'trusted-summary.json').read_bytes())
    assert summary['state']=='complete'
    rows=[];sources={};total_rows=0
    for trial in summary['training']:
        method=trial['method'];report=trial['report'];directory=archive/method/method/'run'
        for point in report['checkpoint_exports']:
            label=point['label'];trajectory=directory/(label+'-trajectory.npz')
            if not trajectory.exists():continue
            statepath=directory/(label+'-state.npz')
            sources[str(trajectory.relative_to(ROOT))]=digest(trajectory)
            sources[str(statepath.relative_to(ROOT))]=digest(statepath)
            normalization=None if method=='target-sac' else 'joint' if method=='joint-moments' else 'split'
            with np.load(statepath,allow_pickle=False) as state:
                actor=restore(Actor(),state,'actor').eval()
                q1=restore(Critic(256,normalization),state,'q1').eval()
                q2=restore(Critic(256,normalization),state,'q2').eval()
                alpha=float(np.exp(state['log_alpha']))
            with np.load(trajectory,allow_pickle=False) as trace:
                obs=torch.from_numpy(trace['obs'].reshape(-1,48).copy())
                next_obs=torch.from_numpy(trace['next_obs'].reshape(-1,48).copy())
                behavior_action=torch.from_numpy(trace['action'].reshape(-1,8).copy())
                reward=trace['reward'].copy();flags=trace['stage_flags'].copy()
                truncated=trace['truncated'].copy()
                complete=point['transitions']%(64*50)==0
                if complete:
                    assert not truncated[:-1].any() and truncated[-1].all()
                total_rows+=len(obs)
                with torch.no_grad():
                    torch.manual_seed(20261028)
                    action,logprob=actor.sample(obs)
                    next_action,_=actor.sample(next_obs)
                    running1=q1(obs,behavior_action);running2=q2(obs,behavior_action)
                    running_min=torch.minimum(running1,running2)
                    train1=copy.deepcopy(q1).train();train2=copy.deepcopy(q2).train()
                    both_obs=torch.cat([obs,next_obs]);both_actions=torch.cat([behavior_action,next_action])
                    train_current1,train_next1=train1(both_obs,both_actions).chunk(2)
                    train_current2,train_next2=train2(both_obs,both_actions).chunk(2)
                    train_current_min=torch.minimum(train_current1,train_current2)
                    state_std=actor(obs)[1].exp()
                    deterministic=actor.deterministic(obs)
                    policy_value=torch.minimum(q1(obs,deterministic),q2(obs,deterministic))
                # Upper bound on exact Q for a FIXED present alpha, reward<=1,
                # action support volume2**8, gamma=.99. Not a lower bound or
                # an estimate of historical targets with changing alpha.
                bound=1+.99*(1+alpha*8*math.log(2))/(1-.99)
                variable_action=deterministic.detach().requires_grad_(True)
                value=torch.minimum(q1(obs,variable_action),q2(obs,variable_action))
                gradient,=torch.autograd.grad(value.sum(),variable_action)
                row=dict(method=method,label=label,transitions=point['transitions'],
                    critic_updates=point['critic_updates'],alpha=alpha,
                    sample_rows=len(obs),complete_50_step_episode_batch=complete,
                    recent_reward_mean=float(reward.mean()),
                    behavior_action_abs_gt_095_fraction=float((behavior_action.abs()>.95).float().mean()),
                    any_stage_counts_64_windows=flags.any(0).sum(0).tolist(),
                    final_stage_counts_64_windows=flags[-1].sum(0).tolist(),
                    current_actor_sample_entropy_estimate=float(-logprob.mean()),
                    actor_logstd_exp_mean=float(state_std.mean()),
                    q_running_behavior_mean=float(running_min.mean()),
                    q_running_policy_mean=float(policy_value.mean()),
                    q_training_context_behavior_mean=float(train_current_min.mean()),
                    q_training_vs_running_rmse=float((train_current_min-running_min).square().mean().sqrt()),
                    per_head_training_current_next_mean_gaps=[float(train_current1.mean()-train_next1.mean()),float(train_current2.mean()-train_next2.mean())],
                    min_q_training_current_next_mean_gap=float(train_current_min.mean()-torch.minimum(train_next1,train_next2).mean()),
                    fixed_present_alpha_exact_soft_q_upper_bound=bound,
                    fraction_running_behavior_min_q_above_bound=float((running_min>bound).float().mean()),
                    fraction_running_policy_min_q_above_bound=float((policy_value>bound).float().mean()),
                    running_policy_action_gradient_mean_l2=float(gradient.norm(dim=-1).mean()),
                    running_policy_action_gradient_max_l2=float(gradient.norm(dim=-1).max()))
                rows.append(row)
    result=dict(state='complete',post_hoc=True,cpu_process_seconds=time.monotonic()-started,
        source_sha256=digest(Path(__file__)),snapshot_sources_sha256=sources,
        robot_transitions=0,optimizer_calls=0,protected_evaluations=0,
        existing_transition_rows_inspected=total_rows,records=rows,
        caveats=[
            'Recent behavior changes during collection; not frozen-policy trajectories.',
            'Final target/joint snapshots cross resets: no complete-episode return claim.',
            'Context is the saved recent3200 transitions, not the historical uniform replay minibatch.',
            'Running-versus-training discrepancy is not true-Q error; no robot value oracle exists.',
            'No target1/target2/optimizer/full replay archive: exact historical target-SAC step cannot be reconstructed.',
            'Entropy estimate is one current-policy sample per saved state with declared CPU RNG, not historical behavior entropy.',
            'Soft-Q bound is a necessary fixed-present-alpha consistency condition, not a historical-target reconstruction.',
            'Action gradients are measured scales, not measured correctness or causal explanation.'
        ])
    target=ROOT/'research/discovery/replay-screen-snapshot-diagnostics-20261028.json'
    if target.exists():raise FileExistsError('Preserve diagnostic receipt; use a new version')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'seconds':result['cpu_process_seconds'],'records':len(rows),'path':str(target)}))


if __name__=='__main__':main()
