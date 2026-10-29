"""Approved first-update mechanism probe; never imports the trusted evaluator.

Production is a hash-bound source, not a rewritten PPO comparator. Its literal
GAE/update blocks serve as the reference, including global gradient clipping.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

PRODUCTION_SHA = 'c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200'


@dataclass
class ProbeArgs:
    num_steps: int = 50
    num_envs: int = 1024
    batch_size: int = 51200
    minibatch_size: int = 1600
    update_epochs: int = 8
    gamma: float = .9
    gae_lambda: float = .9
    finite_horizon_gae: bool = False
    norm_adv: bool = True
    clip_coef: float = .2
    clip_vloss: bool = False
    target_kl: float = .1
    ent_coef: float = .003
    vf_coef: float = .5
    max_grad_norm: float = .5
    learning_rate: float = .0003


def load_production(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRODUCTION_SHA:
        raise ValueError('Production source digest differs; stop probe')
    source = raw.decode()
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in ('Agent', 'layer_init')]
    ns = dict(torch=torch, nn=nn, np=np, Normal=Normal)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'hash-bound-production-agent', 'exec'), ns)
    start = source.index('        # bootstrap value according to termination and truncation\n')
    end = source.index('        # flatten the batch\n', start)
    gae = compile(textwrap.dedent(source[start:end]), 'hash-bound-production-gae', 'exec')
    start = source.index('        # Optimizing the policy and value network\n')
    end = source.index('        update_time = time.time() - update_time\n', start)
    update = compile(textwrap.dedent(source[start:end]), 'hash-bound-production-update', 'exec')
    return ns['Agent'], gae, update


def new_agent(agent_class, device='cpu'):
    spaces = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)),
                             single_action_space=SimpleNamespace(shape=(8,)))
    return agent_class(spaces).to(device)


def optimizer(agent):
    return torch.optim.Adam(agent.parameters(), lr=.0003, eps=1e-5)


@torch.no_grad()
def reconstructed_gae(agent, args, data):
    rewards, values, dones = (data[k] for k in ('rewards', 'values', 'dones'))
    advantages = torch.zeros_like(rewards)
    following = agent.get_value(data['next_obs']).reshape(1, -1)
    last = 0
    for t in reversed(range(args.num_steps)):
        alive = 1.0 - (data['next_done'] if t == args.num_steps-1 else dones[t+1])
        next_v = following if t == args.num_steps-1 else values[t+1]
        actual_v = alive * next_v + data['final_values'][t]
        if args.finite_horizon_gae:
            if t == args.num_steps-1:
                lam_sum, reward_sum, value_sum = 0., 0., 0.
            lam_sum = lam_sum * alive
            reward_sum = reward_sum * alive
            value_sum = value_sum * alive
            lam_sum = 1 + args.gae_lambda * lam_sum
            reward_sum = args.gae_lambda * args.gamma * reward_sum + lam_sum * rewards[t]
            value_sum = args.gae_lambda * args.gamma * value_sum + args.gamma * actual_v
            advantages[t] = (reward_sum + value_sum) / lam_sum - values[t]
        else:
            delta = rewards[t] + args.gamma * actual_v - values[t]
            advantages[t] = last = delta + args.gamma * args.gae_lambda * alive * last
    return advantages, advantages + values


def reconstructed_step(agent, opt, args, batch):
    agent.train()
    indices = np.arange(args.batch_size)
    np.random.shuffle(indices)
    mb = indices[:args.minibatch_size]
    _, logprob, entropy, value = agent.get_action_and_value(batch['b_obs'][mb], batch['b_actions'][mb])
    logratio = logprob - batch['b_logprobs'][mb]
    ratio = logratio.exp()
    with torch.no_grad():
        kl = ((ratio-1)-logratio).mean()
    if args.target_kl is not None and kl > args.target_kl:
        raise RuntimeError('First update rejected by production KL; stop')
    advantage = batch['b_advantages'][mb]
    if args.norm_adv:
        advantage = (advantage-advantage.mean()) / (advantage.std()+1e-8)
    pg = torch.max(-advantage*ratio,
                   -advantage*torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)).mean()
    value = value.view(-1)
    if args.clip_vloss:
        old = batch['b_values'][mb]
        clipped = old + torch.clamp(value-old, -args.clip_coef, args.clip_coef)
        vloss = .5*torch.max((value-batch['b_returns'][mb])**2,
                            (clipped-batch['b_returns'][mb])**2).mean()
    else:
        vloss = .5*((value-batch['b_returns'][mb])**2).mean()
    loss = pg-args.ent_coef*entropy.mean()+vloss*args.vf_coef
    opt.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    opt.step()
    return dict(mb_indices=mb, loss=float(loss.detach()), pre_update_approx_kl=float(kl),
                unclipped_global_gradient_norm=float(grad_norm))


def assert_same(a, b, label='root'):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise AssertionError('Production parity mismatch: '+label)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError('Production parity mismatch: '+label)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise AssertionError('State keys differ: '+label)
        for key in a: assert_same(a[key], b[key], label+'.'+str(key))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise AssertionError('State length differs: '+label)
        for i, (x,y) in enumerate(zip(a,b)): assert_same(x,y,label+'.'+str(i))
    elif a != b:
        raise AssertionError('Production parity mismatch: '+label)


def check_reconstruction(agent, args, data, gae_code, update_code):
    """Exact reference vs independent calculation on identical inputs/RNG.

    The reference executes the unmodified production update block; the optimizer
    is stopped immediately AFTER its first actual step. No second step occurs.
    """
    device = next(agent.parameters()).device
    ns = dict(torch=torch, nn=nn, np=np, time=time, args=args, agent=agent,
              device=device, **data)
    exec(gae_code, ns)
    advantage, returns = reconstructed_gae(agent, args, data)
    assert_same(ns['advantages'], advantage, 'advantages')
    assert_same(ns['returns'], returns, 'returns')
    batch = dict(b_obs=data['obs'].reshape(-1,48), b_actions=data['actions'].reshape(-1,8),
                 b_logprobs=data['logprobs'].reshape(-1), b_values=data['values'].reshape(-1),
                 b_advantages=advantage.reshape(-1), b_returns=returns.reshape(-1))
    initial_np = copy.deepcopy(np.random.get_state())
    initial_cpu = torch.get_rng_state().clone()
    initial_cuda = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    reference, proposed = copy.deepcopy(agent), copy.deepcopy(agent)
    refopt, propopt = optimizer(reference), optimizer(proposed)
    real_step = refopt.step
    class FirstUpdateComplete(Exception): pass
    def step_and_stop(*a, **kw):
        result = real_step(*a, **kw)
        raise FirstUpdateComplete()
    refopt.step = step_and_stop
    refns = dict(torch=torch, nn=nn, np=np, time=time, args=args, agent=reference,
                 optimizer=refopt, logger=SimpleNamespace(add_scalar=lambda *a: None),
                 global_step=args.batch_size, automanip_optimizer_steps=0, **batch)
    try:
        exec(update_code, refns)
    except FirstUpdateComplete:
        pass
    else:
        raise AssertionError('Production did not execute its first update')
    ref_np, ref_cpu = copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    ref_cuda = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    refopt.step = real_step
    np.random.set_state(initial_np)
    torch.set_rng_state(initial_cpu)
    if initial_cuda is not None: torch.cuda.set_rng_state_all(initial_cuda)
    stats = reconstructed_step(proposed, propopt, args, batch)
    assert_same(refns['mb_inds'], stats['mb_indices'], 'first minibatch')
    assert_same(reference.state_dict(), proposed.state_dict(), 'parameters')
    assert_same(refopt.state_dict(), propopt.state_dict(), 'Adam moments and steps')
    assert_same(ref_np, np.random.get_state(), 'numpy RNG')
    assert_same(ref_cpu, torch.get_rng_state(), 'CPU RNG')
    if ref_cuda is not None: assert_same(ref_cuda, torch.cuda.get_rng_state_all(), 'CUDA RNG')
    for (name,x), (_,y) in zip(reference.named_parameters(), proposed.named_parameters()):
        assert_same(x.grad, y.grad, 'clipped gradient '+name)
    np.random.set_state(initial_np)
    torch.set_rng_state(initial_cpu)
    if initial_cuda is not None: torch.cuda.set_rng_state_all(initial_cuda)
    return proposed, batch, stats, dict(passed=True, bitwise_parameters=True,
        bitwise_adam_state=True, bitwise_advantages=True, bitwise_returns=True,
        bitwise_gradients=True, bitwise_rng=True,
        rng_devices_checked=['numpy','torch_cpu']+(['torch_cuda'] if device.type=='cuda' else []),
        reference_optimizer_calls=1, reconstruction_optimizer_calls=1,
        fitting_rows=args.minibatch_size, gae_forward_rows=2*args.num_envs,
        production_source_sha256=PRODUCTION_SHA)


@torch.no_grad()
def interpolate(start, end, fraction):
    branch = copy.deepcopy(start)
    if fraction == 0:
        return branch
    for p, a, b in zip(branch.actor_mean.parameters(), start.actor_mean.parameters(), end.actor_mean.parameters()):
        p.copy_(b if fraction == 1 else a+fraction*(b-a))
    assert_same(branch.critic.state_dict(), start.critic.state_dict(), 'fixed critic')
    assert_same(branch.actor_logstd, start.actor_logstd, 'fixed logstd')
    return branch


@torch.no_grad()
def mean_kl(start, end, states):
    before, after = start.actor_mean(states), end.actor_mean(states)
    per_state = .5*((after-before)/start.actor_logstd.exp()).square().sum(-1)
    return dict(mean=float(per_state.mean()), p95=float(torch.quantile(per_state,.95)),
                maximum=float(per_state.max()), kl_kind='mean_only_fixed_original_std',
                forward=float(per_state.mean()), reverse=float(per_state.mean()),
                mean_action_squared_change=float((after-before).square().mean()),
                clipped_mean_action_squared_change=float((after.clamp(-1,1)-before.clamp(-1,1)).square().mean()))


def choose_branches(start, end, states):
    checks=[]
    for k in range(13):
        fraction=2.**(-k)
        branch=interpolate(start,end,fraction)
        checks.append(dict(fraction=fraction,**mean_kl(start,branch,states)))
    choices={}
    for name, cap in [('bounded',.01),('tiny_forward',.001)]:
        passing=[r for r in checks if r['mean']<=cap]
        if not passing: raise RuntimeError('No allowed grid fraction meets '+name+' KL cap')
        choices[name]=passing[0]['fraction']
    choices=dict(no_update=0.,full=1.,**choices,tiny_reverse=-choices['tiny_forward'])
    return {name:interpolate(start,end,f) for name,f in choices.items()}, choices, checks


def kl_holdout(batch, mb, args):
    used=np.unique(mb % args.num_envs)
    eligible=np.setdiff1d(np.arange(args.num_envs),used)
    if len(eligible)<64: raise RuntimeError('Fewer than64 disjoint KL-check episodes; stop')
    episodes=eligible[:64]
    ids=np.concatenate([episodes+t*args.num_envs for t in range(args.num_steps)])
    return batch['b_obs'][ids], episodes


def cpu_preflight(source):
    started=time.monotonic()
    torch.set_num_threads(2)
    cls, gae, update=load_production(source)
    torch.manual_seed(11031);np.random.seed(11031)
    agent=new_agent(cls)
    with torch.no_grad(): agent.actor_logstd.fill_(-2.4)
    args=ProbeArgs()
    g=torch.Generator().manual_seed(11032)
    obs=torch.randn(50,1024,48,generator=g)
    actions=torch.randn(50,1024,8,generator=g)*.1
    actions[0,0]=torch.tensor([-2.,-1.,-.999,0.,.999,1.,2.,3.])
    with torch.no_grad():
        _,logprobs,_,values=agent.get_action_and_value(obs.reshape(-1,48),actions.reshape(-1,8))
    data=dict(obs=obs,actions=actions,logprobs=logprobs.reshape(50,1024),values=values.reshape(50,1024),
              rewards=torch.rand(50,1024,generator=g),dones=torch.zeros(50,1024),
              next_done=torch.ones(1024),next_obs=torch.randn(1024,48,generator=g),final_values=torch.zeros(50,1024))
    data['dones'][20,:32]=1
    data['final_values'][19,:32]=.3
    data['final_values'][-1]=.7
    reports=[]
    for finite in (False,True):
        args.finite_horizon_gae=finite
        end,batch,stats,report=check_reconstruction(agent,args,data,gae,update)
        states, episodes=kl_holdout(batch,stats['mb_indices'],args)
        branches,choices,checks=choose_branches(agent,end,states[:128])
        assert_same(branches['no_update'].state_dict(),agent.state_dict(),'alpha0')
        assert_same(branches['full'].actor_mean.state_dict(),end.actor_mean.state_dict(),'alpha1 mean')
        analytic=mean_kl(agent,end,states[:128])['mean']
        with torch.no_grad():
            p=Normal(agent.actor_mean(states[:128]),agent.actor_logstd.exp())
            q=Normal(end.actor_mean(states[:128]),agent.actor_logstd.exp())
            library=float(torch.distributions.kl_divergence(p,q).sum(-1).mean())
        if abs(analytic-library)>1e-5: raise AssertionError('Analytic/library Gaussian KL disagree')
        bounded_actions=actions.clamp(-1,1)
        assert_same(bounded_actions,torch.clamp(actions.detach(),torch.full((8,),-1.),torch.ones(8)),'executed clipping')
        reports.append(dict(finite_horizon_gae=finite,**report,branch_fractions=choices,analytic_kl=analytic,
                            disjoint_holdout_episodes=len(episodes)))
    seconds=time.monotonic()-started
    if seconds>300: raise RuntimeError('CPU reconstruction exceeded five minutes')
    return dict(state='passed',cpu_seconds=seconds,new_simulator_transitions=0,checks=reports,
                optimizer_calls=4,production_args_finite_horizon_gae=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--production',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args()
    result=cpu_preflight(a.production)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
