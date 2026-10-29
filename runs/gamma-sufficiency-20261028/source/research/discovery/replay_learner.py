"""Matched replay critics for one current/next-moment mechanism study.

CrossQ-inspired PyTorch adaptation, NOT a verbatim official CrossQ baseline.
Author source: adityab/CrossQ a318a266679bec3f9d93c44e029860a6da077c89.
Actor stays within the frozen safe tanh schema; only critics are normalized.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .sac_baseline import Actor, Replay, transition_successor, export_actor
except ImportError:
    from sac_baseline import Actor, Replay, transition_successor, export_actor

METHODS=('target-sac','joint-moments','split-moments')


class BoundarySqrt(torch.autograd.Function):
    """Unchanged nonnegative sqrt values; define its zero-boundary derivative0.

    The literal PyTorch port otherwise produces0*infinity→NaN for constant
    features, including in the inactive BRN warmup expression. This is a
    documented common implementation safeguard, not a learning intervention.
    """
    @staticmethod
    def forward(ctx,x):
        root=x.sqrt();ctx.save_for_backward(root);return root

    @staticmethod
    def backward(ctx,gradient):
        root,=ctx.saved_tensors
        slope=torch.where(root>0,.5/root,torch.zeros_like(root))
        return gradient*slope


class BatchRenorm(nn.Module):
    """Pinned author's finite-epsilon custom-mean/custom-var BRN equations.

    Split control uses half-batch moments but the SAME pooled EMA update/count
    as joint control. Both run one concatenated critic pass. Thus running-stat
    clock/frequency is not the intervention. No updates in evaluation mode.
    """
    def __init__(self, width, momentum=.99, epsilon=.001, warmup=100000, split=False):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width));self.bias=nn.Parameter(torch.zeros(width))
        self.register_buffer('running_mean',torch.zeros(width))
        self.register_buffer('running_var',torch.ones(width))
        self.register_buffer('steps',torch.zeros((),dtype=torch.long))
        self.momentum,self.epsilon,self.warmup,self.split=momentum,epsilon,warmup,split

    @staticmethod
    def moments(x):
        mean=x.mean(0)
        variance=((x*x).mean(0)-mean.square()).clamp_min(0)
        return mean,variance

    def normalized(self,x,mean,var):
        old_mean,old_var=self.running_mean,self.running_var
        r=((var+self.epsilon).sqrt()/(old_var+self.epsilon).sqrt()).detach().clamp(1/3,3)
        d=((mean-old_mean)/(old_var+self.epsilon).sqrt()).detach().clamp(-5,5)
        renorm_var=var/r.square()
        renorm_mean=mean-d*BoundarySqrt.apply(var)/r
        warmed=(self.steps>=self.warmup).to(x.dtype)
        used_mean=warmed*renorm_mean+(1-warmed)*mean
        used_var=warmed*renorm_var+(1-warmed)*var
        return (x-used_mean)/(used_var+self.epsilon).sqrt()*self.weight+self.bias

    def forward(self,x):
        if not self.training:
            return (x-self.running_mean)/(self.running_var+self.epsilon).sqrt()*self.weight+self.bias
        pooled_mean,pooled_var=self.moments(x)
        if self.split:
            if len(x)%2:raise ValueError('Paired equal current/next halves required')
            a,b=x.chunk(2,dim=0)
            am,av=self.moments(a);bm,bv=self.moments(b)
            output=torch.cat((self.normalized(a,am,av),self.normalized(b,bm,bv)),dim=0)
        else:
            output=self.normalized(x,pooled_mean,pooled_var)
        with torch.no_grad():
            self.running_mean.copy_(self.momentum*self.running_mean+(1-self.momentum)*pooled_mean.detach())
            self.running_var.copy_(self.momentum*self.running_var+(1-self.momentum)*pooled_var.detach())
            self.steps.add_(1)
        return output


class Critic(nn.Module):
    def __init__(self,width=256,normalization=None):
        super().__init__()
        self.first=nn.Linear(56,width);self.second=nn.Linear(width,width);self.output=nn.Linear(width,1)
        def norm(features):
            return nn.Identity() if normalization is None else BatchRenorm(features,split=normalization=='split')
        self.input_norm=norm(56);self.first_norm=norm(width);self.second_norm=norm(width)

    def forward(self,obs,action):
        x=self.input_norm(torch.cat((obs,action),dim=-1))
        x=self.first_norm(F.relu(self.first(x)))
        x=self.second_norm(F.relu(self.second(x)))
        return self.output(x)


class Learner:
    def __init__(self,method,width=256,gamma=.99,learning_rate=.001,policy_delay=3,device='cpu'):
        if method not in METHODS:raise ValueError('Unregistered replay mechanism')
        self.method,self.gamma,self.policy_delay=method,gamma,policy_delay
        self.actor=Actor().to(device)
        normalization=None if method=='target-sac' else 'split' if method=='split-moments' else 'joint'
        self.q1,self.q2=Critic(width,normalization).to(device),Critic(width,normalization).to(device)
        self.target1=copy.deepcopy(self.q1).requires_grad_(False) if method=='target-sac' else None
        self.target2=copy.deepcopy(self.q2).requires_grad_(False) if method=='target-sac' else None
        self.q_optimizer=torch.optim.Adam([*self.q1.parameters(),*self.q2.parameters()],lr=learning_rate,betas=(.5,.999))
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=learning_rate,betas=(.5,.999))
        self.log_alpha=nn.Parameter(torch.zeros((),device=device))
        self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=learning_rate,betas=(.5,.999))
        self.updates,self.actor_updates=0,0
        self.work=dict(critic_forward_rows=0,actor_forward_rows=0,critic_sampled_rows=0,
                       actor_sampled_rows=0,critic_optimizer_calls=0,actor_optimizer_calls=0,alpha_optimizer_calls=0)

    def critic_target(self,data,next_action,next_logprob,alpha):
        self.q1.train();self.q2.train()
        if self.method=='target-sac':
            with torch.no_grad():
                future=torch.minimum(self.target1(data['next_obs'],next_action),self.target2(data['next_obs'],next_action))
            current1,current2=self.q1(data['obs'],data['action']),self.q2(data['obs'],data['action'])
        else:
            # Do not put the next-half critic pass in no_grad: current predictions
            # must retain gradients through the shared normalization moments.
            obs=torch.cat((data['obs'],data['next_obs']),0)
            action=torch.cat((data['action'],next_action),0)
            current1,future1=self.q1(obs,action).chunk(2,dim=0)
            current2,future2=self.q2(obs,action).chunk(2,dim=0)
            future=torch.minimum(future1,future2)
        target=(data['reward']+self.gamma*(1-data['stop'])*(future-alpha*next_logprob)).detach()
        return current1,current2,target

    def update(self,data):
        alpha=self.log_alpha.exp().detach()
        with torch.no_grad():next_action,next_logprob=self.actor.sample(data['next_obs'])
        q1,q2,target=self.critic_target(data,next_action,next_logprob,alpha)
        loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.q_optimizer.zero_grad(set_to_none=True);loss.backward();self.q_optimizer.step()
        self.updates+=1
        count=len(data['obs']);self.work['critic_forward_rows']+=4*count
        self.work['actor_forward_rows']+=count;self.work['critic_sampled_rows']+=count
        self.work['critic_optimizer_calls']+=1
        stats=dict(q_loss=loss.detach(),q_mean=((q1+q2)/2).mean().detach(),target_mean=target.mean(),
            target_abs_max=target.abs().max(),alpha=alpha)
        if self.updates%self.policy_delay==0:
            self.q1.eval();self.q2.eval();self.q1.requires_grad_(False);self.q2.requires_grad_(False)
            action,logprob=self.actor.sample(data['obs'])
            actor_loss=(alpha*logprob-torch.minimum(self.q1(data['obs'],action),self.q2(data['obs'],action))).mean()
            self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward();self.actor_optimizer.step()
            temperature_loss=-(self.log_alpha.exp()*(logprob.detach()-8)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True);temperature_loss.backward();self.alpha_optimizer.step()
            self.q1.requires_grad_(True);self.q2.requires_grad_(True)
            self.actor_updates+=1
            self.work['critic_forward_rows']+=2*count;self.work['actor_forward_rows']+=count
            self.work['actor_sampled_rows']+=count
            for key in ('actor_optimizer_calls','alpha_optimizer_calls'):self.work[key]+=1
            stats.update(actor_loss=actor_loss.detach(),temperature_loss=temperature_loss.detach(),
                         mean_logprob=logprob.detach().mean())
        if self.method=='target-sac':
            with torch.no_grad():
                for current,target_net in ((self.q1,self.target1),(self.q2,self.target2)):
                    for p,q in zip(current.parameters(),target_net.parameters()):q.lerp_(p,.005)
        return stats

    @torch.no_grad()
    def diagnostics(self,data):
        # Original mode and every normalization buffer remain intact.
        modes=[self.q1.training,self.q2.training]
        self.q1.eval();self.q2.eval()
        action,logprob=self.actor.sample(data['obs'])
        q1,q2=self.q1(data['obs'],action),self.q2(data['obs'],action)
        self.q1.train(modes[0]);self.q2.train(modes[1])
        self.work['critic_forward_rows']+=2*len(action);self.work['actor_forward_rows']+=len(action)
        return dict(q_policy_mean=float(torch.minimum(q1,q2).mean()),mean_logprob=float(logprob.mean()),
                    q_abs_max=float(torch.maximum(q1.abs(),q2.abs()).max()))
