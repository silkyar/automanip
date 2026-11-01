"""Closed public probe of learned selection, with an explicit fixed action prior.

This composite policy is not a candidate in the frozen MLP artifact protocol.
It loads bounded numeric outcome/support metadata, never candidate code.
"""
import numpy as np

BRANCHES=('base','random_switch','learned_raw','learned_supported')


class LearnedCorrectionPolicy:
    def __init__(self,base,std,assignment,head,timing_seed):
        self.base=base;self.std=np.asarray(std,dtype=float);self.assignment=np.asarray(assignment).copy()
        if self.std.shape!=(8,) or not np.isfinite(self.std).all() or (self.std<=0).any() or (self.std>1).any():raise ValueError('Invalid source std')
        if self.assignment.shape!=(64,) or sorted(self.assignment.tolist())!=sorted(list(range(4))*16):raise ValueError('Balanced4armassignmentrequired')
        if set(head)!={'coefficients','support_metric','support_cutoff'}:raise ValueError('Closednumericheadschema')
        self.coefficients=np.asarray(head['coefficients'],float);self.metric=np.asarray(head['support_metric'],float);self.cutoff=float(head['support_cutoff'])
        if self.coefficients.shape!=(2,257) or self.metric.shape!=(257,257) or not np.isfinite(self.coefficients).all() or not np.isfinite(self.metric).all() or not np.isfinite(self.cutoff) or not 0<self.cutoff<1000 or np.abs(self.coefficients).max()>1e6 or np.abs(self.metric).max()>1e12:raise ValueError('Invalid boundednumerichead')
        if np.abs(self.metric-self.metric.T).max()>1e-6:raise ValueError('Supportmetricasymmetry')
        np.linalg.cholesky(self.metric)
        self.random_time=np.random.default_rng(timing_seed).integers(10,45,64);self.reset_noise(0)

    def reset_noise(self,seed):
        self.rng=np.random.default_rng(seed);self.step_index=0;self.trigger=np.full(64,-1,int);self.applied=np.zeros(64,int)
        self.override_masks=[];self.scores=[];self.leverages=[]

    def act(self,observations):
        x=np.asarray(observations,dtype=float)
        if x.shape!=(64,48) or not np.isfinite(x).all():raise ValueError('Invalid48stateinput')
        for w,b in self.base.layers[:-1]:x=np.tanh(x@w.T+b)
        phi=np.column_stack([x,np.ones(64)])
        if phi.shape!=(64,257):raise ValueError('Fixedparentrepresentationrequired')
        w,b=self.base.layers[-1];mean=x@w.T+b
        actions=np.clip(mean+self.std*self.rng.standard_normal((64,8)),-1,1)
        score=np.clip(phi@self.coefficients.T,0,1);leverage=np.sum((phi@self.metric)*phi,axis=1)
        favorable=(score[:,1]>=.8)&(score[:,1]-score[:,0]>=.1)
        ready=((self.assignment==1)&(self.step_index>=self.random_time))|((self.assignment==2)&favorable)|((self.assignment==3)&favorable&(leverage<=self.cutoff))
        self.trigger[(self.trigger<0)&ready]=self.step_index
        active=self.trigger>=0;actions[active,:7]=0;actions[active,7]=1
        self.applied[active]+=1;self.override_masks.append(active.copy());self.scores.append(score);self.leverages.append(leverage);self.step_index+=1
        return actions.tolist()
