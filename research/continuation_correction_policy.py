"""Fixed numeric optimal-stopping prototype, outside the frozen MLP protocol."""
import numpy as np

BRANCHES=('learned_fixed_time','fixed_threshold','time_continuation','state_continuation')


class ContinuationCorrectionPolicy:
    def __init__(self,base,std,assignment,head,timing_seed):
        self.base=base;self.std=np.asarray(std,float);self.assignment=np.asarray(assignment).copy()
        if self.std.shape!=(8,) or not np.isfinite(self.std).all() or (self.std<=0).any() or (self.std>1).any() or self.assignment.shape!=(64,) or sorted(self.assignment.tolist())!=sorted(list(range(4))*16):raise ValueError('Invalid source/randomization')
        required={'exercise_coefficient','support_metric','support_cutoff','projection','time_continuation','state_continuation','fixed_time'}
        if set(head)!=required:raise ValueError('Closed stoppingmodel schema')
        shapes={'exercise_coefficient':(259,),'support_metric':(259,259),'projection':(256,16),'time_continuation':(50,1),'state_continuation':(50,17)}
        self.head={}
        for key,shape in shapes.items():
            value=np.asarray(head[key],float)
            if value.shape!=shape or not np.isfinite(value).all() or np.abs(value).max()>1e12:raise ValueError('Invalid numeric stoppingmodel')
            self.head[key]=value
        self.cutoff=float(head['support_cutoff'])
        if not np.isfinite(self.cutoff) or not 0<self.cutoff<1000 or np.abs(self.head['support_metric']-self.head['support_metric'].T).max()>1e-6:raise ValueError('Invalid supportmetadata')
        np.linalg.cholesky(self.head['support_metric'])
        if type(head['fixed_time']) is not int or not 10<=head['fixed_time']<=44:raise ValueError('Invalid learnedfixedtime')
        self.scheduled_time=np.full(64,head['fixed_time'],int);self.reset_noise(0)

    def reset_noise(self,seed):
        self.rng=np.random.default_rng(seed);self.step_index=0;self.trigger=np.full(64,-1,int);self.applied=np.zeros(64,int);self.override_masks=[];self.scores=[];self.leverages=[]

    def act(self,observations):
        x=np.asarray(observations,float)
        if x.shape!=(64,48) or not np.isfinite(x).all() or not 0<=self.step_index<50:raise ValueError('Current48state/fixedhorizonrequired')
        for w,b in self.base.layers[:-1]:x=np.tanh(x@w.T+b)
        phi=np.column_stack([x,np.ones(64)])
        if phi.shape!=(64,257):raise ValueError('Parentrepresentation differs')
        remaining=(50-self.step_index)/50;design=np.column_stack([phi,np.full(64,remaining),np.full(64,remaining**2)])
        exercise=np.clip(design@self.head['exercise_coefficient'],0,1)
        leverage=np.sum((design@self.head['support_metric'])*design,axis=1);eligible=leverage<=self.cutoff
        time_value=np.full(64,float(np.clip(self.head['time_continuation'][self.step_index,0],0,1)))
        state_features=np.column_stack([x@self.head['projection'],np.ones(64)])
        state_value=np.clip(state_features@self.head['state_continuation'][self.step_index],0,1)
        ready=((self.assignment==0)&(self.step_index>=self.scheduled_time))|((self.assignment==1)&eligible&(exercise>=.8))|((self.assignment==2)&eligible&(exercise>time_value+.01))|((self.assignment==3)&eligible&(exercise>state_value+.01))
        self.trigger[(self.trigger<0)&ready]=self.step_index;active=self.trigger>=0
        w,b=self.base.layers[-1];actions=np.clip(x@w.T+b+self.std*self.rng.standard_normal((64,8)),-1,1)
        actions[active,:7]=0;actions[active,7]=1
        self.applied[active]+=1;self.override_masks.append(active.copy());self.scores.append(np.column_stack([exercise,time_value,state_value]));self.leverages.append(leverage);self.step_index+=1
        return actions.tolist()
