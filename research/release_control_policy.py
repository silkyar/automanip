"""Hand-specified causal intervention; never a benchmark candidate policy.

Inputs remain the existing48observations. Each intervention branch makes exactly
five gripper commands; target branches fall back to45..49 if never eligible.
"""
import numpy as np
from research.proxy_calibration_probe import PublicDiagnosticPolicy


class ReleaseControlPolicy(PublicDiagnosticPolicy):
    branches=('no_update','scheduled_open','targeted_open','targeted_open_zero_arm')
    def __init__(self,base,std,branch,lanes=64):
        if branch not in self.branches:raise ValueError('Closed diagnostic branch')
        self.branch,self.lanes=branch,lanes;self.reference=None
        super().__init__(base,std)

    def reset_noise(self,seed):
        super().reset_noise(seed)
        self.step_index=0;self.first_eligible=np.full(self.lanes,-1,dtype=int)
        self.trigger=np.full(self.lanes,-1,dtype=int);self.applied=np.zeros(self.lanes,dtype=int)
        self.fallback=np.zeros(self.lanes,dtype=bool);self.override_masks=[];self.max_prefix_difference=0.

    def act(self,observations):
        obs=np.asarray(observations,dtype=float)
        if obs.shape!=(self.lanes,48):raise ValueError('Fixed diagnostic observation batch required')
        if self.reference is not None:
            unchanged=self.applied==0
            if unchanged.any():
                difference=float(np.abs(obs[unchanged]-self.reference[self.step_index,unchanged]).max())
                self.max_prefix_difference=max(self.max_prefix_difference,difference)
                if difference>1e-5:raise ValueError('Pre-intervention paired trajectory mismatch')
        actions=np.asarray(super().act(observations),dtype=float)
        a,b=obs[:,25:28],obs[:,32:35]
        top=(np.linalg.norm(a[:,:2]-b[:,:2],axis=1)<=np.sqrt(2)*.02+.005)&(np.abs(a[:,2]-b[:,2]-.04)<=.005)
        eligible=top&(obs[:,7:9].sum(1)<=.042)
        self.first_eligible[(self.first_eligible<0)&eligible]=self.step_index
        if self.branch=='scheduled_open':
            if self.step_index==25:self.trigger[:]=25
        elif self.branch!='no_update':
            ready=(self.trigger<0)&(eligible|(self.step_index==45))
            self.trigger[ready]=self.step_index
            self.fallback[ready]=~eligible[ready]
        override=(self.trigger>=0)&(self.step_index>=self.trigger)&(self.step_index<self.trigger+5)
        actions[override,7]=1.
        if self.branch=='targeted_open_zero_arm':actions[override,:7]=0.
        self.applied[override]+=1;self.override_masks.append(override.copy());self.step_index+=1
        return actions.tolist()
