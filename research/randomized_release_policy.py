"""Diagnostic-only opening interventions, randomized by lane before rollout."""
import numpy as np
from research.proxy_calibration_probe import PublicDiagnosticPolicy

BRANCHES=('no_update','scheduled_open','targeted_open','targeted_open_zero_arm')


class RandomizedReleasePolicy(PublicDiagnosticPolicy):
    def __init__(self,base,std,assignment):
        assignment=np.asarray(assignment)
        if assignment.shape!=(64,) or sorted(assignment.tolist())!=sorted(list(range(4))*16):raise ValueError('Exactly16lanes per registered arm required')
        self.assignment=assignment.copy();self.lanes=64
        super().__init__(base,std)

    def reset_noise(self,seed):
        super().reset_noise(seed);self.step_index=0
        self.trigger=np.full(self.lanes,-1,dtype=int);self.applied=np.zeros(self.lanes,dtype=int)
        self.first_eligible=np.full(self.lanes,-1,dtype=int);self.fallback=np.zeros(self.lanes,dtype=bool);self.override_masks=[]

    def act(self,observations):
        obs=np.asarray(observations,dtype=float)
        if obs.shape!=(64,48):raise ValueError('Fixed64lane diagnostic batch required')
        actions=np.asarray(super().act(observations),dtype=float)
        a,b=obs[:,25:28],obs[:,32:35]
        top=(np.linalg.norm(a[:,:2]-b[:,:2],axis=1)<=np.sqrt(2)*.02+.005)&(np.abs(a[:,2]-b[:,2]-.04)<=.005)
        eligible=top&(obs[:,7:9].sum(1)<=.042)
        self.first_eligible[(self.first_eligible<0)&eligible]=self.step_index
        if self.step_index==25:self.trigger[self.assignment==1]=25
        ready=(self.assignment>=2)&(self.trigger<0)&(eligible|(self.step_index==45))
        self.trigger[ready]=self.step_index;self.fallback[ready]=~eligible[ready]
        override=(self.trigger>=0)&(self.step_index>=self.trigger)&(self.step_index<self.trigger+5)
        actions[override,7]=1.
        actions[override&(self.assignment==3),:7]=0.
        self.applied[override]+=1;self.override_masks.append(override.copy());self.step_index+=1
        return actions.tolist()
