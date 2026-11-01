"""Manual2x2post-opening maintenance diagnostic, never a learned policy."""
import numpy as np
from research.proxy_calibration_probe import PublicDiagnosticPolicy

BRANCHES=('resume_actor','keep_open','zero_arm','keep_open_zero_arm')


class ReleaseMaintenancePolicy(PublicDiagnosticPolicy):
    def __init__(self,base,std,assignment):
        assignment=np.asarray(assignment)
        if assignment.shape!=(64,) or sorted(assignment.tolist())!=sorted(list(range(4))*16):raise ValueError('Exactly16lanes per maintenance arm required')
        self.assignment=assignment.copy();self.lanes=64;super().__init__(base,std)

    def reset_noise(self,seed):
        super().reset_noise(seed);self.step_index=0
        self.trigger=np.full(64,-1,dtype=int);self.applied=np.zeros(64,dtype=int)
        self.first_eligible=np.full(64,-1,dtype=int);self.fallback=np.zeros(64,dtype=bool)
        self.override_masks=[];self.arm_masks=[];self.maintenance_counts=np.zeros((64,2),dtype=int)

    def act(self,observations):
        obs=np.asarray(observations,dtype=float)
        if obs.shape!=(64,48):raise ValueError('Fixed diagnostic batch required')
        actions=np.asarray(super().act(observations),dtype=float)
        a,b=obs[:,25:28],obs[:,32:35]
        eligible=(np.linalg.norm(a[:,:2]-b[:,:2],axis=1)<=np.sqrt(2)*.02+.005)&(np.abs(a[:,2]-b[:,2]-.04)<=.005)&(obs[:,7:9].sum(1)<=.042)
        self.first_eligible[(self.first_eligible<0)&eligible]=self.step_index
        ready=(self.trigger<0)&(eligible|(self.step_index==45))
        self.trigger[ready]=self.step_index;self.fallback[ready]=~eligible[ready]
        common=(self.trigger>=0)&(self.step_index<self.trigger+5)
        maintenance=(self.trigger>=0)&(self.step_index>=self.trigger+5)
        keep=maintenance&((self.assignment%2)==1);hold=maintenance&(self.assignment>=2)
        actions[common|keep,7]=1.;actions[hold,:7]=0.
        self.applied[common]+=1;self.override_masks.append((common|keep).copy());self.arm_masks.append(hold.copy())
        self.maintenance_counts[:,0]+=keep;self.maintenance_counts[:,1]+=hold;self.step_index+=1
        return actions.tolist()
