"""Training-only cross-fitted linear value on cached actor features.

The actor never receives a clock. The optional finite-horizon factor belongs to
the value baseline only. No critic MLP, optimizer, simulator or evaluator imports.
"""
import time
import torch


class FeatureValueBaseline:
    def __init__(self,actor,steps,lanes,mode):
        if mode not in ['state','clock'] or steps<2 or lanes<4 or lanes%2:
            raise ValueError('Closed value mode and even complete episode batch required')
        last=list(actor.children())[-1]
        if not isinstance(last,torch.nn.Linear):raise ValueError('Expected existing actor MLP')
        self.steps,self.lanes,self.mode=steps,lanes,mode
        self.width=last.in_features+1
        self.cache=torch.empty((steps,lanes,last.in_features),device=last.weight.device,dtype=last.weight.dtype)
        self.active_step=None;self.seen=set();self.solves=0;self.actor_forward_rows=0
        self.hook=last.register_forward_pre_hook(self._capture)

    def _capture(self,module,inputs):
        values=inputs[0];self.actor_forward_rows+=len(values)
        if self.active_step is None:return
        step=self.active_step
        if type(step) is not int or not 0<=step<self.steps or step in self.seen:
            raise ValueError('Duplicate or invalid rollout feature capture')
        if values.shape!=self.cache[step].shape:raise ValueError('Rollout feature shape differs')
        self.cache[step].copy_(values.detach());self.seen.add(step)

    def begin(self):
        self.seen.clear();self.active_step=None

    def fit(self,returns,gamma=.9):
        started=time.perf_counter()
        if self.active_step is not None or len(self.seen)!=self.steps or returns.shape!=(self.steps,self.lanes):
            raise ValueError('Complete cached episode features required before value fit')
        if not 0<gamma<1 or not torch.isfinite(returns).all():raise ValueError('Finite return/discount required')
        features=self.cache.detach().to(device='cpu',dtype=torch.float64).reshape(-1,self.width-1)
        x=torch.cat([features,torch.ones((len(features),1),dtype=torch.float64,device='cpu')],axis=1)
        if self.mode=='clock':
            mass=1-gamma**torch.arange(self.steps,0,-1,dtype=torch.float64,device='cpu')
            x=x*mass.repeat_interleave(self.lanes)[:,None]
        y=returns.detach().to(device='cpu',dtype=torch.float64).flatten()
        parity=torch.arange(len(x),device='cpu')%self.lanes%2
        predicted=torch.empty_like(y);coefficients=[];ridges=[]
        for held in [0,1]:
            fit=parity!=held;test=~fit;design=x[fit];labels=y[fit]
            covariance=design.T@design/len(design)
            ridge=.001*covariance.trace()/self.width
            beta=torch.linalg.solve(covariance+ridge*torch.eye(self.width,dtype=torch.float64,device='cpu'),design.T@labels/len(design))
            predicted[test]=x[test]@beta;coefficients.append(beta);ridges.append(float(ridge));self.solves+=1
        if not torch.isfinite(predicted).all():raise ValueError('Nonfinite fitted value')
        error=(predicted-y).square()
        result=predicted.reshape(self.steps,self.lanes).to(device=returns.device,dtype=returns.dtype)
        return result,dict(mode=self.mode,crossfit_MSE=float(error.mean()),late10_MSE=float(error.reshape(self.steps,self.lanes)[-min(10,self.steps):].mean()),
            fit_seconds=time.perf_counter()-started,linear_solves_completed=self.solves,
            value_parameters_per_fold=self.width,ridges=ridges,actor_forward_rows=self.actor_forward_rows,
            covariance_flops_estimate=2*self.steps*self.lanes*self.width**2,
            scope='Whole-episode even/odd folds; own future labels excluded. Features cached during ordinary actor inference; clock used only by value baseline.'),torch.stack(coefficients)

    def close(self):self.hook.remove()
