"""Finite-horizon regression of continuation value on CURRENT features.

Training uses realized future cashflows. Execution must see only the current
state/features/payoff; taking a maximum over future realized states is invalid.
"""
import numpy as np


def fit_continuation(features,exercise,eligible,terminal,*,ridge=.001,margin=.01):
    x=np.asarray(features,dtype=float);h=np.asarray(exercise,dtype=float);ok=np.asarray(eligible)
    cash=np.asarray(terminal,dtype=float).copy()
    if x.ndim!=3 or not 1<=x.shape[0]<=50 or x.shape[1]<2 or not 1<=x.shape[2]<=257 or h.shape!=x.shape[:2] or ok.shape!=h.shape or ok.dtype!=bool or cash.shape!=(x.shape[1],):raise ValueError('Invalid stoppingdata shapes')
    if not all(np.isfinite(a).all() for a in [x,h,cash]) or (h<0).any() or (h>1).any() or (cash<0).any() or (cash>1).any() or ridge<=0 or not 0<=margin<1:raise ValueError('Invalid stoppingvalues')
    coefficients=np.zeros((x.shape[0],x.shape[2]))
    for t in reversed(range(x.shape[0])):
        gram=x[t].T@x[t];scale=np.trace(gram)/len(gram)
        if scale<=0:raise ValueError('Nonzerodesignrequired')
        coefficients[t]=np.linalg.solve(gram+ridge*scale*np.eye(len(gram)),x[t].T@cash)
        continuation=np.clip(x[t]@coefficients[t],0,1)
        stop=ok[t]&(h[t]>continuation+margin)
        cash[stop]=h[t,stop]
    return dict(coefficients=coefficients,training_cashflows=cash,linear_solves=x.shape[0],margin=margin)


def current_decision(features,exercise,eligible,coefficients,margin=.01):
    x=np.asarray(features,dtype=float);h=np.asarray(exercise,dtype=float);ok=np.asarray(eligible,dtype=bool)
    if x.ndim!=2 or h.shape!=(len(x),) or ok.shape!=h.shape or np.asarray(coefficients).shape!=(x.shape[1],):raise ValueError('Currentfeaturesonlyrequired')
    continuation=np.clip(x@coefficients,0,1)
    return ok&(h>continuation+margin)


def evaluate_paths(features,exercise,eligible,terminal,model):
    x,h,ok=map(np.asarray,[features,exercise,eligible]);cash=np.asarray(terminal,dtype=float).copy();alive=np.ones(len(cash),bool);times=np.full(len(cash),-1,int)
    for t in range(len(x)):
        stop=alive&current_decision(x[t],h[t],ok[t],model['coefficients'][t],model['margin'])
        cash[stop]=h[t,stop];times[stop]=t;alive[stop]=False
    return cash,times
