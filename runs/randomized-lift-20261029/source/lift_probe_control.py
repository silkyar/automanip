"""Bounded analytic motion intervention, structural assistance not a learner."""
import numpy as np


def rotation_error(target,current):
    r=target@current.transpose(0,2,1)
    vee=np.stack((r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]),-1)/2
    sine=np.linalg.norm(vee,axis=-1);cosine=(np.trace(r,axis1=1,axis2=2)-1)/2
    angle=np.arctan2(sine,cosine)
    if np.any((sine<1e-8)&(angle>1.)):raise ValueError('Orientation correction near pi is ambiguous')
    factor=np.divide(angle,sine,out=np.ones_like(angle),where=sine>1e-8)
    return vee*factor[:,None]


def cap_vectors(x,maximum):
    return x*np.minimum(1.,maximum/np.maximum(np.linalg.norm(x,axis=-1),1e-30))[:,None]


def upward_action(q,transform,jacobian,target,joint_limits):
    translation=cap_vectors(target[:,:3,3]-transform[:,:3,3],.015)
    angular=cap_vectors(rotation_error(target[:,:3,:3],transform[:,:3,:3]),.05)
    wanted=np.concatenate([translation,angular],axis=-1)
    dq=(jacobian.transpose(0,2,1)@np.linalg.solve(
        jacobian@jacobian.transpose(0,2,1)+.05**2*np.eye(6),wanted[...,None]))[...,0]
    joint_target=np.clip(q+dq,joint_limits[:,0]+.01,joint_limits[:,1]-.01)
    action=np.clip((joint_target-q)/.1,-1,1)
    if not np.isfinite(action).all():raise ValueError('Nonfinite assisted controller action')
    return action,wanted
