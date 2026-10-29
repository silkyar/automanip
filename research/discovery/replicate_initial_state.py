"""Diagnostic-only replication via pinned public API. Never imported by trainers."""
import hashlib
from pathlib import Path
import torch

PINNED_APIS={
 'mani_skill.envs.sapien_env':'996d1aaff9cfed36e4753ab271206aeeb8c42dc585835342d88a2457d2fb5334',
 'mani_skill.envs.scene':'fc4de7f3a4a61a5a02ef14861c5dfb3db73ee36442e788525d8963e2cf05e160',
 'mani_skill.agents.controllers.pd_joint_pos':'474d3ce4bac54cc3b006899070795956135a595b5573b5afb84aa3e732a103d5',
 'mani_skill.agents.controllers.base_controller':'43e1c18b0791caabd4c261d5f163c9c3de812ab9a7947f61e64494848c32e1e4'}


def repeat_first(state,batch):
    if not isinstance(state,dict):raise ValueError('Nested state dictionaries required')
    result={}
    for key,value in state.items():
        if isinstance(value,dict):result[key]=repeat_first(value,batch)
        elif isinstance(value,torch.Tensor) and value.ndim==2 and value.shape[0]==batch:
            if not torch.isfinite(value).all():raise ValueError('Finite batched state required')
            result[key]=value[:1].expand_as(value).clone()
        else:raise ValueError('Unexpected state tensor shape/type')
    return result


def maximum_difference(left,right):
    if set(left)!=set(right):raise ValueError('State keys changed')
    values=[]
    for key,value in left.items():
        other=right[key]
        if isinstance(value,dict):values.append(maximum_difference(value,other))
        else:
            if value.shape!=other.shape:raise ValueError('State shape changed')
            values.append(float((value-other).abs().max()))
    return max(values,default=0.)


def replicate(env,counts):
    import importlib
    for name,expected in PINNED_APIS.items():
        module=importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:raise RuntimeError('Pinned state API differs')
    owner=env.unwrapped
    counts['state_read_calls']+=1;state=owner.get_state_dict()
    target=repeat_first(state,1024)
    counts['state_write_calls']+=1;owner.set_state_dict(target)
    # Joint-delta controller buffers must match the newly copied articulation.
    counts['controller_reset_calls']+=1;owner.agent.controller.reset()
    counts['state_read_calls']+=1;actual=owner.get_state_dict()
    difference=maximum_difference(actual,target)
    if difference>1e-5:raise RuntimeError('Replicated physical state differs beyond predefined tolerance')
    counts['extra_observation_read_calls']+=1;obs=owner.get_obs()
    return obs,difference
