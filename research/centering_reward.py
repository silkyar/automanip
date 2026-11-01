"""Training-only restoration of the existing place term after on-top gating.

Underlying task/reward code is untouched. Apply below ManiSkillVectorEnv so the
cost uses the actual post-action state, before horizon autoreset.
"""
import gymnasium as gym
import torch


def alignment_cost(observations,on_top):
    if observations.ndim!=2 or observations.shape[1]!=48 or on_top.shape!=observations.shape[:1] or not torch.isfinite(observations).all():raise ValueError('Frozen48statebatchrequired')
    a=observations[:,25:28];goal=observations[:,32:35].clone();goal[:,2]+=.04
    return on_top.to(observations.dtype)*torch.tanh(5*torch.linalg.vector_norm(a-goal,dim=-1))/8


class CenteringRewardWrapper(gym.Wrapper):
    def __init__(self,env,mode,constant):
        if mode not in ['stock','center','constant'] or not 0<=constant<=.125:raise ValueError('Closedrewardintervention')
        super().__init__(env);self.mode=mode;self.constant=constant;self._records=[];self._last_post=None

    def step(self,action):
        observations,reward,terminated,truncated,info=self.env.step(action)
        potential=alignment_cost(observations,info['is_cubeA_on_cubeB'])
        penalty=potential if self.mode=='center' else info['is_cubeA_on_cubeB'].to(reward.dtype)*self.constant if self.mode=='constant' else torch.zeros_like(reward)
        if reward.shape!=potential.shape:raise ValueError('Rewardbatchdiffers')
        count=min(64,len(reward))
        self._records.append(dict(stock_rewards=reward[:count].detach().cpu().clone(),alignment_cost=potential[:count].detach().cpu().clone(),applied_penalty=penalty[:count].detach().cpu().clone(),task_flags=torch.stack([info[k][:count] for k in ['is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success']],dim=-1).detach().cpu().clone()))
        self._last_post=observations[:count].detach().clone()
        return observations,reward if self.mode=='stock' else reward-penalty,terminated,truncated,info

    def drain(self):
        if len(self._records)!=50:raise ValueError('Exactlyonefull50steptrainingrolloutrequired')
        result={key:torch.stack([r[key] for r in self._records]).numpy() for key in self._records[0]}
        result['terminal_post_observation']=self._last_post.cpu().numpy().copy()
        self._records=[];self._last_post=None
        return result
