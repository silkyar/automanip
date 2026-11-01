"""Execute literal generated PPO update blocks on a synthetic CPU rollout."""
import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import textwrap
import time
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from research.feature_value_baseline import FeatureValueBaseline
from scripts.prepare_feature_value import build_programs
from scripts.prepare_scratch_credit import update_region,gae_region


class FeatureValueUpdateTests(unittest.TestCase):
    def test_generated_updates_keep_value_fit_detached_and_critic_inactive(self):
        torch.set_num_threads(2)
        for method,source in build_programs().items():
            with self.subTest(method=method),torch.random.fork_rng():
                torch.manual_seed(17)
                definitions=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['Agent','layer_init']]
                ns=dict(torch=torch,nn=nn,np=np,Normal=Normal)
                exec(compile(ast.Module(body=definitions,type_ignores=[]),'generated-training-classes','exec'),ns)
                env=SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)),single_action_space=SimpleNamespace(shape=(8,)))
                agent=ns['Agent'](env);agent.critic.requires_grad_(False)
                critic_before={k:v.clone() for k,v in agent.critic.state_dict().items()}
                helper=FeatureValueBaseline(agent.actor_mean,50,4,'clock' if 'clock' in method else 'state');helper.begin()
                obs=torch.randn(50,4,48);actions=[];logprobs=[]
                with torch.no_grad():
                    for t in range(50):
                        helper.active_step=t
                        action,lp,_,value=agent.get_action_and_value(obs[t])
                        helper.active_step=None
                        self.assertEqual(float(value.abs().sum()),0.)
                        actions.append(action);logprobs.append(lp)
                opt=torch.optim.Adam(agent.parameters(),lr=.0003,eps=1e-5)
                logged={}
                logger=SimpleNamespace(add_scalar=lambda name,value,step:logged.update({name:value}))
                args=SimpleNamespace(num_steps=50,gamma=.9,batch_size=200,minibatch_size=100,update_epochs=1,target_kl=.1,norm_adv=True,
                                     clip_coef=.2,clip_vloss=False,ent_coef=.003,vf_coef=.5,max_grad_norm=.5)
                with tempfile.TemporaryDirectory() as folder:
                    ns.update(args=args,automanip_value=helper,rewards=torch.full((50,4),.5),values=torch.zeros(50,4),
                              dones=torch.zeros(50,4),next_done=torch.ones(4),logger=logger,global_step=200,
                              json=json,automanip_journal=io.StringIO(),automanip_root=Path(folder))
                    a,b=gae_region(source)
                    exec(compile(textwrap.dedent(source[a:b]),'literal-generated-target-fit','exec'),ns)
                    fitted=ns['fitted_value'];returns=ns['returns']
                    self.assertFalse(fitted.requires_grad)
                    self.assertTrue((Path(folder)/'value-200.npz').is_file())
                ns.update(agent=agent,args=args,optimizer=opt,device='cpu',time=time,logger=logger,global_step=12,
                    b_obs=obs.reshape(-1,48),b_actions=torch.stack(actions).reshape(-1,8),b_logprobs=torch.stack(logprobs).flatten(),
                    b_advantages=ns['advantages'].flatten(),b_returns=returns.flatten(),b_values=fitted.flatten(),
                    automanip_optimizer_steps=0,automanip_event=lambda *x:None)
                exec(compile(textwrap.dedent(update_region(source)),'literal-generated-ppo-update','exec'),ns)
                self.assertGreater(ns['automanip_optimizer_steps'],0)
                self.assertEqual(logged['audit/critic_gradient_norm'],0.)
                for key,value in agent.critic.state_dict().items():torch.testing.assert_close(value,critic_before[key],rtol=0,atol=0)
                self.assertTrue(all(p.grad is None and p not in opt.state for p in agent.critic.parameters()))
                helper.close()


if __name__=='__main__':unittest.main()
