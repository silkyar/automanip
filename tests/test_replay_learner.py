import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from research.discovery.replay_learner import BatchRenorm, Critic, Learner, METHODS, export_actor
from research.discovery.sac_baseline import numerical_action, numerical_layers


class ReplayMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def batch(self,count=32):
        g=torch.Generator().manual_seed(447)
        return dict(obs=torch.randn(count,48,generator=g),next_obs=torch.randn(count,48,generator=g),
            action=torch.randn(count,8,generator=g).tanh(),reward=torch.rand(count,1,generator=g),stop=torch.zeros(count,1))

    def test_zero_variance_is_finite_before_and_after_renorm_warmup(self):
        for split in (False,True):
            for warm in (0,100000):
                norm=BatchRenorm(3,split=split);norm.steps.fill_(warm)
                x=torch.ones(16,3,requires_grad=True);y=norm(x);y.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(y).all())

    def test_joint_split_isolates_batch_moments_not_pooled_ema_clock(self):
        x=torch.cat((torch.arange(4.)-4,torch.arange(4.)+4)).reshape(-1,1)
        joint,split=BatchRenorm(1),BatchRenorm(1,split=True)
        a,b=joint(x),split(x)
        self.assertGreater(float((a-b).abs().max()),.1)
        torch.testing.assert_close(joint.running_mean,split.running_mean,rtol=0,atol=0)
        torch.testing.assert_close(joint.running_var,split.running_var,rtol=0,atol=0)
        self.assertEqual(int(joint.steps),1);self.assertEqual(int(split.steps),1)

    def test_target_detached_but_current_prediction_cross_half_gradient_present(self):
        norm=BatchRenorm(1)
        x=torch.tensor([[-3.],[-1.],[2.],[4.]],requires_grad=True)
        y=norm(x);y[0].backward()
        self.assertGreater(float(x.grad[2:].abs().sum()),.01)
        model=Learner('joint-moments',width=16);data=self.batch()
        with torch.no_grad():a,lp=model.actor.sample(data['next_obs'])
        q1,q2,target=model.critic_target(data,a,lp,torch.tensor(.1))
        self.assertFalse(target.requires_grad)
        self.assertTrue(q1.requires_grad)

    def test_all_methods_same_actor_initialization_and_joint_split_linear_weights(self):
        models=[]
        for method in METHODS:
            torch.manual_seed(12);models.append(Learner(method,width=16))
        for model in models[1:]:
            for a,b in zip(models[0].actor.parameters(),model.actor.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for layer in ('first','second','output'):
                torch.testing.assert_close(getattr(models[0].q1,layer).weight,getattr(model.q1,layer).weight,rtol=0,atol=0)

    def test_actor_delay_exact_work_and_no_normalization_updates_during_actor_pass(self):
        torch.manual_seed(13);model=Learner('joint-moments',width=16);batch=self.batch()
        batch['obs'][:,0]=0;batch['next_obs'][:,0]=0
        for _ in range(3):stats=model.update(batch)
        self.assertEqual(model.updates,3);self.assertEqual(model.actor_updates,1)
        self.assertEqual(model.work['critic_optimizer_calls'],3)
        self.assertEqual(model.work['actor_optimizer_calls'],1)
        for norm in [m for m in model.q1.modules() if isinstance(m,BatchRenorm)]:self.assertEqual(int(norm.steps),3)
        before={k:v.clone() for k,v in model.q1.state_dict().items()}
        model.diagnostics(batch)
        for key,value in model.q1.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))

    def test_safe_actor_is_squashed_mean_not_clipped_latent(self):
        model=Learner('joint-moments',width=16);x=torch.randn(17,48)
        with torch.no_grad():
            model.actor.mean.bias.fill_(2)
            reference=model.actor.deterministic(x)
            numerical=numerical_action(x,numerical_layers(model.actor))
        torch.testing.assert_close(reference,numerical,rtol=1e-6,atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            record=export_actor(model.actor,Path(tmp)/'safe')
            self.assertEqual(len(record['weights_sha256']),64)


if __name__=='__main__':unittest.main()
