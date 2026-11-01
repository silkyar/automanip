import unittest
import gymnasium as gym
import numpy as np
import torch
from research.centering_reward import alignment_cost,CenteringRewardWrapper


def observations(offset=.015):
    state=torch.zeros((1,48),dtype=torch.float64);state[:,25:28]=torch.tensor([offset,0,.06]);state[:,32:35]=torch.tensor([0,0,.02]);return state


class FakeTask(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(48,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(8,),dtype=np.float32)
    def reset(self,seed=None,options=None):self.state=observations(.3);return self.state,{}
    def step(self,action):
        self.state=observations(.015);flags={k:torch.tensor([v]) for k,v in dict(is_cubeA_on_cubeB=True,is_cubeA_grasped=True,is_cubeA_static=False,success=False).items()}
        return self.state,torch.tensor([.84]),torch.tensor([False]),torch.tensor([False]),flags


class CenteringRewardTests(unittest.TestCase):
    def test_exact_existing_place_cost_and_gradient(self):
        x=observations().requires_grad_(True);cost=alignment_cost(x,torch.tensor([True]));cost.sum().backward()
        expected=5/8/(np.cosh(5*.015)**2)
        self.assertAlmostEqual(float(cost),float(np.tanh(5*.015)/8),places=8)
        self.assertAlmostEqual(float(x.grad[0,25]),expected,places=7)
        self.assertEqual(float(alignment_cost(x,torch.tensor([False]))),0)

    def test_actual_terminal_state_survives_outer_autoreset(self):
        env=CenteringRewardWrapper(FakeTask(),'center',.01);env.reset()
        for _ in range(50):obs,reward,*_=env.step(None)
        expected=.84-float(np.tanh(5*.015)/8)
        self.assertAlmostEqual(float(reward),expected,places=6)
        env.reset()  # Simulate outer VectorEnv horizon reset before trace drain.
        trace=env.drain();self.assertAlmostEqual(float(trace['terminal_post_observation'][0,25]),.015,places=7)
        self.assertEqual(trace['stock_rewards'].shape,(50,1));self.assertEqual(trace['task_flags'].shape,(50,1,4))

    def test_constant_control_changes_level_without_alignment_gradient(self):
        env=CenteringRewardWrapper(FakeTask(),'constant',.01);env.reset();_,reward,*_=env.step(None)
        self.assertAlmostEqual(float(reward),.83,places=6)
        stock=CenteringRewardWrapper(FakeTask(),'stock',.01);stock.reset();_,original,*_=stock.step(None)
        self.assertAlmostEqual(float(original),.84,places=6)
        with self.assertRaises(ValueError):stock.drain()


if __name__=='__main__':unittest.main()
