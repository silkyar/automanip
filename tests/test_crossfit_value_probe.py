import unittest
import torch
from research.crossfit_value_probe import discounted_returns,crossfit_baselines


class ValueProbeTests(unittest.TestCase):
    def test_finite_return_tail_is_explicit(self):
        result=discounted_returns(torch.ones(3,2,dtype=torch.float64),.5)
        torch.testing.assert_close(result,torch.tensor([[1.75,1.75],[1.5,1.5],[1.,1.]],dtype=torch.float64))
    def test_baseline_never_uses_own_episode_rewards(self):
        torch.manual_seed(4);phi=torch.randn(40,3,dtype=torch.float64);phi[:,-1]=1
        rewards=torch.randn(40,dtype=torch.float64);parity=torch.arange(40)%4%2
        a=crossfit_baselines(phi,rewards,4);modified=rewards.clone();modified[parity==0]+=100
        b=crossfit_baselines(phi,modified,4)
        torch.testing.assert_close(a['state'][parity==0],b['state'][parity==0])
        torch.testing.assert_close(a['constant'][parity==0],b['constant'][parity==0])
