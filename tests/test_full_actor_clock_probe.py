import unittest
import numpy as np
import torch
from research.full_actor_clock_probe import FoldCovariance,episode_gradients


class FullActorClockTests(unittest.TestCase):
    def test_time_cancellation_survives_covariance_measurement(self):
        actor=torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad():actor.weight.zero_()
        logstd=torch.zeros(1,dtype=torch.float64,requires_grad=True)
        stat=FoldCovariance(2)
        for lane in range(4):
            actions=torch.full((2,1),float(lane+1),dtype=torch.float64)
            g=episode_gradients(actor,logstd,torch.ones(2,1,dtype=torch.float64),actions,
                                dict(cancel=torch.tensor([1.,-1.],dtype=torch.float64)))['cancel']
            np.testing.assert_allclose(g,0,atol=1e-12)
            stat.add(lane%2,g)
        self.assertEqual(stat.trace(),0)

    def test_exact_gaussian_score_and_detached_baseline(self):
        actor=torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad():actor.weight.zero_()
        obs=torch.ones(1,1,dtype=torch.float64)
        residual=actor(obs).flatten()+3 # Must not backpropagate through baseline.
        g=episode_gradients(actor,torch.zeros(1,dtype=torch.float64,requires_grad=True),obs,
                            torch.full((1,1),2.,dtype=torch.float64),dict(test=residual))['test']
        np.testing.assert_allclose(g,[6.,9.],atol=1e-12)

    def test_fold_covariance_matches_unbiased_sample_variance(self):
        stat=FoldCovariance(1)
        for lane,value in enumerate([1.,2.,3.,4.]):stat.add(lane%2,np.array([value]))
        self.assertAlmostEqual(stat.trace(),2.)


if __name__=='__main__':unittest.main()
