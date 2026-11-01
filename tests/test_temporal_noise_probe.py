import unittest
from types import SimpleNamespace
import numpy as np
from research.temporal_noise_probe import TemporalNoisePolicy,integrated_noise_scale


class NoiseTests(unittest.TestCase):
    def base(self):return SimpleNamespace(schema=SimpleNamespace(observation_dim=1),layers=[(np.zeros((8,1)),np.zeros(8))])
    def test_stationary_variance_and_correlation(self):
        policy=TemporalNoisePolicy(self.base(),[.1]*8,.9);obs=np.zeros((5000,1))
        policy.reset_noise(17);policy.act(obs);first=policy.latent_noises[-1].copy();policy.act(obs);second=policy.latent_noises[-1]
        self.assertLess(abs(float(first.var())-1),.04);self.assertLess(abs(float(second.var())-1),.04)
        self.assertLess(abs(float(np.corrcoef(first.ravel(),second.ravel())[0,1])-.9),.02)
    def test_common_initial_noise_and_episode_reset(self):
        a=TemporalNoisePolicy(self.base(),[.1]*8,0.);b=TemporalNoisePolicy(self.base(),[.1]*8,.9)
        a.reset_noise(9);b.reset_noise(9);self.assertEqual(a.act([[0.]]),b.act([[0.]]))
        a.act([[0.]]);b.act([[0.]]);a.reset_noise(9);b.reset_noise(9)
        self.assertEqual(a.act([[0.]]),b.act([[0.]]))
    def test_sum_variance_matching_uses_finite_horizon(self):
        cov=np.power(.9,np.abs(np.arange(50)[:,None]-np.arange(50)[None,:]))
        self.assertAlmostEqual(integrated_noise_scale(50,.9)**2*50,float(cov.sum()))
        self.assertEqual(integrated_noise_scale(1,.9),1.)
