import unittest
from types import SimpleNamespace
import numpy as np
from research.proxy_calibration_probe import PublicDiagnosticPolicy


class PublicPolicyTests(unittest.TestCase):
    def base(self):
        return SimpleNamespace(schema=SimpleNamespace(observation_dim=1),layers=[(np.zeros((8,1)),np.full(8,2.))],act=lambda obs:[[1.]*8 for _ in obs])
    def test_noise_is_added_before_clipping_raw_mean(self):
        policy=PublicDiagnosticPolicy(self.base(),[1.]*8)
        policy.rng=SimpleNamespace(standard_normal=lambda shape:np.full(shape,-1.5))
        self.assertEqual(policy.act([[0.]]),[[.5]*8])
        self.assertEqual(PublicDiagnosticPolicy(self.base(),None).act([[0.]]),[[1.]*8])
    def test_source_noise_reproducible_across_branches(self):
        a=PublicDiagnosticPolicy(self.base(),[.5]*8);b=PublicDiagnosticPolicy(self.base(),[.5]*8)
        a.reset_noise(937);b.reset_noise(937);self.assertEqual(a.act([[0.]]),b.act([[0.]]))
        with self.assertRaises(ValueError):PublicDiagnosticPolicy(self.base(),[0.]*8)
