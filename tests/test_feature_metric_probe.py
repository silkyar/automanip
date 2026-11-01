import unittest
import torch
from research.feature_metric_probe import normalize_direction


class MetricTests(unittest.TestCase):
    def test_fixed_movement_and_positive_scale_invariance(self):
        g=torch.tensor([[1.,2.]],dtype=torch.float64);c=torch.diag(torch.tensor([2.,.3],dtype=torch.float64))
        d=normalize_direction(g,c)
        self.assertAlmostEqual(float(.5*(d*(d@c)).sum()),.01)
        torch.testing.assert_close(d,normalize_direction(3*g,c))
