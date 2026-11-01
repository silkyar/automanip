import unittest
import torch
from research.old_state_metric_probe import constrained_direction


class OldStateMetricTests(unittest.TestCase):
    def test_separable_case_preserves_old_behavior_without_losing_gradient(self):
        current=torch.eye(2,dtype=torch.float64);old=torch.diag(torch.tensor([1.,0.],dtype=torch.float64));g=torch.ones(1,2,dtype=torch.float64)
        _,a=constrained_direction(g,current,old,0.,.01,1e-8)
        _,b=constrained_direction(g,current,old,1.,.01,1e-8)
        self.assertAlmostEqual(b['current_KL'],.01)
        self.assertLess(b['old_KL']/a['old_KL'],.5)
        self.assertGreater(b['linear_gain']/a['linear_gain'],.8)
    def test_identical_state_metrics_have_no_free_retention_gain(self):
        current=torch.eye(2,dtype=torch.float64);g=torch.ones(1,2,dtype=torch.float64)
        _,a=constrained_direction(g,current,current,0.,.01,1e-8)
        _,b=constrained_direction(g,current,current,1.,.01,1e-8)
        self.assertAlmostEqual(a['old_KL'],b['old_KL'])
