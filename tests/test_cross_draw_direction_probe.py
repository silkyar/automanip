import unittest
import torch
from research.cross_draw_direction_probe import score_projection


class ProjectionTests(unittest.TestCase):
    def test_sample_projection_matches_aggregated_gradient_and_reverse(self):
        torch.manual_seed(17);phi=torch.randn(13,5,dtype=torch.float64);innovation=torch.randn(13,3,dtype=torch.float64)
        advantage=torch.randn(13,dtype=torch.float64);direction=torch.randn(3,5,dtype=torch.float64)
        gradient=(innovation*advantage[:,None]).T@phi/len(phi)
        score=score_projection(phi,innovation,advantage,direction)
        torch.testing.assert_close(score.mean(),(gradient*direction).sum())
        torch.testing.assert_close(score_projection(phi,innovation,advantage,-direction),-score)
