import math
import unittest

import torch
from torch import nn

from research.discovery.amp_v1 import StudentActor
from research.discovery.gaussian_mean_kl_probe_v1 import gaussian_kl_parts, one_actor_update


class GaussianMeanKlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_analytic_parts_match_normal_kl_direction(self):
        generator = torch.Generator().manual_seed(31)
        old_mean = torch.randn(19, 8, dtype=torch.float64, generator=generator)
        new_mean = old_mean+.1*torch.randn(19, 8, dtype=torch.float64, generator=generator)
        old_std = torch.full((1, 8), .6, dtype=torch.float64)
        new_std = torch.full((1, 8), .13, dtype=torch.float64)
        total, mean, variance = gaussian_kl_parts(old_mean, old_std, new_mean, new_std)
        expected = torch.distributions.kl_divergence(torch.distributions.Normal(old_mean, old_std),
                                                   torch.distributions.Normal(new_mean, new_std)).sum(-1).mean()
        torch.testing.assert_close(total, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(total, mean+variance)

    def test_equal_std_mean_kl_scales_inverse_variance(self):
        old = torch.zeros(10, 8, dtype=torch.float64)
        new = torch.full_like(old, .03)
        large, _, _ = gaussian_kl_parts(old, torch.ones(1, 8), new, torch.ones(1, 8))
        small, _, _ = gaussian_kl_parts(old, torch.full((1, 8), .5), new, torch.full((1, 8), .5))
        self.assertAlmostEqual(float(small/large), 4., places=12)

    def test_adam_first_mean_step_does_not_shrink_with_sigma(self):
        result = []
        for std in (math.exp(-.5), .08):
            mean = nn.Parameter(torch.tensor(0., dtype=torch.float64))
            optimizer = torch.optim.Adam([mean], lr=.0003, eps=1e-5)
            loss = -torch.distributions.Normal(mean, std).log_prob(torch.tensor(std, dtype=torch.float64))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            result.append((float(mean.detach()), float(mean.detach().square()/(2*std**2))))
        self.assertAlmostEqual(result[1][0]/result[0][0], 1., delta=1e-4)
        self.assertAlmostEqual(result[1][1]/result[0][1], (math.exp(-.5)/.08)**2, delta=.01)

    def test_mean_lr_intervention_does_not_change_std_update_or_base_actor(self):
        torch.manual_seed(32)
        actor = StudentActor((16, 16, 16))
        original = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        generator = torch.Generator().manual_seed(33)
        obs, probe = torch.randn(64, 48, generator=generator), torch.randn(64, 48, generator=generator)
        noise, probe_noise = torch.randn(64, 8, generator=generator), torch.randn(64, 8, generator=generator)
        advantage = noise[:, 0]+.5*noise[:, 1]
        args = (actor, obs, probe, torch.full((8,), .1), noise, probe_noise, advantage)
        full = one_actor_update(*args, .0003)
        scaled = one_actor_update(*args, .00003)
        self.assertEqual(full["new_std"], scaled["new_std"])
        self.assertLess(scaled["probe_mean_kl_component"], full["probe_mean_kl_component"])
        self.assertAlmostEqual(scaled["mean_parameter_step_l2"]/full["mean_parameter_step_l2"], .1, delta=1e-4)
        for name, value in original.items():
            torch.testing.assert_close(value, actor.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
