import unittest

import torch

from research.discovery.quantile_tail_v1 import episode_cvar_score_weights, lower_tail, quantile_loss


class QuantileTailTests(unittest.TestCase):
    def test_fractional_tail_mass_and_gradients(self):
        values = torch.tensor([[0., 2., 4., 6.]], requires_grad=True)
        result = lower_tail(values, .375)
        self.assertAlmostEqual(float(result), 2/3, places=6)
        result.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[2/3, 1/3, 0., 0.]]))
        torch.testing.assert_close(lower_tail(values, 1.), values.mean(-1, keepdim=True))

    def test_pinball_score_and_detached_observed_targets(self):
        prediction = torch.tensor([[.3]], requires_grad=True)
        target = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
        loss = quantile_loss(prediction, target, torch.tensor([.25]))
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad), .25, places=6)
        self.assertIsNone(target.grad)

    def test_cvar_episode_weights_include_var_baseline(self):
        returns = torch.tensor([0., 1., 2., 3., 4.])
        weights, threshold = episode_cvar_score_weights(returns, .4)
        self.assertAlmostEqual(float(threshold), 1.6, places=6)
        torch.testing.assert_close(weights, torch.tensor([-4., -1.5, 0., 0., 0.]))
        # Selecting bad returns and multiplying them by positive return would
        # produce a different score estimator, including a zero failure weight.
        self.assertLess(float(weights[0]), 0.)

    def test_invalid_mass_nonfinite_and_batch_shapes(self):
        with self.assertRaises(ValueError):
            lower_tail(torch.ones(2, 4), 0.)
        with self.assertRaises(ValueError):
            episode_cvar_score_weights(torch.tensor([0., float("nan")]))
        with self.assertRaises(ValueError):
            quantile_loss(torch.ones(2, 4), torch.ones(3, 1), torch.ones(4))


if __name__ == "__main__":
    unittest.main()
