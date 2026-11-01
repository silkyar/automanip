import unittest

import numpy as np
import torch

from research.executed_rehearsal import objective


class ExecutedRehearsalTests(unittest.TestCase):
    def test_saturated_raw_values_can_change_without_changing_executed_action(self):
        t = np.array([[1.1, 1.]]); r = np.array([[1., 1.]])
        original = np.array([[0.], [-3.]])
        delta = np.linalg.solve(t.T@t+4*r.T@r+.001*np.eye(2), t.T@np.array([[3.2]]))
        feasible = np.array([[12.], [-13.]])-original
        arrays = [torch.tensor(x, dtype=torch.float64) for x in (t, np.array([[.2]]), t@original, r, r@original)]
        old = torch.tensor(delta, requires_grad=True)
        raw_loss = objective(old, *arrays, executed=False)
        raw_gradient, = torch.autograd.grad(raw_loss, old)
        self.assertLess(float(raw_gradient.abs().max()), 1e-10)
        executed_loss = objective(old, *arrays, executed=True)
        gradient, = torch.autograd.grad(executed_loss, old)
        self.assertGreater(float(gradient.abs().max()), .1)
        feasible_tensor = torch.tensor(feasible)
        self.assertLess(float(objective(feasible_tensor, *arrays, executed=True)), float(executed_loss))
        # The old raw value moves substantially, while its robot command stays -1.
        self.assertAlmostEqual(float(np.clip(r@(original+delta), -1, 1)[0, 0]), -1.)
        self.assertAlmostEqual(float(np.clip(r@(original+feasible), -1, 1)[0, 0]), -1.)
        self.assertAlmostEqual(float((t@(original+feasible))[0, 0]), .2)

    def test_teacher_gradient_remains_nonzero_in_wrong_saturation(self):
        delta = torch.tensor([[0.]], requires_grad=True)
        t = torch.ones((1, 1)); old = torch.full((1, 1), -3.); target = torch.ones((1, 1))
        loss = objective(delta, t, target, old, t, old, executed=True)
        loss.backward()
        self.assertLess(float(delta.grad), -1.)


if __name__ == '__main__': unittest.main()
