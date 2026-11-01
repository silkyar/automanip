from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, ValidatedPolicy, load_policy
from research.representation_distill import Student, export_student


def artifact():
    weight = np.zeros((8, 48), dtype=np.float32); weight[0, 7] = 1.
    return ValidatedPolicy((8,), {'layers.0.weight': tuple(weight.reshape(-1)),
        'layers.0.bias': (0.,)*8, 'layers.1.weight': (0.,)*64, 'layers.1.bias': (0.,)*8}, 'fixture')


class RepresentationTests(unittest.TestCase):
    def test_initial_actor_and_safe_export_exclude_auxiliary_head(self):
        torch.manual_seed(2); a = artifact(); model = Student(a)
        schema = PolicySchema('a'*64, 48, 8); obs = np.zeros((4, 48), dtype=np.float32)
        obs[:, 7] = [.01, .02, .03, .04]
        np.testing.assert_array_equal(model(torch.tensor(obs))[0].detach().numpy(), NumpyPolicy(a, schema).act(obs))
        with TemporaryDirectory() as temporary:
            target = Path(temporary)/'policy'; export_student(model, a, schema, target)
            loaded = load_policy(target, schema)
            self.assertEqual(set(loaded.tensors), set(a.tensors))
            self.assertFalse(any('coordinate' in k for k in loaded.tensors))

    def test_known_coordinate_can_be_learned_without_changing_zero_actor(self):
        torch.set_num_threads(1); torch.manual_seed(3); model = Student(artifact())
        x = torch.zeros((128, 48)); x[:, 7] = torch.linspace(0, .04, 128)
        y = (x[:, 7]-.015)/.025
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        before = float((model(x)[1]-y).square().mean().detach())
        for _ in range(256):
            action, coordinate = model(x)
            loss = (coordinate-y).square().mean()+action.square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        action, coordinate = model(x)
        self.assertLess(float((coordinate-y).square().mean().detach()), before/100)
        self.assertLess(float(action.abs().max().detach()), 1e-6)


if __name__ == '__main__': unittest.main()
