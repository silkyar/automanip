from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from research.affine_student import AffineStudent, export_affine_student
from research.representation_distill import Student
from tests.test_representation_distill import artifact


class AffineStudentTests(unittest.TestCase):
    def test_forward_raw_parameter_penalty_and_export_are_function_preserving(self):
        a = artifact(); plain = Student(a)
        # Nonzero actor output makes changes to the transformed bias observable.
        with torch.no_grad():
            plain.layers[-1].weight.fill_(.1)
        tensors = dict(a.tensors); tensors['layers.1.weight'] = tuple(plain.layers[-1].weight.detach().numpy().reshape(-1))
        from automanip.policy_io import ValidatedPolicy
        a = ValidatedPolicy(a.hidden_sizes, tensors, 'fixture'); plain = Student(a)
        mean = np.linspace(-.1, .1, 48); scale = np.linspace(.001, 1, 48)
        scaled = AffineStudent(a, mean, scale)
        x = torch.linspace(-.2, .2, 48*4).reshape(4, 48)
        np.testing.assert_allclose(plain(x)[0].detach(), scaled(x)[0].detach(), atol=1e-7)
        for left, right in zip(plain.layers.parameters(), scaled.raw_parameters()):
            np.testing.assert_allclose(left.detach(), right.detach(), atol=1e-7)
        with TemporaryDirectory() as temporary:
            schema = PolicySchema('a'*64, 48, 8); out = Path(temporary)/'policy'
            export_affine_student(scaled, a, schema, out)
            loaded = load_policy(out, schema)
            np.testing.assert_allclose(NumpyPolicy(loaded, schema).act(x.numpy()), plain(x)[0].detach(), atol=1e-7)

    def test_penalty_stays_in_original_coordinates_after_an_update(self):
        a = artifact(); scale = np.full(48, .01); mean = np.full(48, .02)
        model = AffineStudent(a, mean, scale)
        before = model.raw_parameters()[0].detach().clone()
        with torch.no_grad(): model.layers[0].weight[0, 7] += .001
        change = model.raw_parameters()[0]-before
        self.assertAlmostEqual(float(change[0, 7].detach()), .1, places=5)
        loss = change.square().sum(); loss.backward()
        self.assertAlmostEqual(float(model.layers[0].weight.grad[0, 7]), 20, places=3)


if __name__ == '__main__': unittest.main()
