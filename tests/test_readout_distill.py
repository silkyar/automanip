from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, ValidatedPolicy, load_policy
from research.readout_distill import export_readout, features, fit_delta


class ReadoutDistillTests(unittest.TestCase):
    def test_learns_separable_correction_and_preserves_distinct_early_states(self):
        t = np.tile([1., 0], (20, 1)); r = np.tile([0., 1], (20, 1))
        delta, report = fit_delta(t, np.ones((20, 1)), np.zeros((20, 1)), r)
        self.assertGreater(float((t@delta).mean()), .99)
        self.assertEqual(float((r@delta).max()), 0.)
        self.assertLess(report['normal_equation_max_residual'], 1e-12)

    def test_aliasing_negative_control_exposes_teacher_retention_conflict(self):
        same = np.ones((20, 1))
        delta, _ = fit_delta(same, same, np.zeros_like(same), same)
        self.assertAlmostEqual(float(delta[0, 0]), 1/5.001)
        self.assertLess(float(delta[0, 0]), .3)

    def test_export_reloads_under_frozen_schema_and_keeps_hidden_weights(self):
        schema = PolicySchema('a'*64, 2, 1)
        artifact = ValidatedPolicy((2,), {'layers.0.weight': (1., 0., 0., 1.),
            'layers.0.bias': (0., 0.), 'layers.1.weight': (.2, -.1), 'layers.1.bias': (.05,)}, 'unused')
        base = NumpyPolicy(artifact, schema); obs = np.array([[.1, .3], [.2, -.2]])
        phi = features(base, obs); delta = np.array([[.1], [.05], [.02]])
        with TemporaryDirectory() as temporary:
            target = Path(temporary)/'policy'; export_readout(artifact, schema, delta, target)
            updated = load_policy(target, schema)
            self.assertEqual(updated.tensors['layers.0.weight'], artifact.tensors['layers.0.weight'])
            expected = np.asarray(base.act(obs))+phi@delta
            np.testing.assert_allclose(NumpyPolicy(updated, schema).act(obs), expected, atol=1e-7)


if __name__ == '__main__': unittest.main()
