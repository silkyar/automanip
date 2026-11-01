import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, ValidatedPolicy, load_policy
from automanip.stackcube import STACK_TASK
from research.gripper_hold_initialization import configure, maximum_fixed_change, verify_optimizer
from research.readout_distill import export_readout


def actor():
    return nn.Sequential(nn.Linear(48, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                         nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 8))


class HoldInitializationTests(unittest.TestCase):
    def test_only_known_hold_map_differs_between_matched_actors(self):
        torch.manual_seed(41)
        left = actor()
        right = copy.deepcopy(left)
        lm, _ = configure(left, False)
        rm, _ = configure(right, True)
        for (_, a), (_, b) in zip(lm, rm):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = torch.randn(1001, 48)
        x[:, 7] = torch.linspace(-.01, .04, len(x))
        difference = right(x)-left(x)
        torch.testing.assert_close(difference[:, :7], torch.zeros_like(difference[:, :7]), rtol=0, atol=0)
        error_metres = ((difference[:, 7]-(40*x[:, 7]-.6))*.025).abs().max()
        self.assertLess(error_metres, 3e-7)

    def test_fixed_path_survives_adam_and_global_gradient_clipping(self):
        model = actor()
        masks, _ = configure(model, True)
        before = [p.detach().clone() for p, _ in masks]
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003, eps=1e-5)
        verify_optimizer(optimizer)
        for _ in range(16):
            optimizer.zero_grad()
            (model(torch.randn(32, 48))-torch.randn(32, 8)).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
        self.assertEqual(maximum_fixed_change(masks, before), 0)
        self.assertTrue(any(not torch.equal(p, old) for (p, _), old in zip(masks, before)))
        with self.assertRaises(ValueError):
            verify_optimizer(torch.optim.AdamW(model.parameters()))

    def test_existing_safe_export_needs_no_controller_code(self):
        torch.manual_seed(41)
        model = actor()
        configure(model, True)
        tensors = {}
        for i, layer in enumerate([model[j] for j in (0, 2, 4, 6)]):
            for name in ('weight', 'bias'):
                tensors[f'layers.{i}.{name}'] = tuple(getattr(layer, name).detach().numpy().ravel().tolist())
        artifact = ValidatedPolicy((256, 256, 256), tensors, 'local-test')
        schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
        x = torch.randn(32, 48)
        x[:, 7] = torch.linspace(-.01, .04, len(x))
        with TemporaryDirectory() as directory:
            export_readout(artifact, schema, np.zeros((257, 8)), Path(directory)/'policy')
            policy = NumpyPolicy(load_policy(Path(directory)/'policy', schema), schema)
            actual = np.asarray(policy.act(x.tolist()))
        self.assertLess(np.abs(actual-model(x).detach().numpy().clip(-1, 1)).max(), 5e-6)


if __name__ == '__main__':
    unittest.main()
