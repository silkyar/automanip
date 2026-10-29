import ast
import copy
import inspect
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import torch
from torch import nn

from research.discovery.bc_residual_exploration_v2 import (
    DEFAULT_STD, build_payload, clipped_normal_squared_error, initialize_residual_std,
    lower_noise_counterexample, residual_std, strip_residual_additions,
)


class ResidualExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_residual_rms_is_coordinate_specific_bounded_and_detached(self):
        obs = torch.zeros(10, 48)
        labels = torch.tensor([0., .01, .06, .1, .2, .3, .6, 1.]).expand(10, 8)
        std, rms = residual_std(lambda x: torch.zeros(len(x), 8), obs, labels, batch_size=3)
        torch.testing.assert_close(rms, labels[0].double())
        torch.testing.assert_close(std, labels[0].double().clamp(.05, DEFAULT_STD))
        self.assertFalse(std.requires_grad)
        with self.assertRaises(ValueError):
            residual_std(lambda x: x[:, :8], obs, labels, floor=0)

    def test_exact_clipped_gaussian_moments_match_samples_including_saturation(self):
        mean = torch.tensor([-.8, 0., .8, 2., -2.], dtype=torch.float64)
        std = torch.tensor([.2, .6, .4, .1, .1], dtype=torch.float64)
        target = torch.tensor([-1., -.1, .7, 1., -1.], dtype=torch.float64)
        expected, clipping = clipped_normal_squared_error(mean, std, target)
        generator = torch.Generator().manual_seed(92001)
        draws = mean+std*torch.randn(400000, 5, dtype=torch.float64, generator=generator)
        sampled = (draws.clamp(-1, 1)-target).square().mean(0)
        sampled_clipping = (draws.abs() > 1).double().mean(0)
        torch.testing.assert_close(expected, sampled, atol=.004, rtol=0)
        torch.testing.assert_close(clipping, sampled_clipping, atol=.004, rtol=0)
        self.assertLess(float(expected[-1]), 1e-10)

    def test_training_only_initialization_changes_no_mean_critic_schema_or_rng(self):
        class FakeData:
            dataset_sha256 = "synthetic-contract-fixture"
            train_episode_indices = [1, 4]
            validation_episode_indices = [2]
            def behavior_rows(self, validation=False, device="cpu"):
                if validation:
                    raise AssertionError("Validation rows entered calibration")
                return torch.zeros(100, 48, device=device), torch.full((100, 8), .1, device=device)
        class Agent(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor_mean = nn.Linear(48, 8)
                self.critic = nn.Linear(48, 1)
                self.actor_logstd = nn.Parameter(torch.full((1, 8), -.5))
                with torch.no_grad():
                    self.actor_mean.weight.zero_()
                    self.actor_mean.bias.zero_()
        namespace = dict(torch=torch, math=math, json=json, load_demos=lambda path: FakeData(), residual_std=residual_std,
                         Path=Path, hashlib=hashlib, time=time,
                         AUTOMANIP_RESIDUAL_PATCH_SHA256="synthetic-patch-source", AUTOMANIP_RESIDUAL_BASE_SHA256="synthetic-base-source")
        exec(inspect.getsource(initialize_residual_std), namespace)
        agent = Agent()
        before = copy.deepcopy(agent.state_dict())
        rng = torch.get_rng_state().clone()
        tags = []
        logger = SimpleNamespace(add_scalar=lambda tag, value, step: tags.append(tag))
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary)/"manifest.json").write_text(json.dumps({"weights_sha256": "synthetic-bc-actor"}))
            namespace["initialize_residual_std"](agent, SimpleNamespace(assist_dataset="fixture", assist_bc_actor=temporary), logger)
        self.assertEqual(set(before), set(agent.state_dict()))
        for name, value in before.items():
            if name != "actor_logstd":
                torch.testing.assert_close(value, agent.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(agent.actor_logstd.exp(), torch.full((1, 8), .1), atol=1e-7, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(len(tags), 18)

    def test_lower_noise_is_not_unconditionally_better_for_correction_discovery(self):
        probability = lower_noise_counterexample()["correction_probability"]
        self.assertGreater(probability["original"], .1)
        self.assertLess(probability["floor"], 1e-8)

    def test_payload_patch_exactly_preserves_frozen_v1_code_and_agent(self):
        paths = {
            "assist-bc-ppo-residual-std": "assist-bc-ppo.py",
            "assist-bc-ppo-amp-residual-std": "assist-bc-ppo-amp.py",
        }
        for method, filename in paths.items():
            base = (Path("research/discovery/amp-prior-assisted-v1")/filename).read_bytes()
            payload = build_payload(base, method)
            self.assertEqual(strip_residual_additions(payload.decode()).encode(), base)
            before = next(n for n in ast.parse(base).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            after = next(n for n in ast.parse(payload).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            self.assertEqual(ast.dump(before), ast.dump(after))
            self.assertNotIn(b"from research.", payload)
            self.assertNotIn(b"from automanip.", payload)
            self.assertLess(payload.index(b"initialize_residual_std(agent, args, logger)"), payload.index(b"optimizer = optim.Adam(agent.parameters()"))
            with self.assertRaises(ValueError):
                build_payload(base+b"\n", method)


if __name__ == "__main__":
    unittest.main()
