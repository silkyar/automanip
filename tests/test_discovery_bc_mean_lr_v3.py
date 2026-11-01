import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from research.discovery.bc_mean_lr_v3 import BASE_SHA256, apply_mean_lr, build_payload, configure_mean_lr, strip_additions


class Agent(nn.Module):
    def __init__(self, std=math.exp(-.5)):
        super().__init__()
        self.actor_mean = nn.Sequential(nn.Linear(48, 16), nn.Tanh(), nn.Linear(16, 8))
        self.critic = nn.Sequential(nn.Linear(48, 16), nn.Tanh(), nn.Linear(16, 1))
        self.actor_logstd = nn.Parameter(torch.full((1, 8), math.log(std)))


class MeanLearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def compare_states(self, first, second, opt_first, opt_second):
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(opt_first.state[left].keys(), opt_second.state[right].keys())
            for key in opt_first.state[left]:
                torch.testing.assert_close(opt_first.state[left][key], opt_second.state[right][key], atol=0, rtol=0)

    def test_ratio_one_preserves_joint_parameters_and_adam_states_exactly(self):
        torch.manual_seed(712)
        baseline = Agent()
        candidate = copy.deepcopy(baseline)
        opt_base = torch.optim.Adam(baseline.parameters(), lr=.0003, eps=1e-5)
        opt_candidate = torch.optim.Adam(candidate.parameters(), lr=.0003, eps=1e-5)
        rng = torch.get_rng_state().clone()
        report = configure_mean_lr(opt_candidate, candidate, 2)
        self.assertEqual(report["mean_lr_multiplier"], 1.)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        observation = torch.randn(32, 48)
        for lr in (.0003, .0001, .0002):
            for group in opt_base.param_groups:
                group["lr"] = lr
            for group in opt_candidate.param_groups:
                group["lr"] = lr
            apply_mean_lr(opt_candidate, lr)
            for agent, optimizer in ((baseline, opt_base), (candidate, opt_candidate)):
                optimizer.zero_grad()
                loss = agent.actor_mean(observation).square().mean()+agent.critic(observation).square().mean()+agent.actor_logstd.square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                optimizer.step()
            self.compare_states(baseline, candidate, opt_base, opt_candidate)

    def test_only_mean_lr_scales_and_scheduler_reapplies_fixed_initial_ratio(self):
        for power in (1, 2):
            torch.manual_seed(713)
            baseline = Agent(.1)
            candidate = copy.deepcopy(baseline)
            before = copy.deepcopy(candidate.state_dict())
            first = torch.optim.Adam(baseline.parameters(), lr=.0003, eps=1e-5)
            second = torch.optim.Adam(candidate.parameters(), lr=.0003, eps=1e-5)
            report = configure_mean_lr(second, candidate, power)
            multiplier = (.1/math.exp(-.5))**power
            self.assertAlmostEqual(report["mean_lr_multiplier"], multiplier, places=7)
            self.assertEqual(set(before), set(candidate.state_dict()))
            for name in before:
                torch.testing.assert_close(before[name], candidate.state_dict()[name], atol=0, rtol=0)
            for base_lr in (.0003, .00015, .0001):
                # Exact original constant reset, then original optional anneal.
                for group in second.param_groups:
                    group["lr"] = .0003
                apply_mean_lr(second, .0003)
                second.param_groups[0]["lr"] = base_lr
                apply_mean_lr(second, base_lr)
                self.assertEqual(second.param_groups[0]["lr"], base_lr)
                self.assertEqual(second.param_groups[1]["lr"], base_lr*report["mean_lr_multiplier"])
            # Identical fixed gradients isolate optimizer intervention: std and
            # critic states/weights match exactly; mean parameter displacement scales.
            for left, right in zip(baseline.parameters(), candidate.parameters()):
                left.grad = torch.ones_like(left)*.2
                right.grad = left.grad.clone()
            for group in first.param_groups:
                group["lr"] = .0001
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), .5)
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), .5)
            first.step()
            second.step()
            for name, left in baseline.named_parameters():
                right = dict(candidate.named_parameters())[name]
                if not name.startswith("actor_mean."):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(before[name]-right, (before[name]-left)*report["mean_lr_multiplier"], atol=2e-8, rtol=1e-4)
                for key in first.state[left]:
                    torch.testing.assert_close(first.state[left][key], second.state[right][key], atol=0, rtol=0)

    def test_payload_exact_strip_schema_helpers_and_both_scheduler_anchors(self):
        base = Path("research/discovery/bc-residual-exploration-v2/assist-bc-ppo-residual-std.py").read_bytes()
        self.assertEqual(hashlib.sha256(base).hexdigest(), BASE_SHA256)
        for power in (1, 2):
            source = build_payload(base, power).decode()
            self.assertEqual(strip_additions(source).encode(), base)
            tree = ast.parse(source)
            original_agent = next(n for n in ast.parse(base).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            new_agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            self.assertEqual(ast.dump(original_agent), ast.dump(new_agent))
            self.assertNotIn("from research.", source)
            self.assertNotIn("from automanip.", source)
            self.assertEqual(source.count("        apply_mean_lr(optimizer, automanip_lr)"), 1)
            self.assertEqual(source.count("            apply_mean_lr(optimizer, lrnow)"), 1)
            self.assertLess(source.index("initialize_residual_std(agent, args, logger)", source.index('if __name__ == "__main__"')), source.index("mean_lr_initialize(optimizer, agent, logger)", source.index('if __name__ == "__main__"')))
            # Execute emitted helpers, not an alternate runtime implementation.
            names = {"apply_mean_lr", "configure_mean_lr", "mean_lr_initialize", "audit_mean_lr"}
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            namespace = dict(torch=torch, math=math, time=time, hashlib=hashlib, Path=Path, json=json,
                             __file__=__file__, AUTOMANIP_MEAN_LR_POWER=power,
                             AUTOMANIP_MEAN_LR_PATCH_SHA256="fixture-patch", AUTOMANIP_MEAN_LR_BASE_SHA256=BASE_SHA256,
                             AUTOMANIP_RESIDUAL_LR_CALIBRATION=dict(dataset_sha256="fixture-data", bc_actor_weights_sha256="fixture-actor",
                                 training_episode_indices=[1, 3], excluded_validation_episode_indices=[2],
                                 residual_forward_rows=100, calibration_wall_seconds=.01,
                                 residual_patch_source_sha256="fixture-residual", original_std=[math.exp(-.5)]*8, residual_rms=[.1]*8))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "emitted-helpers", "exec"), namespace)
            agent = Agent(.1)
            optimizer = torch.optim.Adam(agent.parameters(), lr=.0003, eps=1e-5)
            tags = []
            logger = SimpleNamespace(add_scalar=lambda key, value, step: tags.append((key, value, step)))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                namespace["mean_lr_initialize"](optimizer, agent, logger)
                namespace["apply_mean_lr"](optimizer, .0002)
                namespace["audit_mean_lr"](optimizer, logger, 51200, 2)
            lines = output.getvalue().splitlines()
            initialization = json.loads(lines[0].split(" ", 1)[1])
            active = json.loads(lines[1].split(" ", 1)[1])
            self.assertEqual(initialization["additional_bc_forward_rows"], 0)
            self.assertEqual(initialization["training_episode_indices"], [1, 3])
            self.assertEqual(initialization["excluded_validation_episode_indices"], [2])
            self.assertEqual(len(initialization["initial_std"]), 8)
            self.assertEqual(initialization["initial_std_sha256"], hashlib.sha256(json.dumps(initialization["initial_std"], separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(active["critic_lr"], .0002)
            self.assertEqual(active["logstd_lr"], .0002)
            self.assertAlmostEqual(active["mean_lr"], .0002*initialization["mean_lr_multiplier"])
        with self.assertRaises(ValueError):
            build_payload(base+b"\n", 1)

    def test_reject_initialized_optimizer_or_invalid_partition(self):
        agent = Agent()
        optimizer = torch.optim.Adam(agent.parameters())
        agent.actor_logstd.grad = torch.ones_like(agent.actor_logstd)
        optimizer.step()
        with self.assertRaises(ValueError):
            configure_mean_lr(optimizer, agent, 1)


if __name__ == "__main__":
    unittest.main()
