import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from research.discovery.contraction_v1 import closed_loop_jacobian, contraction_loss
from research.discovery.state_variance_v1 import LogStdResidual, gaussian, wrap_agent


def public_agent_class():
    # Only the already permitted pinned public baseline class; no simulator or
    # candidate-provided checkpoint/pickle is imported by these CPU tests.
    tree = ast.parse(Path("third_party/maniskill/ppo.py").read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in ("Agent", "layer_init")]
    namespace = dict(np=np, torch=torch, nn=nn, Normal=Normal)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "public-ppo-agent", "exec"), namespace)
    return namespace["Agent"]


class FrontierTests(unittest.TestCase):
    def test_closed_loop_chain_rule_and_actor_gradients(self):
        model, actor = nn.Linear(3, 2, bias=False), nn.Linear(2, 1, bias=False)
        a = torch.tensor([[1., .16], [0., .8]])
        b = torch.tensor([[.2], [1.]])
        k = torch.tensor([[-.5, -.5]])
        with torch.no_grad():
            model.weight.copy_(torch.cat((a-torch.eye(2), b), 1))
            actor.weight.copy_(k)
        model.requires_grad_(False)
        state = torch.tensor([[.1, -.1], [-.1, .1]])
        jacobian = closed_loop_jacobian(model, actor, state)
        torch.testing.assert_close(jacobian, (a+b@k).expand(2, 2, 2))
        loss = contraction_loss(model, actor, state)
        loss.backward()
        self.assertIsNotNone(actor.weight.grad)
        self.assertTrue(torch.isfinite(actor.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_euclidean_one_step_contraction_is_stronger_than_stability(self):
        stable = torch.tensor([[.7, 0.], [-1.5, 0.]])
        self.assertLess(float(torch.linalg.eigvals(stable).abs().max()), 1.)
        self.assertGreater(float(torch.linalg.svdvals(stable).max()), 1.)

    def test_wrapped_public_agent_initialization_actions_and_rng_match(self):
        torch.set_num_threads(2)
        base_class = public_agent_class()
        envs = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)),
                               single_action_space=SimpleNamespace(shape=(8,)))
        torch.manual_seed(29)
        baseline = base_class(envs)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(29)
        candidate = wrap_agent(base_class)(envs)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
        obs = torch.randn(17, 48)
        action = torch.randn(17, 8)
        expected = baseline.get_action_and_value(obs, action)
        actual = candidate.get_action_and_value(obs, action)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(baseline.get_action(obs, True), candidate.get_action(obs, True), rtol=0, atol=0)
        torch.manual_seed(31)
        expected_sample = baseline.get_action(obs)
        torch.manual_seed(31)
        actual_sample = candidate.get_action(obs)
        torch.testing.assert_close(expected_sample, actual_sample, rtol=0, atol=0)

    def test_variance_changes_without_mean_and_density_is_finite(self):
        residual = LogStdResidual(3, 2, width=4)
        obs = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        mean = torch.zeros(2, 2)
        base = torch.full((1, 2), -.5, requires_grad=True)
        with torch.no_grad():
            residual.net[0].weight.zero_()
            residual.net[0].bias.zero_()
            residual.net[0].weight[0, 0] = 1.
            residual.net[-1].weight[:, 0] = -2.
        distribution = gaussian(mean, base, residual, obs)
        self.assertTrue((distribution.scale[1] < distribution.scale[0]).all())
        torch.testing.assert_close(distribution.loc, mean)
        loss = -distribution.log_prob(torch.full_like(mean, 1000.)).mean()-distribution.entropy().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(base.grad).all())


if __name__ == "__main__":
    unittest.main()
