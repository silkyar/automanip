import ast
import copy
from dataclasses import dataclass
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import textwrap
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from research.discovery.critic_kl_completion_v2 import METHODS, build_payload, critic_only_minibatch, strip_additions


def run_embedded_updates(method, target_kl):
    base = Path("research/discovery/upstream/precision-constant-base.py").read_bytes()
    payload = build_payload(base, method).decode()
    tree = ast.parse(payload)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and getattr(node, "name", "") in ("Agent", "layer_init", "critic_only_minibatch")]
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(name, ast.Name) and name.id.startswith("AUTOMANIP_") for name in node.targets)]
    namespace = dict(torch=torch, nn=nn, np=np, Normal=Normal, time=time, json=json)
    exec(compile(ast.Module(body=definitions+constants, type_ignores=[]), "public-kl-value-definitions", "exec"), namespace)
    torch.manual_seed(23)
    np.random.seed(23)
    envs = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)), single_action_space=SimpleNamespace(shape=(8,)))
    agent = namespace["Agent"](envs)
    optimizer = torch.optim.Adam(agent.parameters(), lr=.0003, eps=1e-5)
    generator = torch.Generator().manual_seed(24)
    obs, actions = torch.randn(128, 48, generator=generator), torch.randn(128, 8, generator=generator)
    with torch.no_grad():
        _, old_logprob, _, old_value = agent.get_action_and_value(obs, actions)
    targets = obs[:, 0].sin()+.5*obs[:, 1]
    initial_mse = float((agent.critic(obs).flatten()-targets).square().mean().detach())
    args = SimpleNamespace(batch_size=128, minibatch_size=64, update_epochs=8, target_kl=target_kl,
                           norm_adv=True, clip_coef=.2, clip_vloss=False, ent_coef=.003, vf_coef=.5, max_grad_norm=.5)
    namespace.update(agent=agent, optimizer=optimizer, args=args, device=torch.device("cpu"),
                     b_obs=obs, b_actions=actions, b_logprobs=old_logprob.detach(),
                     b_advantages=torch.randn(128, generator=generator), b_returns=targets,
                     b_values=old_value.flatten().detach(), global_step=51200,
                     logger=SimpleNamespace(add_scalar=lambda *values: None), automanip_deadline=float("inf"),
                     automanip_optimizer_steps=0, automanip_cumulative_critic_only_calls=0,
                     automanip_cumulative_critic_only_rows=0)
    start = payload.index("        # Optimizing the policy and value network\n")
    end = payload.index("        y_pred, y_true = ", start)
    block = textwrap.dedent(payload[start:end])
    with redirect_stdout(io.StringIO()):
        exec(compile(block, "exact-embedded-ppo-update-block", "exec"), namespace)
    with torch.no_grad():
        _, logprob, entropy, _ = agent.get_action_and_value(obs, actions)
        final_mse = float((agent.critic(obs).flatten()-targets).square().mean())
    actor_state = {name: copy.deepcopy(optimizer.state[parameter]) for name, parameter in agent.named_parameters()
                   if name == "actor_logstd" or name.startswith("actor_mean.")}
    return dict(agent=agent, actor_optimizer_state=actor_state, logprob=logprob, entropy=entropy,
                numpy_state=np.random.get_state(), torch_rng=torch.get_rng_state().clone(), report=namespace["automanip_kl_value_report"],
                initial_mse=initial_mse, final_mse=final_mse)


class CriticCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_adam_zero_gradient_counterexample_and_none_gradient_fix(self):
        actor, critic = nn.Parameter(torch.tensor(1.)), nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.Adam([actor, critic], lr=.1)
        (actor.square()+critic.square()).backward()
        optimizer.step()
        actor_before = actor.detach().clone()
        optimizer.zero_grad(set_to_none=False)
        critic.square().backward()
        optimizer.step()
        self.assertFalse(torch.equal(actor, actor_before))
        actor_before = actor.detach().clone()
        actor_state = copy.deepcopy(optimizer.state[actor])
        optimizer.zero_grad(set_to_none=True)
        critic.square().backward()
        optimizer.step()
        torch.testing.assert_close(actor, actor_before, rtol=0, atol=0)
        for key, value in actor_state.items():
            torch.testing.assert_close(optimizer.state[actor][key], value, rtol=0, atol=0)

    def test_kl_stop_candidate_finishes_8_epochs_actor_ratios_and_moments_frozen(self):
        control = run_embedded_updates("scratch-kl-stop-control", 1e-12)
        candidate = run_embedded_updates("scratch-kl-stop-critic", 1e-12)
        self.assertEqual(control["report"]["joint_actor_critic_optimizer_calls"], 1)
        self.assertEqual(candidate["report"]["joint_actor_critic_optimizer_calls"], 1)
        self.assertEqual(candidate["report"]["critic_only_optimizer_calls"], 15)
        self.assertEqual(candidate["report"]["critic_optimizer_calls"], 16)
        self.assertEqual(candidate["report"]["reached_epoch_count"], 8)
        self.assertEqual(control["report"]["kl_stop_epoch_0based"], candidate["report"]["kl_stop_epoch_0based"])
        self.assertEqual(control["report"]["kl_stop_minibatch_0based"], candidate["report"]["kl_stop_minibatch_0based"])
        self.assertEqual(control["report"]["observed_kl"], candidate["report"]["observed_kl"])
        self.assertLess(candidate["final_mse"], control["final_mse"])
        for name, parameter in control["agent"].named_parameters():
            if name == "actor_logstd" or name.startswith("actor_mean."):
                torch.testing.assert_close(parameter, dict(candidate["agent"].named_parameters())[name], rtol=0, atol=0)
                for key, value in control["actor_optimizer_state"][name].items():
                    torch.testing.assert_close(value, candidate["actor_optimizer_state"][name][key], rtol=0, atol=0)
        torch.testing.assert_close(control["logprob"], candidate["logprob"], rtol=0, atol=0)
        torch.testing.assert_close(control["entropy"], candidate["entropy"], rtol=0, atol=0)
        np.testing.assert_array_equal(control["numpy_state"][1], candidate["numpy_state"][1])
        self.assertEqual(control["numpy_state"][2:], candidate["numpy_state"][2:])
        torch.testing.assert_close(control["torch_rng"], candidate["torch_rng"], rtol=0, atol=0)

    def test_no_kl_stop_preserves_identical_joint_updates_and_parameters(self):
        control = run_embedded_updates("scratch-kl-stop-control", None)
        candidate = run_embedded_updates("scratch-kl-stop-critic", None)
        self.assertEqual(candidate["report"]["critic_only_optimizer_calls"], 0)
        self.assertEqual(candidate["report"]["joint_actor_critic_optimizer_calls"], 16)
        for key, value in control["agent"].state_dict().items():
            torch.testing.assert_close(value, candidate["agent"].state_dict()[key], atol=0, rtol=0)

    def test_deadline_skip_performs_no_optimizer_call_and_clears_stale_actor_grads(self):
        agent = SimpleNamespace(actor_mean=nn.Linear(3, 2), actor_logstd=nn.Parameter(torch.zeros(1, 2)), critic=nn.Linear(3, 1))
        optimizer = torch.optim.Adam([*agent.actor_mean.parameters(), agent.actor_logstd, *agent.critic.parameters()])
        for parameter in agent.actor_mean.parameters():
            parameter.grad = torch.ones_like(parameter)
        stats = critic_only_minibatch(agent, optimizer, torch.zeros(5, 3), torch.zeros(5), .5, .5, 0.)
        self.assertEqual(stats["optimizer_calls"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in agent.actor_mean.parameters()))

    def test_exact_strip_schema_and_no_target_or_interface_change(self):
        base = Path("research/discovery/upstream/precision-constant-base.py").read_bytes()
        original_agent = next(node for node in ast.parse(base).body if isinstance(node, ast.ClassDef) and node.name == "Agent")
        for method in METHODS:
            payload = build_payload(base, method)
            self.assertEqual(strip_additions(payload.decode()).encode(), base)
            agent = next(node for node in ast.parse(payload).body if isinstance(node, ast.ClassDef) and node.name == "Agent")
            self.assertEqual(ast.dump(agent), ast.dump(original_agent))
            self.assertEqual(payload.count(b"returns = advantages + values"), 1)
            self.assertIn(b"args.checkpoint is not None", payload)
            self.assertNotIn(b"from automanip.", payload)
            self.assertNotIn(b"from research.", payload)


if __name__ == "__main__":
    unittest.main()
