"""Synthetic CPU checks only; no policy is trained in a real simulator here."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import train
    from automanip.inference import MLPPolicy
    from automanip.policy_io import load_policy, PolicySchema
    from automanip.task_spec import TASK


@unittest.skipUnless(HAS_TORCH, "PPO tests require optional PyTorch dependency")
class PPOTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(train.PPOConfig(hidden_sizes=(8,), num_envs=1,
                                       total_timesteps=4, num_steps=2,
                                       num_minibatches=1, update_epochs=1), **kwargs)

    def test_default_config_and_invalid_budgets(self):
        train.PPOConfig().validate()
        for kwargs in ({"seed": True}, {"max_seconds": math.inf}, {"num_envs": 0},
                       {"gamma": 1.1}, {"hidden_sizes": [512]}, {"partial_reset": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.config(**kwargs).validate()

    def test_reset_boundaries_stop_gae_recursion(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[3.0], [4.0]])
        advantages, returns = train.generalized_advantage(
            rewards, values, next_values, torch.tensor([[True], [False]]), 0.5, 1)
        torch.testing.assert_close(advantages, torch.tensor([[2.5], [4.0]]))
        torch.testing.assert_close(returns, advantages)

    def test_bootstrap_uses_final_observation_and_terminal_option(self):
        class IdentityCritic:
            def critic(self, obs):
                return obs[:, :1]
        next_obs = torch.tensor([[100.0], [3.0]])  # 100 is the reset observation
        term = torch.tensor([True, False])
        trunc = torch.tensor([False, False])
        info = {"final_observation": torch.tensor([[7.0], [999.0]]),
                "_final_observation": torch.tensor([True, False])}
        torch.testing.assert_close(train.successor_values(IdentityCritic(), next_obs, term, trunc, info),
                                   torch.tensor([7.0, 3.0]))
        torch.testing.assert_close(train.successor_values(IdentityCritic(), next_obs, term, trunc, info, False),
                                   torch.tensor([0.0, 3.0]))
        torch.testing.assert_close(train.successor_values(IdentityCritic(), next_obs, trunc, term, info, False),
                                   torch.tensor([7.0, 3.0]))
        with self.assertRaises(ValueError):
            train.successor_values(IdentityCritic(), next_obs, term, trunc, {})

    def test_clipped_surrogate_known_values(self):
        loss, kl = train.clipped_policy_loss(torch.tensor([2.0, 0.5]).log(),
                                             torch.zeros(2), torch.tensor([1.0, -1.0]), 0.2)
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertGreater(kl.item(), 0)

    def test_reproducible_random_initialization(self):
        torch.manual_seed(42)
        first = train.Agent(self.config())
        torch.manual_seed(42)
        second = train.Agent(self.config())
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)

    def test_optimizer_updates_and_deadline(self):
        torch.manual_seed(9)
        config = self.config()
        agent = train.Agent(config)
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
        obs = torch.randn(4, TASK.observation_dim)
        with torch.no_grad():
            actions, logprob, _, values = agent.action_and_value(obs)
        batch = (obs, actions, logprob, torch.tensor([1.0, -1.0, 2.0, -2.0]), values+1)
        before = [p.detach().clone() for p in agent.parameters()]
        count, stats = train.ppo_update(agent, optimizer, batch, config)
        self.assertEqual(count, 1)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.parameters())))
        self.assertTrue(all(math.isfinite(v) for v in stats.values()))
        self.assertEqual(train.ppo_update(agent, optimizer, batch, config, deadline=0)[0], 0)

    def test_actor_export_matches_trusted_inference(self):
        torch.manual_seed(2)
        agent = train.Agent(self.config())
        obs = torch.randn(3, TASK.observation_dim)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"policy"
            digest = train.export_actor(agent, directory)
            schema = PolicySchema(TASK.fingerprint, TASK.observation_dim)
            artifact = load_policy(directory, schema)
            actual = MLPPolicy(artifact, schema).act(obs.tolist())
            with torch.no_grad():
                expected = agent.actor(obs).clamp(-1, 1)
            torch.testing.assert_close(torch.tensor(actual), expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(digest, artifact.weights_sha256)
            with self.assertRaises(FileExistsError):
                train.export_actor(agent, directory)

    def test_synthetic_training_run_records_budget_and_safe_artifact(self):
        class FakeEnv:
            closed = False
            def reset(self, seed):
                self.seed = seed
                return torch.zeros(1, TASK.observation_dim), {}
            def step(self, actions):
                obs = torch.ones(1, TASK.observation_dim)
                return obs, torch.ones(1), torch.tensor([False]), torch.tensor([False]), {}
            def close(self):
                self.closed = True
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as temp, patch.object(train, "make_environment", return_value=env):
            output = Path(temp)/"run"
            record = train.run(self.config(), "physx_cpu", output, "synthetic unit check")
            self.assertEqual(record["environment_transitions"], 4)
            self.assertEqual(record["learner_updates"], 2)
            self.assertEqual(record["status"], "step_budget")
            self.assertEqual(record["decision"], "unevaluated")
            self.assertIsNone(record["evaluation"])
            self.assertEqual(json.loads((output/"run.json").read_text())["checkpoint_sha256"], record["checkpoint_sha256"])
            self.assertTrue((output/"train_source.py").exists())
            self.assertTrue((output/"policy"/"weights.json").exists())
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
