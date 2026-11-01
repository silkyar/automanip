import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from research.discovery.amp_collect_v1 import collect_batch
from research.discovery.amp_precision_v1 import METHODS, build_assist_trainer, strip_assist_additions
from research.discovery.amp_v1 import (
    DATA_FORMAT, PRODUCTION_PINS, MotionPrior, StudentActor, actual_successor,
    amp_style_reward, discriminator_loss, export_actor, load_demos,
    mix_training_reward, read_actor,
)


def dataset_fixture(directory, cpu_smoke=False):
    obs = np.arange(6*51*48, dtype=np.float32).reshape(6, 51, 48)/10000
    actions = np.zeros((6, 50, 8), dtype=np.float32)
    success = np.array([True]*5+[False])
    np.savez_compressed(directory/"trajectories.npz", observations=obs, actions=actions,
                        task_rewards=np.zeros((6, 50), dtype=np.float32), final_success=success,
                        public_training_seed=np.arange(6, dtype=np.int64)+100000)
    metadata = dict(format=DATA_FORMAT, sha256=hashlib.sha256((directory/"trajectories.npz").read_bytes()).hexdigest(),
                    env_id="StackCube-v1", observation_dim=48, action_dim=8, horizon=50,
                    control_mode="pd_joint_delta_pos", obs_mode="state", reward_mode="normalized_dense",
                    simulation_backend="physx_cuda", source_partition="public-training", versions=PRODUCTION_PINS,
                    completed_episodes=6, successful_episodes=5)
    if cpu_smoke:
        metadata.update(simulation_backend="physx_cpu", purpose="local-cpu-dataflow-smoke", comparison_eligible=False,
                        versions=dict(PRODUCTION_PINS, sapien="3.0.4.dev20261009+7dba23f1"))
    (directory/"dataset.json").write_text(json.dumps(metadata))
    return obs


class AmpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_safe_bc_actor_numeric_roundtrip_and_clipping(self):
        torch.manual_seed(7)
        actor = StudentActor((32, 32, 32))
        obs = torch.randn(17, 48)*1000
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"actor"
            export_actor(actor, directory)
            recovered, manifest = read_actor(directory)
            torch.testing.assert_close(actor.deterministic(obs), recovered.deterministic(obs), rtol=0, atol=0)
            data = json.loads((directory/"weights.json").read_bytes())
            independent = obs.numpy()
            for i in range(4):
                w = data[f"layers.{i}.weight"]
                b = data[f"layers.{i}.bias"]
                independent = independent@np.array(w["values"], dtype=np.float32).reshape(w["shape"]).T+np.array(b["values"], dtype=np.float32)
                if i < 3:
                    independent = np.tanh(independent)
            np.testing.assert_allclose(actor.deterministic(obs).detach().numpy(), independent.clip(-1, 1), atol=2e-6, rtol=0)
            self.assertEqual(manifest["hidden_sizes"], [32, 32, 32])

    def test_demo_episode_split_retains_failures_without_crossing_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            original = dataset_fixture(path)
            data = load_demos(path)
            self.assertEqual(len(data.observations), 6)
            self.assertFalse(bool(data.final_success[-1]))
            self.assertFalse(set(data.train_episode_indices) & set(data.validation_episode_indices))
            self.assertNotIn(5, data.train_episode_indices+data.validation_episode_indices)
            obs, following = data.training_pairs()
            np.testing.assert_allclose(following.numpy()-obs.numpy(), np.full_like(obs.numpy(), .0048), atol=1e-7)

    def test_cpu_smoke_requires_explicit_mode_and_cannot_pass_production_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            dataset_fixture(path, cpu_smoke=True)
            with self.assertRaises(ValueError):
                load_demos(path)
            smoke = load_demos(path, allow_local_cpu_smoke=True)
            self.assertIs(smoke.metadata["comparison_eligible"], False)

    def test_final_successor_and_amp_reward_never_use_reset_jump(self):
        following = torch.zeros(2, 48)
        final = torch.full((2, 48), 7.)
        done = torch.tensor([False, True])
        actual = actual_successor(following, done, {"final_observation": final, "_final_observation": done})
        torch.testing.assert_close(actual[0], following[0])
        torch.testing.assert_close(actual[1], final[1])
        with self.assertRaises(ValueError):
            actual_successor(following, done, {})

    def test_least_squares_expert_gradient_penalty_and_detachment(self):
        discriminator = nn.Linear(96, 1, bias=False)
        with torch.no_grad():
            discriminator.weight.fill_(.2)
        expert = torch.zeros(4, 96, requires_grad=True)
        policy = torch.ones(4, 96, requires_grad=True)
        class Flatten(nn.Module):
            def forward(self, x):
                return discriminator(x).squeeze(-1)
        loss, stats = discriminator_loss(Flatten(), expert, policy)
        self.assertAlmostEqual(stats["gradient_penalty"], 96*.2**2, places=5)
        self.assertAlmostEqual(float(loss), 1+(96*.2+1)**2+5*(96*.2**2), places=3)
        loss.backward()
        self.assertIsNone(expert.grad)
        self.assertIsNone(policy.grad)
        self.assertTrue(torch.isfinite(discriminator.weight.grad).all())

    def test_style_bounds_train_only_reward_and_private_rng(self):
        scores = torch.tensor([-3., -1., 0., 1., 3., 5.], requires_grad=True)
        style = amp_style_reward(scores)
        torch.testing.assert_close(style, torch.tensor([0., 0., .75, 1., 0., 0.]))
        task = torch.ones(6, requires_grad=True)
        mixed = mix_training_reward(task, style)
        torch.testing.assert_close(task, torch.ones(6))
        self.assertFalse(mixed.requires_grad)
        obs = torch.randn(32, 48)
        following = .8*obs
        prior = MotionPrior(obs, following, seed=1, capacity=40)
        before = torch.get_rng_state().clone()
        prior.reward_and_remember(obs, -following)
        prior.reward_and_remember(obs[:16], -following[:16])
        stats = prior.update(rounds=1, batch_size=8)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(stats["replay_rows"], 40)
        self.assertEqual(stats["learner_transition_rows"], 48)
        self.assertEqual(stats["discriminator_sample_rows"], 16)

    def test_collector_retains_true_final_state_for_complete_full_horizon(self):
        class FakeEnv:
            def reset(self, seed):
                self.step_index = 0
                return torch.zeros(2, 48), {}
            def step(self, action):
                self.step_index += 1
                return (torch.full((2, 48), float(self.step_index)), torch.ones(2), torch.zeros(2, dtype=torch.bool),
                        torch.full((2,), self.step_index == 50), {"episode": {"success_at_end": torch.tensor([True, False])}})
        class FakeActor:
            def deterministic(self, obs):
                return torch.zeros(len(obs), 8)
        progress = dict(explicit_reset_calls=0, explicit_reset_episodes=0, vector_step_calls=0,
                        environment_transitions=0, partial=None)
        data = collect_batch(FakeEnv(), FakeActor(), 1001, progress)
        self.assertEqual(data["observations"].shape, (2, 51, 48))
        self.assertTrue((data["observations"][:, -1] == 50).all())
        self.assertEqual(progress["environment_transitions"], 100)
        self.assertEqual(data["final_success"].tolist(), [True, False])

    def test_assisted_payloads_exact_strip_and_ordinary_agent_schema(self):
        base = Path("research/discovery/upstream/precision-constant-base.py").read_bytes()
        original = next(n for n in ast.parse(base).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
        for method in METHODS:
            payload = build_assist_trainer(base, method)
            self.assertEqual(strip_assist_additions(payload.decode()).encode(), base)
            agent = next(n for n in ast.parse(payload).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            self.assertEqual(ast.dump(agent), ast.dump(original))
            self.assertNotIn(b"from research.", payload)
            self.assertNotIn(b"from automanip.", payload)
            self.assertIn(b"args.checkpoint is not None", payload)
            self.assertIn(b"rewards[step] = reward.view(-1) * args.reward_scale", payload)

    def test_self_contained_runtime_initialization_keeps_critic_std_and_schema(self):
        import copy
        base = Path("research/discovery/upstream/precision-constant-base.py").read_bytes()
        payload = build_assist_trainer(base, "assist-bc-ppo-amp")
        # Execute only public classes/helper definitions, never simulator imports
        # or the generated trainer main, to exercise the actual embedded runtime.
        tree = ast.parse(payload)
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and getattr(n, "name", "") not in ("Args", "Logger")]
        namespace = dict(torch=torch, nn=nn, np=np, dataclass=dataclass, Path=Path,
                         hashlib=hashlib, json=json, TASK_FINGERPRINT="b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f",
                         DATA_FORMAT=DATA_FORMAT, PRODUCTION_PINS=PRODUCTION_PINS, __name__=__name__)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "assist-runtime-cpu-fixture", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            dataset_fixture(path)
            metadata = json.loads((path/"dataset.json").read_text())
            metadata.update(teacher_prefix_transitions=1, teacher_prefix_trainer_seconds=1., teacher_prefix_dollars=0.,
                            environment_transitions=300, failed_completed_episodes=1)
            (path/"dataset.json").write_text(json.dumps(metadata))
            actor = StudentActor()
            artifact = export_actor(actor, path/"bc-actor")
            report = dict(teacher_weights_loaded=False, student_initialization="random", comparison_eligible=True,
                          dataset_sha256=metadata["sha256"], final=artifact,
                          optimizer_updates=1, demonstration_minibatch_rows=1, wall_seconds=1.)
            (path/"bc-report.json").write_text(json.dumps(report))
            args = SimpleNamespace(assist_dataset=str(path), assist_bc_actor=str(path/"bc-actor"),
                                   assist_bc_report=str(path/"bc-report.json"), seed=1)
            envs = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)),
                                   single_action_space=SimpleNamespace(shape=(8,)))
            agent = namespace["Agent"](envs)
            before = copy.deepcopy(agent.state_dict())
            prior = namespace["assist_initialize"](agent, args, True, "cpu")
            self.assertIsNotNone(prior)
            self.assertEqual(set(agent.state_dict()), set(before))
            torch.testing.assert_close(agent.actor_logstd, before["actor_logstd"], rtol=0, atol=0)
            for key, value in before.items():
                if key.startswith("critic."):
                    torch.testing.assert_close(agent.state_dict()[key], value, rtol=0, atol=0)
            probe = torch.randn(8, 48)
            torch.testing.assert_close(agent.actor_mean(probe), actor(probe), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
