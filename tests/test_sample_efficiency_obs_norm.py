import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK

SOURCE = Path(__file__).resolve().parents[1]/'research/sample-efficiency-20261031'
sys.path.insert(0, str(SOURCE))
try:
    import tensorboard  # noqa: F401
except ImportError:  # training-only logger; not needed for these numerical checks
    import types
    stub = types.ModuleType('torch.utils.tensorboard')
    stub.SummaryWriter = object
    sys.modules['torch.utils.tensorboard'] = stub
spec = importlib.util.spec_from_file_location('se_ppo', SOURCE/'ppo.py')
ppo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo)
from export_helper import export_actor  # noqa: E402


class FakeEnvs:
    single_observation_space = SimpleNamespace(shape=(48,))
    single_action_space = SimpleNamespace(shape=(8,))


def observations(generator, n):
    # Mixed scales like StackCube state: joint angles, small velocities, metre-scale poses, a near-constant column.
    scale = torch.tensor([1.]*9+[.05]*9+[.3]*28+[1e-3, 0.], dtype=torch.float64)
    offset = torch.linspace(-.5, .8, 48, dtype=torch.float64)
    return (torch.randn(n, 48, generator=generator, dtype=torch.float64)*scale+offset).float()


class ObsNormTests(unittest.TestCase):
    def test_merged_batches_match_full_data_statistics(self):
        generator = torch.Generator().manual_seed(3)
        batches = [observations(generator, n) for n in (50, 700, 1200)]
        norm = ppo.RunningObsNorm(48, 'cpu', True)
        for batch in batches:
            norm.update(batch.reshape(-1, 10, 48) if len(batch) % 10 == 0 else batch)
        full = torch.cat(batches).double()
        torch.testing.assert_close(norm.mean, full.mean(0), rtol=0, atol=1e-12)
        torch.testing.assert_close(norm.var, full.var(0, unbiased=False), rtol=0, atol=1e-12)
        self.assertEqual(norm.count, 1950)

    def test_disabled_norm_is_identity_and_fold_is_unchanged_copy(self):
        torch.manual_seed(0)
        agent = ppo.Agent(FakeEnvs())
        norm = ppo.RunningObsNorm(48, 'cpu', False)
        x = observations(torch.Generator().manual_seed(1), 64)
        norm.update(x)
        self.assertIs(norm(x), x)
        folded = ppo.fold_obs_norm(agent.actor_mean, norm)
        for a, b in zip(agent.actor_mean.parameters(), folded.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_exported_folded_actor_matches_training_policy_on_raw_observations(self):
        torch.manual_seed(5)
        agent = ppo.Agent(FakeEnvs())
        with torch.no_grad():  # a nontrivial, non-initial output layer
            agent.actor_mean[-1].weight.normal_(0, .05)
        norm = ppo.RunningObsNorm(48, 'cpu', True)
        generator = torch.Generator().manual_seed(7)
        for _ in range(3):
            norm.update(observations(generator, 4096))
        raw = observations(generator, 512)
        with torch.no_grad():
            expected = agent.actor_mean(norm(raw)).numpy()
        with TemporaryDirectory() as directory:
            artifact = export_actor(SimpleNamespace(net=ppo.fold_obs_norm(agent.actor_mean, norm), hidden_sizes=[256, 256, 256]),
                                    Path(directory)/'actor')
            schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
            policy = load_policy(Path(artifact['path']), schema)
            self.assertEqual(policy.weights_sha256, artifact['weights_sha256'])
            actual = NumpyPolicy(policy, schema)
            # Evaluator clips to the action box; compare the unclipped mean where it is inside the box.
            got = np.asarray(actual.act(raw.numpy()))
        inside = np.abs(expected) < 1
        self.assertGreater(inside.mean(), .5)
        np.testing.assert_allclose(got[inside], expected[inside], rtol=0, atol=2e-5)


if __name__ == '__main__':
    unittest.main()


class HeavyTailExplorationTests(unittest.TestCase):
    def test_default_agent_matches_original_gaussian_and_initialization(self):
        torch.manual_seed(2); plain = ppo.Agent(FakeEnvs())
        torch.manual_seed(2); explicit = ppo.Agent(FakeEnvs(), 256, False, "none", 2.0)
        for a, b in zip(plain.parameters(), explicit.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = torch.randn(32, 48); action = torch.randn(32, 8)
        _, logprob, entropy, _ = plain.get_action_and_value(x, action)
        mean = plain.actor_mean(x); std = plain.actor_logstd.exp().expand_as(mean)
        reference = torch.distributions.Normal(mean, std)
        torch.testing.assert_close(logprob, reference.log_prob(action).sum(1))
        torch.testing.assert_close(entropy, reference.entropy().sum(1))

    def test_gripper_student_t_density_and_tails(self):
        torch.manual_seed(4)
        agent = ppo.Agent(FakeEnvs(), heavy_tail_dims="gripper", t_df=2.0)
        x = torch.randn(4096, 48)
        mean = agent.actor_mean(x).detach(); std = agent.actor_logstd.exp().expand_as(mean)
        action, logprob, entropy, _ = agent.get_action_and_value(x)
        expected = (torch.distributions.Normal(mean[:, :7], std[:, :7]).log_prob(action[:, :7]).sum(1)
                    + torch.distributions.StudentT(2.0, mean[:, 7], std[:, 7]).log_prob(action[:, 7]))
        torch.testing.assert_close(logprob, expected)
        self.assertTrue(torch.isfinite(entropy).all())
        z = ((action-mean)/std).detach()
        # Tail beyond 3.6 scale units: Gaussian about 3e-4; Student-t(2) about 3.5e-2 (one-sided 1.7e-2 each side).
        self.assertLess((z[:, :7].abs() > 3.6).float().mean().item(), .002)
        self.assertGreater((z[:, 7].abs() > 3.6).float().mean().item(), .02)


class BinaryGripperTests(unittest.TestCase):
    def test_bernoulli_gripper_logprob_and_binary_samples(self):
        torch.manual_seed(6)
        agent = ppo.Agent(FakeEnvs(), binary_gripper=True)
        with torch.no_grad():
            agent.actor_mean[-1].bias[-1] = .7
        x = torch.randn(2048, 48)
        action, logprob, entropy, _ = agent.get_action_and_value(x)
        self.assertTrue(set(action[:, 7].unique().tolist()) <= {-1., 1.})
        mean = agent.actor_mean(x).detach(); std = agent.actor_logstd.exp().expand_as(mean)
        p = torch.sigmoid(mean[:, 7])
        expected = (torch.distributions.Normal(mean[:, :7], std[:, :7]).log_prob(action[:, :7]).sum(1)
                    + torch.where(action[:, 7] > 0, p.log(), (1-p).log()))
        torch.testing.assert_close(logprob, expected)
        self.assertAlmostEqual((action[:, 7] > 0).float().mean().item(), p.mean().item(), delta=.04)
        _, again, _, _ = agent.get_action_and_value(x, action)
        torch.testing.assert_close(again, logprob)
        self.assertTrue(torch.isfinite(entropy).all())

    def test_exported_binary_actor_gives_gripper_mode_and_unchanged_arm(self):
        torch.manual_seed(8)
        agent = ppo.Agent(FakeEnvs(), binary_gripper=True)
        with torch.no_grad():
            agent.actor_mean[-1].weight.normal_(0, .05)
        norm = ppo.RunningObsNorm(48, 'cpu', True)
        generator = torch.Generator().manual_seed(9)
        norm.update(observations(generator, 4096))
        raw = observations(generator, 512)
        with torch.no_grad():
            mean = agent.actor_mean(norm(raw)).numpy()
        with TemporaryDirectory() as directory:
            artifact = export_actor(SimpleNamespace(net=agent.export_net(norm), hidden_sizes=[256, 256, 256]),
                                    Path(directory)/'actor')
            schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
            got = np.asarray(NumpyPolicy(load_policy(Path(artifact['path']), schema), schema).act(raw.numpy()))
        decided = np.abs(mean[:, 7]) > .05
        self.assertGreater(decided.mean(), .5)
        np.testing.assert_array_equal(got[decided, 7], np.sign(mean[decided, 7]))
        inside = np.abs(mean[:, :7]) < 1
        np.testing.assert_allclose(got[:, :7][inside], mean[:, :7][inside], rtol=0, atol=2e-5)


class ExplorationFloorTests(unittest.TestCase):
    def test_gripper_std_floor_only_touches_gripper(self):
        torch.manual_seed(10)
        agent = ppo.Agent(FakeEnvs(), gripper_std_floor=.5)
        with torch.no_grad():
            agent.actor_logstd.fill_(-3.)
        x = torch.randn(64, 48); action = torch.randn(64, 8)
        _, logprob, _, _ = agent.get_action_and_value(x, action)
        mean = agent.actor_mean(x)
        std = torch.full((1, 8), float(np.exp(-3.))); std[0, 7] = .5
        torch.testing.assert_close(logprob, torch.distributions.Normal(mean, std.expand_as(mean)).log_prob(action).sum(1))
        torch.testing.assert_close(agent.last_gripper_param, mean[:, 7])

    def test_binary_eps_floor_keeps_open_probability(self):
        torch.manual_seed(12)
        agent = ppo.Agent(FakeEnvs(), binary_gripper=True, binary_eps=.05)
        with torch.no_grad():
            agent.actor_mean[-1].bias[-1] = -40.
        x = torch.randn(20000, 48)
        action, logprob, _, _ = agent.get_action_and_value(x)
        self.assertAlmostEqual((action[:, 7] > 0).float().mean().item(), .05, delta=.006)
        opened = action[:, 7] > 0
        mean = agent.actor_mean(x).detach(); std = agent.actor_logstd.exp().expand_as(mean)
        arm = torch.distributions.Normal(mean[:, :7], std[:, :7]).log_prob(action[:, :7]).sum(1)
        p = .05 + .9*torch.sigmoid(mean[:, 7])
        torch.testing.assert_close(logprob, arm + torch.where(opened, p.log(), (1-p).log()))

    def test_sticky_binary_gripper_mixture_logprob(self):
        torch.manual_seed(14)
        agent = ppo.Agent(FakeEnvs(), binary_gripper=True, binary_eps=.02, binary_sticky=.5)
        x = torch.randn(30000, 48)
        prev = torch.tensor([-1., 0., 1.]).repeat(10000)
        action, logprob, _, _ = agent.get_action_and_value(x, prev_grip=prev)
        q = .02 + .96*torch.sigmoid(agent.actor_mean(x)[:, 7].detach())
        p = torch.where(prev > 0, .5+.5*q, torch.where(prev < 0, .5*q, q))
        opened = action[:, 7] > 0
        mean = agent.actor_mean(x).detach(); std = agent.actor_logstd.exp().expand_as(mean)
        arm = torch.distributions.Normal(mean[:, :7], std[:, :7]).log_prob(action[:, :7]).sum(1)
        torch.testing.assert_close(logprob, arm + torch.where(opened, p.log(), (1-p).log()))
        for value in (-1., 0., 1.):
            chosen = prev == value
            self.assertAlmostEqual(opened[chosen].float().mean().item(), p[chosen].mean().item(), delta=.02)
        _, plain, _, _ = ppo.Agent(FakeEnvs(), binary_gripper=True, binary_eps=.02).get_action_and_value(x, action, prev)
        self.assertTrue(torch.isfinite(plain).all())
