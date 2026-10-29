import math
from types import MappingProxyType
import unittest
from automanip.evaluator import evaluate_batch
from automanip.inference import MLPPolicy
from automanip.policy_io import ValidatedPolicy, PolicySchema


class ZeroPolicy:
    def act(self, obs):
        return [[0.0]*7 for _ in obs]


class MockEnvironment:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.resets = 0

    def reset(self, seeds):
        self.resets += 1
        self.seeds = list(seeds)
        return [[0.0] for _ in seeds]

    def step(self, actions):
        frame = self.frames[self.index]
        self.index += 1
        if isinstance(frame, Exception):
            raise frame
        success, term, trunc = frame
        return [[0.0] for _ in success], success, term, trunc


class RolloutTests(unittest.TestCase):
    def test_fixed_horizon_keeps_ever_and_terminal_separate(self):
        env = MockEnvironment([([True, False], [True, False], [False, False]),
                               ([False, True], [False, True], [False, False])])
        first, second = evaluate_batch(env, ZeroPolicy(), [12, 34], horizon=2)
        self.assertTrue(first.outcome.ever_success)
        self.assertFalse(first.outcome.success_at_end)
        self.assertTrue(second.outcome.success_at_end)
        self.assertTrue(first.outcome.terminated)
        self.assertTrue(first.outcome.truncated)
        self.assertEqual(env.resets, 1)
        self.assertEqual(env.seeds, [12, 34])

    def test_shared_crash_invalidates_previous_success(self):
        env = MockEnvironment([([True], [True], [False]), RuntimeError("private seed 123")])
        result, = evaluate_batch(env, ZeroPolicy(), [1], horizon=2)
        self.assertTrue(result.outcome.failed)
        self.assertFalse(result.outcome.success_at_end)
        self.assertEqual(result.error, "RuntimeError")
        self.assertEqual(result.steps, 1)

    def test_early_truncation_fails(self):
        env = MockEnvironment([([True], [False], [True])])
        result, = evaluate_batch(env, ZeroPolicy(), [1], horizon=2)
        self.assertTrue(result.outcome.failed)

    def test_invalid_action_fails_before_simulator_step(self):
        class InvalidPolicy:
            def act(self, obs):
                return [[math.nan]*7]
        env = MockEnvironment([])
        result, = evaluate_batch(env, InvalidPolicy(), [1], horizon=2)
        self.assertTrue(result.outcome.failed)
        self.assertEqual(env.index, 0)

    def test_batch_mismatch_fails(self):
        env = MockEnvironment([([True], [True], [False])])
        self.assertTrue(all(r.outcome.failed for r in evaluate_batch(env, ZeroPolicy(), [1, 2], horizon=1)))

    def test_bad_seed_configuration_is_not_candidate_failure(self):
        with self.assertRaises(ValueError):
            evaluate_batch(None, None, [True])


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.schema = PolicySchema("a"*64, 1, 7)
        tensors = MappingProxyType({"layers.0.weight": (1.0,), "layers.0.bias": (0.0,),
                                   "layers.1.weight": (2.0,)*7, "layers.1.bias": (0.0,)*7})
        self.policy = MLPPolicy(ValidatedPolicy((1,), tensors, "b"*64), self.schema)

    def test_known_network_and_clipping(self):
        self.assertAlmostEqual(self.policy.act([[0.1]])[0][0], 2*math.tanh(0.1))
        self.assertEqual(self.policy.act([[100.0]]), [[1.0]*7])
        self.assertEqual(self.policy.act([[-100.0]]), [[-1.0]*7])

    def test_invalid_observations(self):
        for obs in ([[math.nan]], [[math.inf]], [[1.0, 2.0]]):
            with self.assertRaises(ValueError):
                self.policy.act(obs)
