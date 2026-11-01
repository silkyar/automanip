import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from research.discovery.episodic_risk_cpu_gate_v1 import analytic_check, exact_cvar, outcomes
from research.discovery.episodic_risk_v1 import actor_score_advantages, episode_returns, score_weights, training_diagnostics
from research.discovery.episodic_risk_precision_v1 import BASE_SHA256, METHODS, build_precision_trainer, strip_risk_additions


def complete_batch():
    rewards = torch.arange(32, dtype=torch.float32)[None, :].expand(50, 32)/32
    dones = torch.zeros_like(rewards)
    next_done = torch.ones(32)
    success = torch.ones(32, dtype=torch.bool)
    success[:3] = False
    info = {"_final_info": torch.ones(32, dtype=torch.bool),
            "final_info": {"episode": {"success_at_end": success}}}
    return rewards, dones, next_done, info


class EpisodicRiskTests(unittest.TestCase):
    def test_exact_stochastic_score_gradient_alpha1_and_var_counterexample(self):
        records = analytic_check()
        self.assertEqual(len(records), 9)
        self.assertLess(max(r["absolute_error"] for r in records), 1e-8)
        self.assertTrue(any(abs(r["score_gradient"]-r["incorrectly_tail_centered_gradient"]) > .01
                            for r in records if r["alpha"] < 1))

    def test_behavior_policy_clipped_surrogate_has_expected_horizon_and_rms_scale(self):
        import math
        theta0 = math.log(.4/.6)
        actions, rewards = [], []
        specifications = ((0, 0, .9, 1.), (0, 1, 1.1, 1.), (1, 0, .8, 1.),
                          (1, 1, 0., .1), (1, 1, 1.3, .9))
        for (a, b, r, _), (_, mass, _) in zip(specifications, outcomes(theta0)):
            actions.extend([[a, b]]*round(1000*mass))
            rewards.extend([r]*round(1000*mass))
        actions = torch.tensor(actions, dtype=torch.float64).T
        returns = torch.tensor(rewards, dtype=torch.float64)
        for alpha in (.2, 1.):
            theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
            logprob = torch.distributions.Bernoulli(logits=theta).log_prob(actions)
            ratio = (logprob-logprob.detach()).exp()
            weights, _ = score_weights(returns, alpha)
            rms = weights.square().mean().sqrt()
            normalized = weights/rms
            surrogate = torch.minimum(ratio*normalized, ratio.clamp(.8, 1.2)*normalized).mean()
            surrogate.backward()
            exact = (exact_cvar(theta0+1e-5, alpha)-exact_cvar(theta0-1e-5, alpha))/2e-5
            self.assertAlmostEqual(float(theta.grad), exact/(2*float(rms)), places=8)

    def test_discounted_complete_return_and_constant_episode_score(self):
        torch.testing.assert_close(episode_returns(torch.tensor([[1., 2.], [3., 4.]]), .5), torch.tensor([2.5, 4.]))
        rewards, dones, next_done, info = complete_batch()
        advantage, diagnostic = actor_score_advantages(rewards, dones, next_done, info, .9, "cvar")
        torch.testing.assert_close(advantage, advantage[:1].expand_as(advantage))
        self.assertFalse(advantage.requires_grad)
        self.assertEqual(diagnostic["training_failures"], 3)
        self.assertEqual(diagnostic["strict_tail_failures"], 3)
        self.assertGreater(diagnostic["failure_enrichment"], 1.)

    def test_partial_episode_rejected_and_success_labels_do_not_change_actor_weights(self):
        rewards, dones, next_done, info = complete_batch()
        expected, _ = actor_score_advantages(rewards, dones, next_done, info, .9, "cvar")
        info["final_info"]["episode"]["success_at_end"] = ~info["final_info"]["episode"]["success_at_end"]
        actual, _ = actor_score_advantages(rewards, dones, next_done, info, .9, "cvar")
        torch.testing.assert_close(expected, actual)
        dones[20, 4] = 1
        with self.assertRaises(ValueError):
            actor_score_advantages(rewards, dones, next_done, info, .9, "cvar")

    def test_ties_have_zero_score_and_zero_failure_denominator_is_null(self):
        returns = torch.ones(32)
        weight, _ = score_weights(returns, .1)
        self.assertEqual(int(torch.count_nonzero(weight)), 0)
        diagnostic = training_diagnostics(returns, torch.ones(32, dtype=torch.bool), .1)
        self.assertIsNone(diagnostic["failure_enrichment"])
        self.assertIsNone(diagnostic["failure_capture_fraction"])

    def test_exact_marked_patch_preserves_agent_critic_targets_audits_and_schedule(self):
        base = Path("research/discovery/upstream/precision-constant-base.py").read_bytes()
        self.assertEqual(hashlib.sha256(base).hexdigest(), BASE_SHA256)
        base_tree = ast.parse(base)
        original_agent = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
        for method in METHODS:
            payload = build_precision_trainer(base, method)
            self.assertEqual(strip_risk_additions(payload.decode()).encode(), base)
            parsed = ast.parse(payload)
            agent = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
            self.assertEqual(ast.dump(agent), ast.dump(original_agent))
            self.assertEqual(payload.count(b"returns = advantages + values"), 1)
            for tag in (b'"audit/optimizer_steps"', b'"audit/logstd_dim_', b'"audit/training_entropy_coef"'):
                self.assertEqual(payload.count(tag), base.count(tag))
            self.assertNotIn(b"from research.", payload)
            self.assertNotIn(b"from automanip.", payload)
        with self.assertRaises(ValueError):
            build_precision_trainer(base+b"\n", "risk-episode-cvar10")


if __name__ == "__main__":
    unittest.main()
