"""Scientific invariants and counterexamples for the candidate-side toy."""
import unittest

from research.discovery.credit_toy import gae_check, replay_chain, selection_bias


class CreditMechanismTests(unittest.TestCase):
    def test_reverse_order_propagates_reused_data_without_new_transitions(self):
        reverse = replay_chain(mode="reverse")
        forward = replay_chain(mode="forward")
        self.assertEqual(reverse["bellman_updates"], 50)
        self.assertEqual(forward["bellman_updates"], 2500)
        self.assertEqual(reverse["toy_environment_transitions"],
                         forward["toy_environment_transitions"])

    def test_frozen_targets_remove_order_advantage(self):
        counts = [replay_chain(mode=m, target="frozen")["bellman_updates"]
                  for m in ("forward", "shuffle", "reverse")]
        self.assertEqual(counts, [2500, 2500, 2500])

    def test_short_gae_trace_is_not_proof_of_expected_credit_loss(self):
        for lam in (0., .5, .9, 1.):
            result = gae_check(lam=lam)
            self.assertAlmostEqual(result["exact_critic_gap_fraction"], 1.)
            self.assertAlmostEqual(result["zero_critic_gap_fraction"], lam**49)

    def test_outcome_selection_does_not_estimate_unconditional_value(self):
        result = selection_bias()
        self.assertLess(abs(result["uniform_estimate_mean"]-.2), .02)
        self.assertEqual(result["success_only_estimate_mean"], 1.)


if __name__ == "__main__": unittest.main()
