import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional existing torch environment required")
class NeuralCreditTests(unittest.TestCase):
    def test_recursive_shifted_target_matches_independent_forward_expansion(self):
        from research.discovery.neural_credit_toy import HORIZON, GAMMA, targets
        torch.manual_seed(8)
        values = torch.randn(3,HORIZON+1)
        values[:,-1] = 0
        rewards = torch.randn(3,HORIZON)
        rhos = .25 + 1.5*torch.rand(3,HORIZON)
        expected = values[:,:-1].clone()
        for t in range(HORIZON):
            for j in range(t,HORIZON):
                coefficient = GAMMA**(j-t)
                for i in range(t,j): coefficient = coefficient*rhos[:,i].clamp(max=1)
                delta = rhos[:,j]*(rewards[:,j]+GAMMA*values[:,j+1]-values[:,j])
                expected[:,t] += coefficient*delta
        self.assertTrue(torch.allclose(expected,targets(values,rewards,rhos,"recursive_vtrace"),atol=2e-6))

    def test_recursive_target_equals_full_return_on_policy(self):
        from research.discovery.neural_credit_toy import HORIZON, targets
        torch.manual_seed(1)
        values = torch.randn(5, HORIZON+1)
        values[:, -1] = 0
        rewards = torch.randn(5, HORIZON)
        ratio = torch.ones_like(rewards)
        self.assertTrue(torch.allclose(targets(values,rewards,ratio,"n_step_full"),
                                        targets(values,rewards,ratio,"recursive_vtrace"), atol=2e-6))

    def test_terminal_reward_does_not_cross_episode_boundaries(self):
        from research.discovery.neural_credit_toy import HORIZON, targets
        rewards = torch.zeros(2, HORIZON)
        rewards[0,-1] = 1
        result = targets(torch.zeros(2,HORIZON+1),rewards,torch.ones_like(rewards),"recursive_vtrace")
        self.assertEqual(float(result[1].abs().sum()), 0)
        self.assertGreater(float(result[0,0]), 0)

    def test_oracle_bellman_identity_in_shifted_policy(self):
        from research.discovery.neural_credit_toy import HORIZON, GAMMA, oracle_values
        # Exact quadrature for degree-two Bellman integrand over action and noise.
        k = .75
        value = oracle_values(torch.tensor([[0., .1]]), k)
        actions = torch.tensor([-1/(3**.5), 1/(3**.5)])
        noises = torch.tensor([-.03, .03])
        next_x = .09 + .2*actions[:,None] + noises[None,:]
        next_input = torch.stack((torch.full_like(next_x,1/HORIZON),next_x),-1)
        expected = ((1+k*actions[:,None])*oracle_values(next_input,k)).mean()*GAMMA
        self.assertAlmostEqual(float(value),float(expected),places=6)

    def test_data_includes_failures_and_no_outcome_filter(self):
        from research.discovery.neural_credit_toy import HORIZON, make_data
        inputs, actions, rewards, success = make_data(0)
        self.assertEqual(actions.numel(),128*HORIZON)
        self.assertTrue(bool(success.any()))
        self.assertTrue(bool((~success).any()))
        self.assertEqual(len(inputs), len(success))


if __name__ == "__main__": unittest.main()
