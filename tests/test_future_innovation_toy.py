import unittest
import numpy as np
from research.future_innovation_toy import future_sum,estimate,data


class FutureInnovationTests(unittest.TestCase):
    def test_future_sum_excludes_current_action(self):
        np.testing.assert_allclose(future_sum(np.array([[1.,2.,3.]]),.5),[[1.75,1.5,0]])

    def test_full_linear_control_has_exact_analytic_gradient(self):
        e=np.random.default_rng(4).standard_normal((200,5))
        state,reward=data(e,False);adv=reward[:,None]-state
        gradient=estimate(e,adv-future_sum(e,1)-e,np.array([5.,0.]))
        np.testing.assert_allclose(gradient,np.tile([5.,0.],(200,1)),atol=1e-12)

    def test_current_control_requires_correction(self):
        # Enumeration with E[epsilon²]=1 isolates the missing score correction.
        e=np.array([[-1.],[1.]])
        uncorrected=estimate(e,np.zeros_like(e))
        corrected=estimate(e,np.zeros_like(e),np.array([1.,0.]))
        self.assertEqual(uncorrected[:,0].mean(),0);self.assertEqual(corrected[:,0].mean(),1)


if __name__=='__main__':unittest.main()
