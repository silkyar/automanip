import unittest
import numpy as np
from research.random_switch_collection import RandomSwitchPolicy
from tests.test_release_control_policy import Base


class RandomSwitchCollectionTests(unittest.TestCase):
    def test_decisions_are_time_randomized_and_have_exact_budgets(self):
        treatment=np.tile([0,1],32);decision=np.arange(64)%35+10
        policy=RandomSwitchPolicy(Base(),np.full(8,.1),treatment,decision,9)
        for t in range(50):
            actions=policy.act(np.zeros((64,48)));active=(treatment==1)&(t>=decision)
            np.testing.assert_array_equal(actions[active,:7],0);np.testing.assert_array_equal(actions[active,7],1)
            np.testing.assert_allclose(actions[~active],np.clip(policy.raw_proposal[~active],-1,1),atol=3e-8)
        np.testing.assert_array_equal(policy.applied,np.where(treatment==1,50-decision,0))

    def test_randomization_is_validated_before_collection(self):
        with self.assertRaises(ValueError):RandomSwitchPolicy(Base(),np.ones(8),np.zeros(64),np.full(64,10),1)
        with self.assertRaises(ValueError):RandomSwitchPolicy(Base(),np.ones(8),np.tile([0,1],32),np.full(64,9),1)


if __name__=='__main__':unittest.main()
