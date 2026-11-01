import unittest
import numpy as np
from research.randomized_release_policy import RandomizedReleasePolicy
from tests.test_release_control_policy import Base,observations


class RandomizedReleaseTests(unittest.TestCase):
    def test_heterogeneous_lanes_have_fixed_budgets(self):
        assignment=np.tile(np.arange(4),16);p=RandomizedReleasePolicy(Base(),None,assignment)
        x=np.tile(observations(False)[:1],(64,1));rng=np.random.default_rng(37)
        starts=rng.integers(0,60,size=64)
        for t in range(50):
            eligible=t>=starts;x[:,25:28]=np.where(eligible[:,None],[.1,0,.06],[0,0,.02])
            actions=np.asarray(p.act(x));active=np.asarray(p.override_masks[-1])
            np.testing.assert_array_equal(actions[:,7],np.where(active,1.,.2))
            np.testing.assert_array_equal(actions[:,0],np.where(active&(assignment==3),0.,.2))
        np.testing.assert_array_equal(p.applied,np.where(assignment==0,0,5))
        np.testing.assert_array_equal(p.trigger[assignment==1],25)
        self.assertTrue(np.all(p.trigger[assignment>=2]<=45))

    def test_assignment_cannot_depend_on_eligibility(self):
        assignment=np.tile(np.arange(4),16);p=RandomizedReleasePolicy(Base(),None,assignment)
        assignment[:]=0
        np.testing.assert_array_equal(p.assignment,np.tile(np.arange(4),16))
        with self.assertRaises(ValueError):RandomizedReleasePolicy(Base(),None,assignment)


if __name__=='__main__':unittest.main()
