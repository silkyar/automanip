import unittest
import numpy as np
from research.release_maintenance_policy import ReleaseMaintenancePolicy
from tests.test_release_control_policy import Base,observations


class ReleaseMaintenanceTests(unittest.TestCase):
    def test_factorial_groups_differ_only_after_common_opening(self):
        assignment=np.tile(np.arange(4),16);p=ReleaseMaintenancePolicy(Base(),None,assignment)
        for t in range(50):
            x=np.tile(observations(t>=17)[:1],(64,1));a=np.asarray(p.act(x))
            grip=np.full(64,.2);arm=np.full(64,.2)
            if 17<=t<22:grip[:]=1
            if t>=22:grip[assignment%2==1]=1;arm[assignment>=2]=0
            np.testing.assert_array_equal(a[:,7],grip);np.testing.assert_array_equal(a[:,0],arm)
        np.testing.assert_array_equal(p.applied,5)
        np.testing.assert_array_equal(p.maintenance_counts[:,0],np.where(assignment%2==1,28,0))
        np.testing.assert_array_equal(p.maintenance_counts[:,1],np.where(assignment>=2,28,0))

    def test_fallback_has_no_post_window_treatment(self):
        p=ReleaseMaintenancePolicy(Base(),np.full(8,.1),np.tile(np.arange(4),16))
        for t in range(50):p.act(np.tile(observations(False)[:1],(64,1)))
        np.testing.assert_array_equal(p.applied,5);np.testing.assert_array_equal(p.maintenance_counts,0);self.assertTrue(p.fallback.all())


if __name__=='__main__':unittest.main()
