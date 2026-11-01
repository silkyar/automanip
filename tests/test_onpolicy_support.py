import unittest

import numpy as np

from research.onpolicy_support import action


class SupportActionTests(unittest.TestCase):
    def test_all_candidate_prefixes_match_deterministic_source(self):
        mean=np.full((64,8),.4);std=np.ones(8)*.5;z=np.full((64,8),.05)
        obs=np.zeros((64,48));obs[:,7]=.02
        for candidate in range(4):
            np.testing.assert_array_equal(action(mean,std,z,29,obs,obs[:,7],candidate),mean.astype(np.float32))
        np.testing.assert_array_equal(action(mean,std,z,40,obs,obs[:,7],0),mean.astype(np.float32))

    def test_proposal_arm_commands_match_across_sources_but_gripper_tracks_own_state(self):
        mean=np.zeros((64,8));std=np.ones(8)*.2;z=np.full((64,8),.05)
        a=np.zeros((64,48));a[:,7]=.02;b=a.copy();b[:,7]=.03
        first=action(mean,std,z,35,a,a[:,7],1);second=action(mean,std,z,35,b,b[:,7],1)
        np.testing.assert_array_equal(first[:,:7],second[:,:7])
        np.testing.assert_allclose(second[:,7]-first[:,7],.4,atol=1e-6)


if __name__=='__main__':unittest.main()
