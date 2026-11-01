import unittest
import numpy as np
from research.persistent_switch_toy import rollout,wilson_lower


class PersistentSwitchTests(unittest.TestCase):
    def test_gate_latches_after_first_decision(self):
        calls=0
        def once(state):
            nonlocal calls
            calls+=1;return np.full(len(state),calls==11)
        # At step10the position is.1. Continuing afterward would lose success.
        self.assertTrue(rollout(np.array([1.1]),np.array([.1]),'learned',once)[0])
        self.assertEqual(calls,50)

    def test_always_stop_can_destroy_a_useful_base(self):
        x,v=np.array([1.]),np.array([.02])
        self.assertTrue(rollout(x,v,'base')[0])
        self.assertFalse(rollout(x,v,'learned',lambda s:np.ones(len(s),bool))[0])

    def test_calibration_uses_uncertainty(self):
        self.assertLess(wilson_lower(1,1),.8);self.assertGreater(wilson_lower(500,512),.8)


if __name__=='__main__':unittest.main()
