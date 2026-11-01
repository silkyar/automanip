import unittest
import numpy as np
from research.invariant_support_toy import support_gate,stateless_rollout
from research.persistent_switch_toy import rollout


class InvariantSupportTests(unittest.TestCase):
    def test_failed_or_untreated_examples_do_not_expand_support(self):
        state=np.array([[-.1,1],[.1,1],[2,1],[-2,1]])
        gate,region=support_gate(state,np.array([1,1,1,0]),np.array([True,True,False,True]))
        self.assertEqual(region['low'],-.1);self.assertEqual(region['high'],.1)
        np.testing.assert_array_equal(gate(np.array([[0,0],[.11,0]])),[True,False])

    def test_stateless_hold_matches_latched_execution(self):
        gate=lambda s:(s[:,0]>=-.1)&(s[:,0]<=.1)
        x=np.linspace(.4,2,200);v=np.full(200,.03)
        np.testing.assert_array_equal(stateless_rollout(x,v,gate),rollout(x,v,'learned',gate))


if __name__=='__main__':unittest.main()
