import unittest
import numpy as np
from research.maintenance_command_audit import clipped_moments


class ClippedNormalTests(unittest.TestCase):
    def test_unclipped_limit_and_symmetry(self):
        first,second=clipped_moments(np.array([0.,.1,-.1]),.02)
        np.testing.assert_allclose(first,[0.,.1,-.1],atol=1e-12)
        np.testing.assert_allclose(second,[.0004,.0104,.0104],atol=1e-12)

    def test_saturated_limits_and_numeric_integral(self):
        first,second=clipped_moments(np.array([-20.,20.,.3]),.7)
        np.testing.assert_allclose(first[:2],[-1,1],atol=1e-12);np.testing.assert_allclose(second[:2],1,atol=1e-12)
        nodes,weights=np.polynomial.hermite.hermgauss(200)
        action=np.clip(.3+.7*np.sqrt(2)*nodes,-1,1)
        self.assertAlmostEqual(first[2],float(weights@action/np.sqrt(np.pi)),delta=.002)
        self.assertAlmostEqual(second[2],float(weights@(action**2)/np.sqrt(np.pi)),delta=.003)


if __name__=='__main__':unittest.main()
