import unittest
import numpy as np
from scripts.analyze_random_switch_training import partition,fit_head,interval


class RandomSwitchAnalysisTests(unittest.TestCase):
    def test_whole_batches_have_disjoint_fixed_partitions(self):
        groups={k:{b for b in range(32) if partition(b)==k} for k in ['fit','calibration','test']}
        self.assertEqual([len(groups[k]) for k in ['fit','calibration','test']],[16,8,8])
        self.assertFalse(groups['fit']&groups['test']);self.assertFalse(groups['fit']&groups['calibration'])

    def test_constant_binary_target_is_bounded_after_prediction(self):
        x=np.column_stack([np.linspace(-1,1,50),np.ones(50)])
        coefficient=fit_head(x,np.zeros(50));np.testing.assert_allclose(x@coefficient,0)
        self.assertEqual(interval(np.full(8,.1))['mean_gain'],.1)
        with self.assertRaises(ValueError):interval(np.zeros(7))


if __name__=='__main__':unittest.main()
