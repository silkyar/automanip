import unittest
import numpy as np
from scripts.analyze_randomized_release import interval,analyze


class RandomizedAnalysisTests(unittest.TestCase):
    def test_block_contrast_respects_constant_effect(self):
        result=interval(np.full((2,32),.25))
        self.assertEqual(result['mean_difference'],.25);self.assertEqual(result['interval_98_75'],[.25,.25])

    def test_missing_or_invalid_blocks_rejected(self):
        with self.assertRaises(ValueError):interval(np.zeros((2,31)))
        with self.assertRaises(ValueError):analyze([{'state':'failed'}]*2)


if __name__=='__main__':unittest.main()
