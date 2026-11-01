import unittest
import numpy as np
from scripts.analyze_learned_correction import interval,analyze


class LearnedCorrectionAnalysisTests(unittest.TestCase):
    def test_fixedblockcontrast_is_preserved(self):
        r=interval(np.full((2,8),.25));self.assertEqual(r['mean_gain'],.25);self.assertEqual(r['interval_98_75'],[.25,.25])

    def test_incomplete_work_not_scored_as_failure(self):
        with self.assertRaises(ValueError):analyze([{'state':'failed'}]*2)
        with self.assertRaises(ValueError):interval(np.zeros((2,7)))


if __name__=='__main__':unittest.main()
