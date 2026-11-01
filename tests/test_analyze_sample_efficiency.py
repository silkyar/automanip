import unittest

from scripts.analyze_sample_efficiency import auc, sustained_threshold


class SampleEfficiencyAnalysisTests(unittest.TestCase):
    def test_sustained_crossing_requires_following_checkpoint(self):
        points = [(0, 0), (1, 110), (2, 90), (3, 104), (4, 103)]
        self.assertEqual(sustained_threshold(points, 103), 3)
        self.assertIsNone(sustained_threshold(points[:4], 103))
        self.assertIsNone(sustained_threshold([(0, 0), (1, 128)], 103))

    def test_auc_is_mean_success_fraction_over_common_horizon(self):
        self.assertAlmostEqual(auc([(0, 0), (2, 128), (4, 128)], 4), .75)
        self.assertIsNone(auc([(0, 0), (2, 128)], 4))


if __name__ == '__main__':
    unittest.main()
