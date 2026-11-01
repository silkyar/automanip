import unittest

from research.capg_analysis import crossing, normalized_auc, summarize_pair


def curve(counts, steps=None):
    if steps is None:
        steps = [5_120_000 * (i + 1) for i in range(len(counts))]
    return [dict(steps=s, successes=n, episodes=128) for s, n in zip(steps, counts)]


class ScreenAnalysisTests(unittest.TestCase):
    def test_crossing_reports_nonmonotonicity_and_censoring_without_interpolation(self):
        result = crossing(curve([0, 116, 100]), .9)
        self.assertEqual(result['previous_sample_steps'], 5_120_000)
        self.assertEqual(result['first_passing_sample_steps'], 10_240_000)
        self.assertFalse(result['all_later_samples_pass'])
        self.assertFalse(result['continuous_crossing_measured'])
        self.assertFalse(crossing(curve([0, 100]), .9)['reached_at_sample'])

    def test_auc_has_no_assumed_zero_step_and_weights_unequal_intervals(self):
        self.assertAlmostEqual(normalized_auc(curve([0, 128, 128], [10, 20, 40])), 5 / 6)
        result = summarize_pair(curve([0, 64, 128]), curve([0, 80, 128]))
        self.assertEqual(result['auc_domain_steps'], [5_120_000, 15_360_000])
        self.assertAlmostEqual(result['candidate_auc_gain'], .0625)
        self.assertTrue(result['follow_up_training_seeds_warranted'])

    def test_final_guard_and_unmatched_budget_prevent_follow_up(self):
        self.assertFalse(summarize_pair(curve([0, 0, 128]), curve([0, 100, 124]))
                         ['follow_up_training_seeds_warranted'])
        result = summarize_pair(curve([0, 0, 128]), curve([0, 100, 128],
                                                           [5_120_000, 10_240_000, 14_000_000]))
        self.assertFalse(result['final_steps_match'])
        self.assertFalse(result['follow_up_training_seeds_warranted'])

    def test_invalid_counts_grids_and_insufficient_shared_evidence_rejected(self):
        for bad in (curve([True, 1]), curve([129, 1]), curve([0, 1], [2, 2])):
            with self.assertRaises(ValueError):
                normalized_auc(bad)
        with self.assertRaises(ValueError):
            summarize_pair(curve([0, 1], [1, 2]), curve([0, 1], [2, 3]))


if __name__ == '__main__':
    unittest.main()
