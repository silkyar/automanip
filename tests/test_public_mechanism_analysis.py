import unittest

import numpy as np

from scripts.analyze_public_mechanism_probes import paired_summary, validate


class PublicMechanismAnalysisTests(unittest.TestCase):
    def test_pairing_cancels_large_shared_episode_variation(self):
        rng = np.random.default_rng(7)
        baseline = rng.normal(0, 100, (3, 512))
        candidate = baseline + np.array([1., 2., 3.])[:, None]
        draws = rng.integers(0, 8, (3, 10000, 8))
        result = paired_summary(candidate - baseline, draws)
        self.assertAlmostEqual(result['mean'], 2.)
        np.testing.assert_allclose(result['paired_98_75'], [2., 2.], atol=1e-12)

    def test_resamples_whole_vector_batches(self):
        values = np.repeat(np.arange(8), 64)[None, :]
        # Selecting the same batch eight times must NOT select arbitrary lanes.
        draws = np.repeat(np.arange(8)[None, :, None], 8, axis=2)
        result = paired_summary(values, draws)
        self.assertAlmostEqual(result['paired_95'][0], .175)
        self.assertAlmostEqual(result['paired_95'][1], 6.825)

    def test_incomplete_execution_cannot_be_scientific_failure(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete execution'):
            validate({'state': 'failed'}, 12, 307200)

    def test_reject_nonfinite_or_wrong_episode_count(self):
        for values in [np.zeros((3, 511)), np.full((3, 512), np.nan)]:
            with self.assertRaises(ValueError):
                paired_summary(values, np.zeros((3, 2, 8), dtype=int))


if __name__ == '__main__':
    unittest.main()
