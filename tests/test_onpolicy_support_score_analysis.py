import unittest

from scripts.analyze_onpolicy_support_score import contrast, decision


class SupportScoreTests(unittest.TestCase):
    def test_pairing_and_fixed_effect_sizes(self):
        small = contrast([[2]*4]*2, [[1]*4]*2)
        large = contrast([[16]*4]*2, [[0]*4]*2)
        self.assertGreater(small['interval_97_5'][0], 0)
        self.assertFalse(decision(small, large, True))
        self.assertFalse(decision(large, small, True))
        self.assertFalse(decision(large, large, False))
        self.assertTrue(decision(large, large, True))

    def test_pairing_requires_registered_layout(self):
        with self.assertRaises(ValueError):
            contrast([[1]*8]*2, [[0]*8]*2)


if __name__ == '__main__':
    unittest.main()
