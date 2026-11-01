import unittest
import numpy as np
from scripts.analyze_release_control import paired_interval,analyze


class ReleaseAnalysisTests(unittest.TestCase):
    def test_pairing_preserves_constant_effect(self):
        a=paired_interval(np.full((2,512),.25))
        self.assertEqual(a['interval_98_75'],[.25,.25]);self.assertEqual(a['mean_difference'],.25)

    def test_batch_level_dependence_is_preserved(self):
        d=np.zeros((2,8,64));d[:,:4]=1
        a=paired_interval(d.reshape(2,512))
        self.assertEqual(a['mean_difference'],.5)
        self.assertLess(a['interval_98_75'][0],.3);self.assertGreater(a['interval_98_75'][1],.7)

    def test_incomplete_data_fails_closed(self):
        with self.assertRaises(ValueError):paired_interval(np.zeros((2,511)))
        with self.assertRaises(ValueError):analyze([{'state':'failed'},{'state':'complete'}])


if __name__=='__main__':unittest.main()
