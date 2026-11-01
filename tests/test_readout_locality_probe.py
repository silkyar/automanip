import unittest
import numpy as np
import torch
from research.readout_locality_probe import recovered_shuffle


class ShuffleRecoveryTests(unittest.TestCase):
    def test_saved_state_recovers_historical_minibatch_without_guessing_data(self):
        for consumed in (0,5):
            rng=np.random.RandomState(19);rng.rand(consumed);state=rng.get_state()
            indices=np.arange(128);rng.shuffle(indices)
            saved=np.random.get_state()
            try:
                recovered,attempts=recovered_shuffle(state[1],indices[:16],128)
                np.random.set_state(recovered);actual=np.arange(128);np.random.shuffle(actual)
                self.assertTrue(np.array_equal(actual,indices));self.assertLessEqual(attempts,625)
            finally:np.random.set_state(saved)
