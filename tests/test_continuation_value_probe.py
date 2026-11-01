import unittest
import numpy as np
from research.continuation_value_probe import exercise_features


class ExerciseFeatureTests(unittest.TestCase):
    def test_last_action_has_one_control_step_remaining(self):
        x=np.zeros((2,257));a=exercise_features(x,np.array([0,49]))
        np.testing.assert_allclose(a[:,-2:],[[1,1],[.02,.0004]])
        self.assertEqual(a.shape,(2,259))

    def test_terminal_or_missing_clock_is_rejected(self):
        with self.assertRaises(ValueError):exercise_features(np.zeros((2,257)),np.array([0,50]))
        with self.assertRaises(ValueError):exercise_features(np.zeros((2,257)),np.array([0]))


if __name__=='__main__':unittest.main()
