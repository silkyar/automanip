import math
import unittest

import numpy as np

from research.prefix_search import choose_action, proposal
from research.relative_gripper_search import HALF_RANGE, MIDPOINT, ideal_tracking, inverse_target, relative_action


class RelativeGripperTests(unittest.TestCase):
    def test_inverse_mapping_and_zero_innovation_hold_current_position(self):
        q = np.array([.001, .018, .02, .04])
        np.testing.assert_allclose(MIDPOINT+HALF_RANGE*inverse_target(q), q, atol=1e-17)
        for mode in ('anchor', 'current'):
            np.testing.assert_allclose(ideal_tracking(q, np.zeros(4), 20, mode), q, atol=1e-17)

    def test_identical_arm_commands_and_prefix_only_gripper_coordinate_changes(self):
        mean = np.zeros((64, 8)); std = np.ones(8)*.2; noise = np.ones((64, 8))
        obs = np.zeros((64, 48)); obs[:, 7] = .018
        perturbation = proposal(1, 12)
        absolute = choose_action(mean, std, noise, 1, perturbation, 30)
        anchor = relative_action(mean, std, noise, 1, perturbation, 30, obs, obs[:, 7], 'anchor')
        current = relative_action(mean, std, noise, 1, perturbation, 30, obs, obs[:, 7], 'current')
        np.testing.assert_array_equal(absolute[:, :7], anchor[:, :7])
        np.testing.assert_array_equal(anchor, current)
        np.testing.assert_array_equal(relative_action(mean, std, noise, 1, perturbation, 29, obs, obs[:, 7], 'anchor'),
                                      choose_action(mean, std, noise, 1, perturbation, 29))
        np.testing.assert_allclose((MIDPOINT+HALF_RANGE*anchor[:, 7])-.018,
                                   HALF_RANGE*perturbation[:, 7], atol=2e-9)

    def test_actuator_toy_distinguishes_bias_from_accumulation(self):
        q = np.array([.018, .018]); innovation = np.array([.05, -.05])
        np.testing.assert_allclose(ideal_tracking(q, innovation, 20, 'anchor'), [.01925, .01675])
        np.testing.assert_allclose(ideal_tracking(q, innovation, 20, 'current'), [.04, -.007])
        # At q=.020, old absolute N(0,.05²) opens beyond current only ~3e-5
        # of the time; relative innovations are symmetric, before saturation.
        probability = .5*math.erfc(float(inverse_target(.02))/.05/math.sqrt(2))
        self.assertLess(probability, .00004)
        self.assertGreater(probability, .00003)

    def test_relative_targets_stay_in_frozen_action_bounds(self):
        obs = np.zeros((64, 48)); obs[:, 7] = .04
        a = relative_action(np.zeros((64, 8)), np.ones(8), np.zeros((64, 8)),
                            1, np.ones((64, 8)), 30, obs, obs[:, 7], 'current')
        self.assertTrue((a <= 1).all() and (a >= -1).all())


if __name__ == '__main__': unittest.main()
