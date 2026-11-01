"""Deterministic source histories for one controlled data-aggregation test."""
import numpy as np

from research.relative_gripper_search import relative_action


def action(mean, std, perturbation, step, observations, anchor_finger, candidate):
    if type(candidate) is not int or not 0 <= candidate <= 3:
        raise ValueError('Closed support candidate index')
    if step < 30 or candidate == 0:
        return np.clip(mean, -1, 1).astype(np.float32)
    return relative_action(mean, std, np.zeros_like(mean), candidate, perturbation,
                           step, observations, anchor_finger, 'current')
