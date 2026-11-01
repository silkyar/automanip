"""State-relative proposals translated to the frozen absolute gripper command.

Pinned Panda gripper normalization maps a -> .015 + .025*a metres. The arm
coordinates are joint deltas; the gripper coordinate is an absolute target.
This module changes the sampler, not that controller or task definition.
"""
import numpy as np

from research.prefix_search import ANCHOR, choose_action


LOWER, UPPER = -.01, .04
MIDPOINT, HALF_RANGE = .015, .025
CONTROLLED_FINGER = 7


def inverse_target(position):
    position = np.asarray(position)
    if not np.isfinite(position).all():
        raise ValueError('Nonfinite joint position')
    return (position-MIDPOINT)/HALF_RANGE


def relative_action(mean, std, innovation, candidate, perturbation, step,
                    observations, anchor_finger, mode):
    if mode not in ('anchor', 'current') or type(candidate) is not int or not 1 <= candidate <= 8:
        raise ValueError('Closed gripper proposal condition required')
    action = choose_action(mean, std, innovation, candidate, perturbation, step)
    if step < ANCHOR:
        return action
    observations, anchor_finger = np.asarray(observations), np.asarray(anchor_finger)
    if observations.shape != (len(action), 48) or anchor_finger.shape != (len(action),):
        raise ValueError('Observation interface differs')
    position = anchor_finger if mode == 'anchor' else observations[:, CONTROLLED_FINGER]
    action[:, 7] = np.clip(inverse_target(position)+perturbation[:, 7], -1, 1)
    return action


def ideal_tracking(position, innovation, steps, mode):
    """Analytic actuator toy only: perfect tracking, no contact or dynamics."""
    q = np.asarray(position, dtype=np.float64).copy()
    initial = q.copy()
    for _ in range(steps):
        base = initial if mode == 'anchor' else q
        q = MIDPOINT+HALF_RANGE*np.clip(inverse_target(base)+innovation, -1, 1)
    return q
