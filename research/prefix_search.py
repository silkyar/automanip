"""Closed action proposals for a training-only prefix-replay feasibility test.

Quiet actions are an explicit, generic action-space prior, not a supplied release
option. All eight normalized action coordinates use the same distribution.
No geometry, stage label, reward, or outcome enters proposal generation.
"""
import hashlib

import numpy as np


CANDIDATES = 16
ANCHOR = 30
QUIET_SCALE = .05
OFFSET_SCALE = .25


def candidate_kind(candidate):
    if type(candidate) is not int or not 0 <= candidate < CANDIDATES:
        raise ValueError('Closed candidate index required')
    return 'base' if candidate == 0 else 'quiet' if candidate <= 8 else 'policy_offset'


def proposal(candidate, seed, lanes=64):
    kind = candidate_kind(candidate)
    if kind == 'base':
        return np.zeros((lanes, 8), dtype=np.float64)
    scale = QUIET_SCALE if kind == 'quiet' else OFFSET_SCALE
    return np.random.default_rng(seed+candidate).standard_normal((lanes, 8))*scale


def choose_action(mean, std, innovation, candidate, perturbation, step):
    mean, std, innovation, perturbation = map(np.asarray, (mean, std, innovation, perturbation))
    if mean.ndim != 2 or mean.shape[1] != 8 or innovation.shape != mean.shape or perturbation.shape != mean.shape or std.shape != (8,):
        raise ValueError('Invalid action inputs')
    if not all(np.isfinite(x).all() for x in (mean, std, innovation, perturbation)) or (std <= 0).any():
        raise ValueError('Invalid action numbers')
    kind = candidate_kind(candidate)
    if step < ANCHOR or kind == 'base':
        action = mean+std*innovation
    elif kind == 'quiet':
        action = perturbation
    else:
        action = mean+std*innovation+perturbation
    return np.clip(action, -1, 1).astype(np.float32)


def raw_mean(policy, observations):
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
        raise ValueError('Invalid policy observations')
    for i, (weight, bias) in enumerate(policy.layers):
        x = x @ weight.T+bias
        if i < len(policy.layers)-1:
            x = np.tanh(x)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite actor mean')
    return x


def prefix_digest(trace):
    h = hashlib.sha256()
    for key in sorted(trace):
        if key == 'parent_means':
            continue
        stop = ANCHOR if key in ('flags', 'executed_actions') else ANCHOR+1
        value = np.ascontiguousarray(trace[key][:stop])
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()
