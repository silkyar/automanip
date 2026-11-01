from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/sample-efficiency-20261031'))
from symmetry import rotate_obs  # noqa: E402


def state(n, generator):
    obs = torch.randn(n, 48, generator=generator, dtype=torch.float64)*.3
    for i in (21, 28, 35):
        obs[:, i:i+4] /= obs[:, i:i+4].norm(dim=1, keepdim=True)
    tcp, a, b = obs[:, 18:21], obs[:, 25:28], obs[:, 32:35]
    obs[:, 39:42], obs[:, 42:45], obs[:, 45:48] = a-tcp, b-tcp, b-a
    return obs


class SymmetryTests(unittest.TestCase):
    def test_rotation_is_a_group_action_and_keeps_relative_vectors_consistent(self):
        generator = torch.Generator().manual_seed(0)
        obs = state(256, generator); center = torch.tensor([-.615, 0.], dtype=torch.float64)
        t1 = torch.rand(256, generator=generator, dtype=torch.float64)-.5
        t2 = torch.rand(256, generator=generator, dtype=torch.float64)-.5
        once = rotate_obs(obs, t1, center)
        torch.testing.assert_close(rotate_obs(once, t2, center), rotate_obs(obs, t1+t2, center))
        torch.testing.assert_close(rotate_obs(once, -t1, center), obs)
        tcp, a, b = once[:, 18:21], once[:, 25:28], once[:, 32:35]
        torch.testing.assert_close(once[:, 39:42], a-tcp); torch.testing.assert_close(once[:, 45:48], b-a)
        torch.testing.assert_close(once[:, 9:18], obs[:, 9:18]); torch.testing.assert_close(once[:, 1:9], obs[:, 1:9])
        for i in (21, 28, 35):
            torch.testing.assert_close(once[:, i:i+4].norm(dim=1), torch.ones(256, dtype=torch.float64))
        # base distance and heights are invariant
        torch.testing.assert_close((once[:, 25:27]-center).norm(dim=1), (obs[:, 25:27]-center).norm(dim=1))
        torch.testing.assert_close(once[:, 27], obs[:, 27])


if __name__ == '__main__':
    unittest.main()
