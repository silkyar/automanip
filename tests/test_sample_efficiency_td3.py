from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/sample-efficiency-20261031'))
try:
    import tyro  # noqa: F401
except ImportError:
    import types; sys.modules['tyro'] = types.ModuleType('tyro')
import td3  # noqa: E402
from export_helper import export_actor  # noqa: E402
from automanip.numpy_inference import NumpyPolicy  # noqa: E402
from automanip.policy_io import PolicySchema, load_policy  # noqa: E402
from automanip.stackcube import STACK_TASK  # noqa: E402


class TD3Tests(unittest.TestCase):
    def test_n_step_sum_wraps_ring_and_respects_window(self):
        rew = torch.arange(40, dtype=torch.float32).reshape(10, 4)  # rew[t, e] = 4t + e
        t = torch.tensor([0, 8, 9, 3]); e = torch.tensor([1, 2, 3, 0]); m = torch.tensor([3, 3, 1, 2])
        got = td3.n_step_targets(rew, t, e, m, .5)
        expected = [1 + .5*5 + .25*9, 34 + .5*38 + .25*2, 39, 12 + .5*16]
        torch.testing.assert_close(got, torch.tensor(expected))

    def test_actor_exports_as_restricted_policy_matching_mean(self):
        torch.manual_seed(0)
        actor = td3.Actor()
        with torch.no_grad():
            actor.net[-1].weight.normal_(0, .05)
        x = torch.randn(256, 48)*.3
        with TemporaryDirectory() as d:
            art = export_actor(actor, Path(d)/'a')
            schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
            got = np.asarray(NumpyPolicy(load_policy(Path(art['path']), schema), schema).act(x.numpy()))
        want = actor(x).detach().numpy()
        inside = np.abs(want) < 1
        np.testing.assert_allclose(got[inside], want[inside], atol=2e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
