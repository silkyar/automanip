import unittest
from types import SimpleNamespace
import numpy as np
from research.release_control_policy import ReleaseControlPolicy


class Base:
    schema=SimpleNamespace(observation_dim=48)
    layers=[(np.zeros((8,48)),np.full(8,.2))]
    def act(self,obs):return np.full((len(obs),8),.2).tolist()


def observations(eligible):
    x=np.zeros((2,48));x[:,7:9]=.015;x[:,25:28]=[.1 if eligible else 0,0,.06 if eligible else .02];x[:,32:35]=[.1,0,.02]
    return x


class ReleaseControlTests(unittest.TestCase):
    def test_one_five_step_window_and_arm_commands(self):
        policy=ReleaseControlPolicy(Base(),None,'targeted_open_zero_arm',2)
        for t in range(50):
            action=np.asarray(policy.act(observations(t>=17)))
            np.testing.assert_allclose(action[:,:7],0 if 17<=t<22 else .2)
            np.testing.assert_allclose(action[:,7],1 if 17<=t<22 else .2)
        np.testing.assert_array_equal(policy.trigger,[17,17]);np.testing.assert_array_equal(policy.applied,[5,5])

    def test_fallback_preserves_exact_command_budget(self):
        policy=ReleaseControlPolicy(Base(),None,'targeted_open',2)
        for t in range(50):policy.act(observations(False))
        np.testing.assert_array_equal(policy.trigger,[45,45]);np.testing.assert_array_equal(policy.applied,[5,5]);self.assertTrue(policy.fallback.all())

    def test_scheduled_control_ignores_eligibility(self):
        policy=ReleaseControlPolicy(Base(),None,'scheduled_open',2)
        for t in range(50):policy.act(observations(True))
        np.testing.assert_array_equal(policy.trigger,[25,25]);np.testing.assert_array_equal(policy.applied,[5,5])

    def test_common_noise_draws_continue_through_overrides(self):
        base=ReleaseControlPolicy(Base(),np.full(8,.1),'no_update',2)
        target=ReleaseControlPolicy(Base(),np.full(8,.1),'targeted_open',2)
        base.reset_noise(7);target.reset_noise(7)
        for t in range(50):
            a=np.asarray(base.act(observations(t>=17)));b=np.asarray(target.act(observations(t>=17)))
            np.testing.assert_array_equal(a[:,:7],b[:,:7])
            if not 17<=t<22:np.testing.assert_array_equal(a,b)

    def test_prefix_guard_rejects_unpaired_state(self):
        policy=ReleaseControlPolicy(Base(),None,'targeted_open',2)
        policy.reference=np.zeros((51,2,48))
        with self.assertRaisesRegex(ValueError,'paired'):policy.act(observations(False))


if __name__=='__main__':unittest.main()
