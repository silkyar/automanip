import unittest
import torch
from research.feature_value_baseline import FeatureValueBaseline


class FeatureValueTests(unittest.TestCase):
    def actor(self):return torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Tanh(),torch.nn.Linear(4,1))
    def fill(self,helper,actor):
        helper.begin()
        for t in range(helper.steps):
            helper.active_step=t;actor(torch.ones(helper.lanes,2))
            helper.active_step=None

    def test_clock_positive_control_and_no_random_draws(self):
        torch.manual_seed(7);actor=self.actor();before=torch.random.get_rng_state().clone()
        state=FeatureValueBaseline(actor,50,4,'state');clock=FeatureValueBaseline(actor,50,4,'clock')
        torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
        # Both hooks must be active together when filling identical features.
        state.begin();clock.begin()
        for t in range(50):
            state.active_step=clock.active_step=t;actor(torch.ones(4,2));state.active_step=clock.active_step=None
        returns=(5*(1-.9**torch.arange(50,0,-1))).repeat_interleave(4).reshape(50,4)
        _,a,_=state.fit(returns);_,b,_=clock.fit(returns)
        self.assertLess(b['crossfit_MSE'],a['crossfit_MSE']*.001)
        torch.testing.assert_close(torch.random.get_rng_state(),before,rtol=0,atol=0)
        state.close();clock.close()

    def test_own_episode_rewards_never_enter_its_baseline(self):
        torch.manual_seed(8);actor=self.actor();helper=FeatureValueBaseline(actor,3,4,'clock');self.fill(helper,actor)
        rewards=torch.arange(12,dtype=torch.float32).reshape(3,4)
        a,_,_=helper.fit(rewards)
        changed=rewards.clone();changed[:,::2]+=1000
        b,_,_=helper.fit(changed)
        torch.testing.assert_close(a[:,::2],b[:,::2],rtol=0,atol=0)
        helper.close()

    def test_update_forwards_do_not_overwrite_rollout_cache(self):
        actor=self.actor();helper=FeatureValueBaseline(actor,3,4,'state');self.fill(helper,actor)
        before=helper.cache.clone();actor(torch.randn(7,2))
        torch.testing.assert_close(helper.cache,before,rtol=0,atol=0);helper.close()

    def test_partial_capture_cannot_fit_value(self):
        actor=self.actor();helper=FeatureValueBaseline(actor,3,4,'clock');helper.begin()
        with self.assertRaisesRegex(ValueError,'Complete cached'):helper.fit(torch.ones(3,4))
        helper.close()


if __name__=='__main__':unittest.main()
