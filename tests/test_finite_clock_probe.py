import unittest
import numpy as np
import torch
from research.finite_clock_probe import horizon_mass,summarize_geometry
from research.crossfit_value_probe import crossfit_baselines,discounted_returns


class FiniteClockTests(unittest.TestCase):
    def test_clock_solves_constant_state_constant_reward_positive_control(self):
        rewards=torch.full((50,64),.5,dtype=torch.float64)
        returns=discounted_returns(rewards).flatten()
        state=torch.ones((3200,1),dtype=torch.float64)
        clock=state*torch.from_numpy(np.repeat(horizon_mass(50),64))[:,None]
        a=crossfit_baselines(state,returns,64)['state']
        b=crossfit_baselines(clock,returns,64)['state']
        self.assertGreater(float(((returns-a)**2).mean()),.1)
        self.assertLess(float(((returns-b)**2).mean()),float(((returns-a)**2).mean())*.001)
        np.testing.assert_allclose(horizon_mass(3),[.271,.19,.1])

    def test_terminal_reward_does_not_invent_terminal_geometry(self):
        obs=np.zeros((50,64,48),dtype=np.float32)
        obs[:,:,25:28]=[0,0,.02];obs[:,:,32:35]=[.1,0,.02]
        obs[:,:,39:42]=obs[:,:,25:28]-obs[:,:,18:21]
        obs[:,:,42:45]=obs[:,:,32:35]-obs[:,:,18:21]
        obs[:,:,45:48]=obs[:,:,32:35]-obs[:,:,25:28]
        rewards=np.zeros((50,64));rewards[-1]=1
        result=summarize_geometry(obs,rewards)
        self.assertEqual(result['ever_stock_success_reward'],1)
        self.assertEqual(result['ever_on_top_geometry_before_last_action'],0)

    def test_wrong_observation_layout_stops_diagnostic(self):
        obs=np.zeros((50,64,48));obs[:,:,39]=1
        with self.assertRaisesRegex(ValueError,'layout'):
            summarize_geometry(obs,np.zeros((50,64)))


if __name__=='__main__':unittest.main()
