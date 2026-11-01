import unittest
from research.old_state_direction_probe import RewardCapture


class RewardCaptureTests(unittest.TestCase):
    def test_observation_does_not_repeat_physics_or_change_reward(self):
        class Base:
            calls=0;control_freq=20
            def step(self,actions):self.calls+=1;return 'obs',7.,False,False,{'value':1}
        base=Base();events=[];env=RewardCapture(base,lambda event,reward:events.append((event,reward)))
        result=env.step([1.])
        self.assertEqual(base.calls,1);self.assertEqual(result,('obs',7.,False,False,{'value':1}))
        self.assertEqual(env.control_freq,20);self.assertEqual(events,[('reserved',None),('completed',7.)])
