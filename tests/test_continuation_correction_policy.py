import unittest
import numpy as np
from research.continuation_correction_policy import ContinuationCorrectionPolicy
from tests.test_learned_correction_policy import base


def head():
    exercise=np.zeros(259);exercise[256]=.4
    time=np.full((50,1),.6);time[20:]=.2
    state=np.zeros((50,17));state[:,-1]=.6;state[10:,-1]=.2
    return dict(exercise_coefficient=exercise,support_metric=np.eye(259),support_cutoff=10,projection=np.zeros((256,16)),time_continuation=time,state_continuation=state,fixed_time=25)


class ContinuationPolicyTests(unittest.TestCase):
    def test_decision_uses_current_continuation_and_latches(self):
        assignment=np.tile(np.arange(4),16);p=ContinuationCorrectionPolicy(base(),np.full(8,.1),assignment,head(),14)
        for t in range(50):p.act(np.zeros((64,48)))
        np.testing.assert_array_equal(p.trigger[assignment==1],-1)
        np.testing.assert_array_equal(p.trigger[assignment==2],20)
        np.testing.assert_array_equal(p.trigger[assignment==3],10)
        np.testing.assert_array_equal(p.applied[assignment==3],40)
        np.testing.assert_array_equal(p.applied[assignment==0],25)

    def test_unsupported_exercise_keeps_base_behavior(self):
        h=head();h['support_cutoff']=.1
        p=ContinuationCorrectionPolicy(base(),np.full(8,.1),np.tile(np.arange(4),16),h,1)
        for _ in range(50):p.act(np.zeros((64,48)))
        self.assertTrue((p.trigger[p.assignment!=0]==-1).all())


if __name__=='__main__':unittest.main()
