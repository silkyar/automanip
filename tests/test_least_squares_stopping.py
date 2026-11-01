import unittest
import numpy as np
from research.least_squares_stopping import fit_continuation,current_decision,evaluate_paths
from research.learned_waiting_toy import paths


class StoppingTests(unittest.TestCase):
    def test_current_decision_cannot_read_future(self):
        now=np.array([[1.,0],[1.,0]])
        decision=current_decision(now,np.full(2,.4),np.ones(2,bool),np.array([.5,0]))
        np.testing.assert_array_equal(decision,[False,False])

    def test_state_continuation_learns_known_optimal_rule(self):
        rng=np.random.default_rng(17);train=paths(rng,2048);test=paths(rng,8192)
        model=fit_continuation(*train);cash,times=evaluate_paths(*test,model)
        signal=test[0][0,:,1]
        self.assertTrue((times[signal==0]==0).all());self.assertTrue((times[signal==1]==1).all())
        self.assertAlmostEqual(cash.mean(),.57,delta=.015)

    def test_unsupported_exercise_is_not_taken(self):
        train=list(paths(np.random.default_rng(2),100));train[2][:]=False
        model=fit_continuation(*train);cash,times=evaluate_paths(*train,model)
        np.testing.assert_array_equal(times,-1);np.testing.assert_array_equal(cash,0)


if __name__=='__main__':unittest.main()
