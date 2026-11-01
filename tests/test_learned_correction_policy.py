import unittest
from types import SimpleNamespace
import numpy as np
from research.learned_correction_policy import LearnedCorrectionPolicy


def base():
    return SimpleNamespace(layers=[(np.zeros((256,48)),np.zeros(256)),(np.zeros((8,256)),np.full(8,.2))])


def head(cutoff):
    coefficients=np.zeros((2,257));coefficients[1,-1]=.9
    return dict(coefficients=coefficients,support_metric=np.eye(257),support_cutoff=cutoff)


class LearnedCorrectionTests(unittest.TestCase):
    def test_support_abstains_where_raw_head_would_switch(self):
        assignment=np.tile(np.arange(4),16);p=LearnedCorrectionPolicy(base(),np.full(8,.1),assignment,head(.5),17)
        for t in range(50):
            a=np.asarray(p.act(np.zeros((64,48))))
            np.testing.assert_array_equal(a[assignment==2,:7],0);np.testing.assert_array_equal(a[assignment==2,7],1)
        np.testing.assert_array_equal(p.applied[assignment==2],50);np.testing.assert_array_equal(p.applied[assignment==3],0);np.testing.assert_array_equal(p.applied[assignment==0],0)
        np.testing.assert_array_equal(p.applied[assignment==1],50-p.random_time[assignment==1])

    def test_accepted_correction_latches_independent_of_later_scores(self):
        assignment=np.tile(np.arange(4),16);p=LearnedCorrectionPolicy(base(),np.full(8,.1),assignment,head(2),1)
        p.act(np.zeros((64,48)));p.coefficients[:]=0
        a=np.asarray(p.act(np.zeros((64,48))))
        np.testing.assert_array_equal(a[assignment==3,:7],0);np.testing.assert_array_equal(a[assignment==3,7],1)
        np.testing.assert_array_equal(p.trigger[assignment==3],0)

    def test_malformed_support_metadata_rejected(self):
        bad=head(2);bad['support_metric'][0,0]=-1
        with self.assertRaises(np.linalg.LinAlgError):LearnedCorrectionPolicy(base(),np.full(8,.1),np.tile(np.arange(4),16),bad,1)


if __name__=='__main__':unittest.main()
