import unittest
import numpy as np

from research.correction_motion_information import extract, fit_map, quaternion_velocity, shuffle_motion


class MotionInformationTests(unittest.TestCase):
    def fixture(self):
        observations=np.zeros((51,64,48))
        observations[:,:,28]=1
        for t in range(51):observations[t,:,25]=t*.01
        linear=np.broadcast_to(np.arange(50)[:,None,None],(50,64,3)).copy().astype(float)
        angular=-linear
        decision=np.full(64,20)
        return observations,linear,angular,decision

    def test_oracle_is_before_intervention_and_history_has_no_future(self):
        observations,linear,angular,decision=self.fixture()
        result=extract(observations,linear,angular,decision)
        np.testing.assert_array_equal(result['oracle'][:,:3],19)
        np.testing.assert_allclose(result['history'][:,0],.2)
        observations[21:]=np.nan
        linear[20:]=np.nan
        angular[20:]=np.nan
        changed=extract(observations,linear,angular,decision)
        for k in result:np.testing.assert_array_equal(result[k],changed[k])

    def test_world_rotation_and_double_cover(self):
        previous=np.array([[1.,0,0,0]])
        current=np.array([[2**-.5,0,0,2**-.5]])
        np.testing.assert_allclose(quaternion_velocity(previous,current),[[0,0,10*np.pi]],atol=1e-12)
        np.testing.assert_allclose(quaternion_velocity(previous,-current),[[0,0,10*np.pi]],atol=1e-12)
        np.testing.assert_array_equal(previous,[[1.,0,0,0]])
        nonunit_previous, nonunit_current=previous*2,current*3
        saved_previous,saved_current=nonunit_previous.copy(),nonunit_current.copy()
        quaternion_velocity(nonunit_previous,nonunit_current)
        np.testing.assert_array_equal(nonunit_previous,saved_previous)
        np.testing.assert_array_equal(nonunit_current,saved_current)

    def test_scaler_does_not_use_test_statistics(self):
        train=np.array([[1.,2.],[2.,4.],[3.,6.]])
        _,first=fit_map(train,np.vstack([train,[0,0]]),1,8)
        _,second=fit_map(train,np.vstack([train,[1e9,-1e9]]),1,8)
        for k in first:np.testing.assert_array_equal(first[k],second[k])

    def test_shuffled_control_preserves_motion_multiset_per_time_and_batch(self):
        motion=np.arange(48).reshape(8,6)
        batch=np.repeat([0,1],4)
        decision=np.tile([10,10,15,15],2)
        shuffled=shuffle_motion(motion,batch,decision,97100000)
        for group in [0,1]:
            for time in [10,15]:
                mask=(batch==group)&(decision==time)
                np.testing.assert_array_equal(np.sort(motion[mask],axis=0),np.sort(shuffled[mask],axis=0))


if __name__=='__main__':unittest.main()
