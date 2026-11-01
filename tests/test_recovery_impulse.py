import unittest
from types import SimpleNamespace
import torch
from research.recovery_impulse import apply_impulse


class ImpulseTests(unittest.TestCase):
    def test_pinned_velocity_data_api_order_and_momentum(self):
        events=[];old=torch.tensor([[.1,.2,.3],[.4,.5,.6]])
        def setter(value):events.append(('set',value))
        actor=SimpleNamespace(linear_velocity=old.clone(),mass=torch.tensor([.064,.032]),set_linear_velocity=setter,
            scene=SimpleNamespace(gpu_sim_enabled=True,px=SimpleNamespace(gpu_apply_rigid_dynamic_data=lambda:events.append(('apply',None))),_gpu_fetch_all=lambda:events.append(('fetch',None))))
        apply_impulse(actor,[[0.,.08,0.],[0.,-.08,0.]])
        self.assertEqual([x[0] for x in events],['set','apply','fetch'])
        torch.testing.assert_close((events[0][1]-old)*actor.mass[:,None],torch.tensor([[0.,.08,0.],[0.,-.08,0.]]))
        torch.testing.assert_close(actor.linear_velocity,old)
    def test_invalid_mass_rejected_before_mutation(self):
        actor=SimpleNamespace(linear_velocity=torch.zeros(2,3),mass=torch.zeros(2),set_linear_velocity=lambda v:self.fail('must not mutate'))
        with self.assertRaises(ValueError):apply_impulse(actor,torch.zeros(2,3))
