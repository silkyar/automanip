import copy
import unittest
import torch
from research.discovery.amp_v1 import StudentActor
from research.readout_reuse import configure_actor,residual_equivalence


class ReadoutReuseTests(unittest.TestCase):
    def test_zero_correction_exact_fusion_and_frozen_hidden_updates(self):
        torch.manual_seed(7);source=StudentActor((16,16));actor=copy.deepcopy(source)
        info=configure_actor(actor,'readout');self.assertEqual(info['trainable_actor_parameters'],16*8+8)
        obs=torch.randn(32,48);self.assertTrue(torch.equal(actor(obs),source(obs)))
        optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
        optimizer.zero_grad();loss=(actor(obs)-.4).square().mean();loss.backward();optimizer.step()
        errors=residual_equivalence(source,actor,obs)
        self.assertLess(errors['max_unclipped_difference'],1e-6)
        self.assertFalse(info['retention_guaranteed'])
        self.assertGreater(float((source(obs)-actor(obs)).abs().max().detach()),.001)
        # Retaining hidden tensors does not retain the old action mapping.
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(list(source.net.parameters())[:-2],list(actor.net.parameters())[:-2])))
    def test_hidden_mutation_and_unknown_modes_rejected(self):
        source=StudentActor((8,));actor=copy.deepcopy(source)
        with torch.no_grad():actor.net[0].weight.add_(.01)
        with self.assertRaises(ValueError):residual_equivalence(source,actor,torch.zeros(2,48))
        with self.assertRaises(ValueError):configure_actor(actor,'automatic')
