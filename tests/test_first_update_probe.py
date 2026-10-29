import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from research.discovery import first_update_probe as probe

SOURCE=Path('research/discovery/bc-residual-exploration-v2/assist-bc-ppo-residual-std.py')


class FirstUpdateProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_production_reconstruction_and_both_bootstrap_modes(self):
        result=probe.cpu_preflight(SOURCE)
        self.assertEqual(result['state'],'passed')
        self.assertEqual(result['new_simulator_transitions'],0)
        self.assertEqual(result['optimizer_calls'],4)
        for row in result['checks']:
            self.assertTrue(row['bitwise_adam_state'])
            self.assertEqual(row['rng_devices_checked'],['numpy','torch_cpu'])

    def test_reconstruction_mismatch_is_a_hard_failure(self):
        original=probe.reconstructed_step
        def wrong_step(agent,*a,**kw):
            result=original(agent,*a,**kw)
            with torch.no_grad():next(agent.actor_mean.parameters()).add_(.01)
            return result
        with patch.object(probe,'reconstructed_step',wrong_step):
            with self.assertRaisesRegex(AssertionError,'Production parity mismatch'):
                probe.cpu_preflight(SOURCE)

    def test_source_drift_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed=Path(temporary)/'changed.py';changed.write_bytes(SOURCE.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Production source digest'):
                probe.load_production(changed)

    def test_disjoint_complete_episodes_and_no_sufficient_holdout_fails(self):
        args=probe.ProbeArgs();rows=torch.arange(51200).reshape(-1,1).repeat(1,48)
        mb=np.arange(1600)*2 %51200
        states,episodes=probe.kl_holdout({'b_obs':rows},mb,args)
        self.assertEqual(len(states),3200)
        self.assertFalse(set(episodes)&set(mb%1024))
        with self.assertRaisesRegex(RuntimeError,'Fewer than64'):
            probe.kl_holdout({'b_obs':rows},np.arange(1600),args)


if __name__=='__main__':unittest.main()
