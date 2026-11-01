import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from research.prefix_search import prefix_digest
from scripts.analyze_onpolicy_support import analyze


def fixture(root,sparse=False):
    for index in range(2):
        for kind in ('original','student'):
            for candidate,repeat in [(c,0) for c in range(4)]+[(1,1)]:
                child=root/f'collection-{index}'/f'{kind}-candidate-{candidate}-r{repeat}';child.mkdir(parents=True)
                t={k:np.zeros((51,64,3),dtype=np.float32) for k in ('position','qpos','qvel','linear_velocity','angular_velocity')}
                t.update(quaternion=np.tile(np.array([1.,0,0,0],dtype=np.float32),(51,64,2,1)),observations=np.zeros((51,64,48),dtype=np.float32),executed_actions=np.zeros((50,64,8),dtype=np.float32),parent_means=np.zeros((50,64,8),dtype=np.float32),flags=np.zeros((50,64,7),dtype=np.int8))
                if candidate==1:t['flags'][-1,:5 if sparse and kind=='student' else 48,6]=1
                np.savez_compressed(child/'trace.npz',**t)
                (child/'report.json').write_text(json.dumps(dict(state='complete',transitions_completed=3200,state_write_calls=0,policy_noise_enabled=False,source_kind=kind,candidate=candidate,repeat=repeat,collection_index=index,prefix_sha256=prefix_digest(t),wall_seconds=1)))


PLAN=dict(parents=[2,3],source_split_seed=96740000,tie_seed=96730000,max_transitions=64000)


class SupportAnalysisTests(unittest.TestCase):
    def test_exports_both_prespecified_banks_only_after_fidelity_and_enough_labels(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);fixture(root/'raw');r=analyze(root/'raw',PLAN,root/'data')
            self.assertTrue(r['screen_passed']);self.assertEqual(r['new_training_transitions'],64000)
            self.assertTrue((root/'data/student-2.npz').exists());self.assertTrue((root/'data/original-3.npz').exists())

    def test_insufficient_positive_rows_in_one_bank_block_the_comparison(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);fixture(root/'raw',sparse=True);r=analyze(root/'raw',PLAN,root/'data')
            self.assertTrue(r['fidelity_passed']);self.assertFalse(r['fitting_data_sufficient']);self.assertFalse((root/'data').exists())


if __name__=='__main__':unittest.main()
