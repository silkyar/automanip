import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from research.prefix_search import prefix_digest
from scripts.analyze_prefix_search import analyze


def fixture(root, bad_prefix=False, successes=True):
    for index in range(2):
        for candidate, repeat in [(c, 0) for c in range(16)]+[(0, 1), (1, 1)]:
            child = root/f'collection-{index}'/f'candidate-{candidate}-r{repeat}'
            child.mkdir(parents=True)
            trace = {k: np.zeros((51, 64, 3), dtype=np.float32) for k in
                     ('position', 'qpos', 'qvel', 'linear_velocity', 'angular_velocity')}
            trace['quaternion'] = np.tile(np.array([1., 0, 0, 0], dtype=np.float32), (51, 64, 2, 1))
            trace.update(observations=np.zeros((51, 64, 48), dtype=np.float32),
                         executed_actions=np.zeros((50, 64, 8), dtype=np.float32),
                         parent_means=np.zeros((50, 64, 8), dtype=np.float32),
                         flags=np.zeros((50, 64, 7), dtype=np.int8))
            if successes and candidate == 1: trace['flags'][-1, :32, 6] = 1
            if successes and candidate == 2: trace['flags'][-1, 32:, 6] = 1
            if bad_prefix and candidate == 3: trace['position'][1, 0, 0] = .005
            np.savez_compressed(child/'trace.npz', **trace)
            (child/'report.json').write_text(json.dumps(dict(state='complete', collection_index=index,
                candidate=candidate, repeat=repeat, transitions_completed=3200, state_write_calls=0,
                prefix_sha256=prefix_digest(trace), wall_seconds=1)))


PLAN = dict(parents=[2, 3], max_transitions=115200, tie_seed=11, random_control_seed=12)


class SearchAnalysisTests(unittest.TestCase):
    def test_oracle_is_not_a_policy_and_exports_only_successful_training_targets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root/'raw')
            r = analyze(root/'raw', PLAN, root/'dataset')
            self.assertTrue(r['fidelity_passed'])
            self.assertTrue(r['coverage_passed'])
            self.assertEqual(r['covered_source_states'], 128)
            self.assertEqual(r['pooled_oracle_minus_fixed_quiet'], .5)
            with np.load(root/'dataset/parent-2.npz', allow_pickle=False) as z:
                self.assertEqual(z['teacher_observations'].shape, (1280, 48))
                self.assertEqual(z['retain_observations'].shape, (64, 30, 48))
                self.assertTrue((z['selected_candidate'][:32] == 1).all())
                self.assertTrue((z['selected_candidate'][32:] == 2).all())

    def test_prefix_mismatch_blocks_export_even_with_perfect_label_coverage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root/'raw', bad_prefix=True)
            r = analyze(root/'raw', PLAN, root/'dataset')
            self.assertTrue(r['coverage_passed'])
            self.assertFalse(r['fidelity_passed'])
            self.assertFalse((root/'dataset').exists())

    def test_sparse_labels_block_export(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root/'raw', successes=False)
            r = analyze(root/'raw', PLAN, root/'dataset')
            self.assertTrue(r['fidelity_passed'])
            self.assertFalse(r['coverage_passed'])
            self.assertEqual(r['covered_source_states'], 0)
            self.assertFalse((root/'dataset').exists())


if __name__ == '__main__': unittest.main()
