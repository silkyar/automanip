import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from research.prefix_search import prefix_digest
from scripts.analyze_relative_gripper import analyze
from tests.test_prefix_search_analysis import fixture as absolute_fixture


def relative_fixture(root, absolute, arm_mismatch=False):
    for index in range(2):
        for mode in ('anchor', 'current'):
            for candidate, repeat in [(c, 0) for c in range(1, 9)]+[(1, 1)]:
                source = absolute/f'collection-{index}'/f'candidate-{candidate}-r0/trace.npz'
                with np.load(source, allow_pickle=False) as z: trace = {k: z[k] for k in z.files}
                trace['executed_actions'][30:, :, 7] = .3
                if mode == 'current':
                    trace['executed_actions'][31:, :, 7] = .5
                    if candidate == 1: trace['flags'][-1, :32, 6] = 1
                if arm_mismatch and mode == 'current' and candidate == 3:
                    trace['executed_actions'][35, 0, 0] = .1
                child = root/f'collection-{index}'/f'{mode}-candidate-{candidate}-r{repeat}'
                child.mkdir(parents=True)
                np.savez_compressed(child/'trace.npz', **trace)
                (child/'report.json').write_text(json.dumps(dict(state='complete', transitions_completed=3200,
                    state_write_calls=0, collection_index=index, candidate=candidate, repeat=repeat,
                    mode=mode, wall_seconds=1, prefix_sha256=prefix_digest(trace))))


PLAN = dict(parents=[2, 3], tie_seed=12, max_transitions=115200)


class RelativeAnalysisTests(unittest.TestCase):
    def test_exports_only_prespecified_current_mode_after_fidelity_and_coverage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); absolute_fixture(root/'absolute', successes=False)
            relative_fixture(root/'relative', root/'absolute')
            r = analyze(root/'relative', root/'absolute', PLAN, root/'dataset')
            self.assertTrue(r['screen_passed'])
            self.assertEqual(r['covered_source_states'], dict(absolute=0, anchor=0, current=64))
            with np.load(root/'dataset/parent-2.npz', allow_pickle=False) as z:
                self.assertEqual(z['teacher_observations'].shape, (640, 48))
                self.assertTrue((z['teacher_actions'][:, 7] >= .3).all())

    def test_changed_arm_commands_block_causal_admission(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); absolute_fixture(root/'absolute', successes=False)
            relative_fixture(root/'relative', root/'absolute', arm_mismatch=True)
            r = analyze(root/'relative', root/'absolute', PLAN, root/'dataset')
            self.assertTrue(r['coverage_passed'])
            self.assertFalse(r['fidelity_passed'])
            self.assertFalse((root/'dataset').exists())


if __name__ == '__main__': unittest.main()
