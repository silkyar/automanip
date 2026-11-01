import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analyze_branching_noop import analyze


def fixture(root, prefix_error=False):
    for condition in ('no_write', 'physical', 'augmented'):
        for repeat in range(2):
            directory = root/(condition+'-'+str(repeat)); directory.mkdir()
            trace = {key: np.zeros((51, 2, 3)) for key in
                     ('position', 'qpos', 'qvel', 'linear_velocity', 'angular_velocity')}
            trace['quaternion'] = np.tile([1., 0, 0, 0], (51, 2, 3, 1))
            trace['flags'] = np.zeros((50, 2, 7), dtype=np.int8)
            if condition != 'no_write':
                trace['qvel'][16:, 0, 0] = .02
                if prefix_error: trace['position'][8, 0, 0] = .01
            np.savez_compressed(directory/'trace.npz', **trace)
            (directory/'report.json').write_text(json.dumps(dict(state='complete', condition=condition,
                repeat=repeat, transitions_completed=3200, wall_seconds=1)))


class NoopAnalysisTests(unittest.TestCase):
    def test_detects_write_effect_only_after_matching_prefix_and_controls(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            result = analyze(root)
            self.assertTrue(result['no_write_controls']['passed'])
            self.assertTrue(result['prefixes_match'])
            self.assertEqual(result['new_transitions'], 19200)
            self.assertTrue(all(not r['first_action_after_write']['passed'] for r in result['comparisons']))
            self.assertTrue(result['conclusion'].startswith('No-op state writing perturbed'))

    def test_different_prefix_prevents_causal_attribution(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, prefix_error=True)
            result = analyze(root)
            self.assertFalse(result['prefixes_match'])
            self.assertTrue(result['conclusion'].startswith('Inconclusive'))


if __name__ == '__main__': unittest.main()
