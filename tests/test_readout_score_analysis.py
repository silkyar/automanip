import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_readout_score import analyze, contrast


def fixture(root, retain=True):
    for index in range(2):
        for method, successes in [('base', 0), ('search', 32), ('unbranched', 16)]:
            path = root/f'collection-{index}'/method; path.mkdir(parents=True)
            episodes = [dict(outcome=dict(terminal_success=lane < successes, failed=False,
                                          terminated=True, truncated=True)) for _ in range(8) for lane in range(64)]
            r = dict(state='complete', collection_index=index, method=method, parent=index+2,
                     environment_completed=25600, episode_count=512, successes=successes*8,
                     episodes=episodes, batches=[dict(successes=successes) for _ in range(8)],
                     ever_grasp=512 if retain or method != 'search' else 400,
                     weights_sha256='fixture', wall_seconds=1)
            (path/'report.json').write_text(json.dumps(r))


class ReadoutScoreAnalysisTests(unittest.TestCase):
    def test_learning_and_search_contribution_are_separate_registered_tests(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            r = analyze(root)
            self.assertEqual(r['pooled_success']['search'], .5)
            self.assertEqual(r['new_public_transitions'], 153600)
            self.assertTrue(r['learning_screen_passed'])
            self.assertTrue(r['search_specific_screen_passed'])
            self.assertEqual(contrast([[1]*8]*2, [[1]*8]*2)['interval_97_5'], [0., 0.])

    def test_success_without_grasp_retention_is_not_admitted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root, retain=False)
            r = analyze(root)
            self.assertFalse(r['search_grasp_retention_passed'])
            self.assertFalse(r['learning_screen_passed'])


if __name__ == '__main__': unittest.main()
