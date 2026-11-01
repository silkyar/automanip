import ast
from pathlib import Path
import unittest

import numpy as np

from research.prefix_search import choose_action, prefix_digest, proposal, raw_mean


class PrefixSearchTests(unittest.TestCase):
    def test_proposals_share_prefix_and_quiet_suffix_is_persistent(self):
        mean = np.full((64, 8), .2); std = np.full(8, .1); noise = np.ones((64, 8))
        p = proposal(1, 4)
        np.testing.assert_array_equal(choose_action(mean, std, noise, 0, p, 29),
                                      choose_action(mean, std, noise, 1, p, 29))
        np.testing.assert_array_equal(choose_action(mean, std, noise, 1, p, 30), p.astype(np.float32))
        np.testing.assert_array_equal(choose_action(mean, std, -noise, 1, p, 49), p.astype(np.float32))
        self.assertFalse(np.array_equal(p, proposal(2, 4)))
        np.testing.assert_array_equal(p, proposal(1, 4))

    def test_raw_gaussian_mean_is_not_clipped_before_noise(self):
        class P: layers = [(np.zeros((8, 48)), np.full(8, 2.))]
        mean = raw_mean(P(), np.zeros((2, 48)))
        self.assertTrue((mean == 2).all())
        action = choose_action(mean, np.ones(8), np.full((2, 8), -1.5), 0, np.zeros((2, 8)), 0)
        self.assertTrue((action == .5).all())

    def test_digest_excludes_suffix_but_detects_prefix_changes(self):
        t = dict(observations=np.zeros((51, 2, 48)), executed_actions=np.zeros((50, 2, 8)), flags=np.zeros((50, 2, 7)))
        digest = prefix_digest(t)
        t['observations'][31:] = 1
        self.assertEqual(prefix_digest(t), digest)
        t['observations'][30] = 1
        self.assertNotEqual(prefix_digest(t), digest)

    def test_worker_has_no_state_write_or_candidate_code_loader(self):
        source = Path('research/prefix_search_probe.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr in
                             ('set_state', 'set_state_dict', 'set_sim_state', 'load') and
                             not (isinstance(c.func.value, ast.Name) and c.func.value.id in ('json', 'np'))
                             for c in calls))
        self.assertNotIn('allow_pickle=True', source)


if __name__ == '__main__': unittest.main()
