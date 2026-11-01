import copy
import unittest

from scripts.analyze_gripper_hold import analyze, GRID, METHODS


def fixture():
    reports, receipts = [], []
    for method in METHODS:
        reports.append(dict(state='complete', method=method, seed=41, confirmation_queries=0,
            final_queries=0, environment_completed=32000, wall_seconds=10,
            points=[dict(steps=s, episodes=128, failures=0, successes=0 if s == 0 else
                         {'full':0, 'masked':16, 'hold':64}[method]) for s in GRID]))
        receipts.append(dict(state='complete', method=method, seed=41, trainer_process_seconds=250,
            wrapper_seconds=255, actual_work=dict(environment_completed=4096000, optimizer_completed=100)))
    return reports, receipts


class HoldAnalysisTests(unittest.TestCase):
    def test_all_controls_and_budget_accounted(self):
        reports, receipts = fixture()
        result = analyze(reports, receipts)
        self.assertTrue(result['single_seed_material_screen_passed'])
        self.assertEqual(result['new_training_transitions'], 12288000)
        self.assertEqual(result['new_public_transitions'], 96000)

    def test_winning_only_weaker_control_is_not_success(self):
        reports, receipts = fixture()
        for p in reports[0]['points']:
            p['successes'] = 100
        self.assertFalse(analyze(reports, receipts)['single_seed_material_screen_passed'])

    def test_missing_middle_point_rejected(self):
        reports, receipts = fixture()
        del reports[2]['points'][2]
        with self.assertRaises(ValueError):
            analyze(reports, receipts)

    def test_no_censored_threshold_speedup_claim(self):
        reports, receipts = fixture()
        result = analyze(reports, receipts)
        self.assertTrue(all(m['T80']['censored'] for m in result['methods'].values()))


if __name__ == '__main__':
    unittest.main()
