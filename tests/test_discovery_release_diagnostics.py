import unittest

from research.discovery.release_diagnostics import summarize_episode


def episode(release_step=20, off_stack_after=None, last_next_value=4., terminal=False):
    snapshots = [dict(step=t, on_stack=t >= 10 and (off_stack_after is None or t < off_stack_after),
                      grasped=1 <= t < release_step, static=t >= release_step+2,
                      xy_offset=.01, z_error=.001,
                      linear_speed=.005 if t >= release_step+2 else .02,
                      angular_speed=.1 if t >= release_step+2 else .6)
                 for t in range(51)]
    rows = [dict(step=t, reward=1., value=2., next_value=last_next_value if t == 50 else 2.,
                 advantage=.1, normalized_advantage=-.2,
                 true_terminal=terminal and t == 50, truncated=not terminal and t == 50)
            for t in range(1, 51)]
    return snapshots, rows


class ReleaseDiagnosticsTests(unittest.TestCase):
    def test_placement_and_release_coverage_do_not_identify_timing(self):
        early = summarize_episode(*episode(20))
        late = summarize_episode(*episode(49))
        self.assertEqual(early["first_on_stack_step"], late["first_on_stack_step"])
        self.assertEqual(early["releases"][0]["remaining_steps"], 30)
        self.assertEqual(late["releases"][0]["remaining_steps"], 1)
        self.assertEqual(late["first_on_stack_to_first_release_steps"], 39)

    def test_alignment_at_release_separate_from_later_geometry_loss(self):
        before, rows = episode(20, off_stack_after=19)
        after, other_rows = episode(20, off_stack_after=25)
        a, b = summarize_episode(before, rows), summarize_episode(after, other_rows)
        self.assertFalse(a["releases"][0]["on_stack_after"])
        self.assertFalse(a["geometry_lost_after_release"])
        self.assertTrue(b["releases"][0]["on_stack_after"])
        self.assertTrue(b["geometry_lost_after_release"])
        self.assertTrue(b["upstream_success_lost_later"])

    def test_conditional_td_uses_final_successor_at_truncation(self):
        truncated = summarize_episode(*episode())
        terminal = summarize_episode(*episode(terminal=True))
        a = truncated["phase_statistics"]["on_stack_released_static"]
        b = terminal["phase_statistics"]["on_stack_released_static"]
        self.assertAlmostEqual((a["mean_td_residual"]-b["mean_td_residual"])*a["count"], 3.6)
        self.assertEqual(a["raw_to_normalized_advantage_sign_changes"], a["count"])

    def test_no_cross_episode_or_nonfinite_records(self):
        states, rows = episode()
        rows[19]["truncated"] = True
        with self.assertRaises(ValueError):
            summarize_episode(states, rows)
        states, rows = episode()
        states[-1]["linear_speed"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_episode(states, rows)


if __name__ == "__main__":
    unittest.main()
