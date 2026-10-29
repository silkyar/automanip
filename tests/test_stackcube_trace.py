import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

class TraceSelectionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy not installed")
    def test_examples_use_development_and_distinct_failure_stages(self):
        from scripts.trace_stackcube import select_cases
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            failures = ("grasped_not_lifted", "placed_still_grasped",
                        "released_not_stable", "lost_stack_before_horizon")
            for seed in (1, 2, 3):
                directory = root / "runs" / f"stackcube-seed-{seed}"
                directory.mkdir(parents=True)
                roles = ["success", *failures] if seed == 1 else ["success"]
                report = dict(episodes=[dict(diagnostics=dict(first_unmet_stage=role)) for role in roles])
                (directory / "final-development.json").write_text(json.dumps(report))
            cases = select_cases(root)
            self.assertEqual(len(cases), 7)
            self.assertEqual({c["training_seed"] for c in cases if c["role"] == "success"}, {1, 2, 3})
            self.assertEqual({c["role"] for c in cases if c["role"] != "success"}, set(failures))
            self.assertEqual(len({(c["training_seed"], c["episode_index"]) for c in cases}), 7)


if __name__ == "__main__": unittest.main()
