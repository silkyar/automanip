import ast
from pathlib import Path
import unittest

from automanip.capg import build_trainer as base_trainer
from automanip.precision import METHODS, build_trainer, refinement_schedule, strip_precision


class PrecisionTests(unittest.TestCase):
    def test_recipes_are_identical_until_refinement_and_have_exact_endpoints(self):
        for method in METHODS:
            for step in (0, 10_240_000, 24_985_600, 25_600_000):
                self.assertEqual(refinement_schedule(step, method), (.003, .0003))
        self.assertEqual(refinement_schedule(49_971_200, METHODS[0]), (.003, .0003))
        self.assertEqual(refinement_schedule(49_971_200, METHODS[1]), (0., .0003))
        self.assertAlmostEqual(refinement_schedule(49_971_200, METHODS[2])[1], .00003)
        self.assertEqual(refinement_schedule(60_000_000, METHODS[1]), (0., .0003))
        with self.assertRaises(ValueError):
            refinement_schedule(0, 'unregistered')

    def test_schedules_are_monotone_and_only_change_the_intended_parameter(self):
        steps = range(25_600_000, 49_971_201, 51_200)
        for method, changed in ((METHODS[1], 0), (METHODS[2], 1)):
            values = [refinement_schedule(step, method) for step in steps]
            self.assertTrue(all(a[changed] >= b[changed] for a, b in zip(values, values[1:])))
            self.assertTrue(all(v[1 - changed] == (.0003 if changed == 0 else .003) for v in values))

    def test_exact_source_scope_and_real_optimizer_counter(self):
        official = (Path(__file__).resolve().parents[1] / 'third_party/maniskill/ppo.py').read_bytes()
        expected = base_trainer(official, 'full-horizon-control').decode()
        for method in METHODS:
            source = build_trainer(official, method).decode()
            self.assertEqual(strip_precision(source), expected)
            tree = ast.parse(source)
            increments = [n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
                          and isinstance(n.target, ast.Name) and n.target.id == 'automanip_optimizer_steps']
            self.assertEqual(len(increments), 1)
            self.assertIn('optimizer.step()\n                # BEGIN PRECISION counter-step\n                automanip_optimizer_steps += 1', source)
            self.assertIn('audit/logstd_dim_', source)


if __name__ == '__main__':
    unittest.main()
