"""CPU registration checks; importing the probe constructs no simulator."""
import unittest
from research.recovery_probe import direction,force_vectors,classify,PUSH_STEP,PUSH_NEWTONS


class RecoveryProbeTests(unittest.TestCase):
    def test_fixed_force_and_balanced_directions(self):
        self.assertEqual(PUSH_STEP,20);self.assertEqual(PUSH_NEWTONS,8.)
        seeds=list(range(93460000,93460128))
        self.assertEqual(sum(direction(s) for s in seeds),0)
        self.assertEqual(force_vectors([2,3],True),[[0.,8.,0.],[0.,-8.,0.]])
        self.assertEqual(force_vectors([2,3],False),[[0.,0.,0.],[0.,0.,0.]])
        with self.assertRaises(ValueError):direction(True)
    def test_decision_gates_do_not_admit_trivial_or_invalid_training(self):
        self.assertEqual(classify(124,122),'zero_shot_coverage_no_learning_gap')
        self.assertEqual(classify(124,100),'material_recovery_gap_adaptation_probe_eligible')
        self.assertEqual(classify(124,1),'large_deficit_check_feasibility_before_learning')
        self.assertEqual(classify(110,80),'source_applicability_failed')
        self.assertEqual(classify(116,115),'insufficient_material_gap')
        self.assertEqual(classify(124,80,execution_failures=1),'invalid_execution')
        with self.assertRaises(ValueError):classify(129,100)
