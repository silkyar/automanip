import unittest
import numpy as np
from scripts.analyze_release_maintenance import analyze,CONTRASTS
from research.release_maintenance_policy import BRANCHES


class MaintenanceAnalysisTests(unittest.TestCase):
    def test_factorial_interaction_separates_combined_repair(self):
        reports=[]
        for seed in [2,3]:
            rows=[]
            for b,p in zip(BRANCHES,[0,.25,.25,.75]):
                rows.append(dict(collection_seed=seed,branch=b,episodes=512,execution_failures=0,batch_success_rates=[p]*32,successes=int(512*p)))
            reports.append(dict(state='complete',new_simulator_transitions_reserved=102400,new_simulator_transitions_completed=102400,optimizer_calls=0,records=rows,wall_seconds=1))
        r=analyze(reports)
        self.assertEqual(r['comparisons']['factorial_interaction']['mean_difference'],.25)
        self.assertTrue(all(r['material_maintenance_screen'].values()))
        self.assertTrue(all(sum(c)==0 for c in CONTRASTS.values()))

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError):analyze([{'state':'failed'}]*2)


if __name__=='__main__':unittest.main()
