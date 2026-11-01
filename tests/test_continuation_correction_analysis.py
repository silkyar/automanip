import unittest
from scripts.analyze_continuation_correction import analyze
from research.continuation_correction_policy import BRANCHES


class WaitingAnalysisTests(unittest.TestCase):
    def test_strong_fixed_time_and_waiting_controls_are_compared(self):
        reports=[]
        for seed in [2,3]:
            records=[dict(collection_seed=seed,branch=b,episodes=128,execution_failures=0,batch_success_rates=[p]*8,successes=int(p*128),batch_stage_counts=[dict(grasp=16)]*8) for b,p in zip(BRANCHES,[.5,.5,.625,.875])]
            reports.append(dict(state='complete',new_simulator_transitions_reserved=25600,new_simulator_transitions_completed=25600,optimizer_calls=0,confirmation_queries=0,final_queries=0,records=records,wall_seconds=1))
        r=analyze(reports);self.assertTrue(r['registered_overall_screen_passed']);self.assertEqual(r['contrasts']['state_continuation-minus-learned_fixed_time']['mean_gain'],.375)

    def test_failed_physics_is_not_zero_success(self):
        with self.assertRaises(ValueError):analyze([{'state':'failed'}]*2)


if __name__=='__main__':unittest.main()
