import unittest
from scripts.analyze_scratch_credit import normalized_auc,sustained_crossing,analyze


class ScratchCreditAnalysisTests(unittest.TestCase):
    def test_auc_uses_actual_transition_spacing(self):
        # The long final interval dominates; averaging point successes is wrong.
        self.assertAlmostEqual(normalized_auc([(0,0),(1,1),(10,1)]),.95)

    def test_one_isolated_success_does_not_confirm_threshold(self):
        result=sustained_crossing([(0,0),(1,.9),(2,.7),(3,.9)],.8)
        self.assertTrue(result['censored']);self.assertIsNone(result['observed_first_step'])

    def test_crossing_has_distinct_observation_and_confirmation_times(self):
        result=sustained_crossing([(0,0),(100,.81),(300,.85)],.8)
        self.assertEqual(result,dict(observed_first_step=100,confirmed_at=300,censored=False))

    def test_incomplete_or_nonmonotonic_auc_rejected(self):
        for points in [[(1,.5),(2,.6)],[(0,0)],[(0,0),(2,.5),(1,.5)]]:
            with self.assertRaises(ValueError):normalized_auc(points)

    def test_other_method_names_preserve_paired_auc(self):
        reports=[];receipts=[]
        for seed in [1,2,3]:
            for method,n in [('state',0),('clock',64)]:
                reports.append(dict(state='complete',seed=seed,method=method,final_queries=0,confirmation_queries=0,points=[dict(steps=s,episodes=128,failures=0,successes=k) for s,k in [(0,0),(512000,n)]]))
                receipts.append(dict(state='complete',seed=seed,method=method,actual_work=dict(environment_completed=512000,optimizer_completed=10),trainer_process_seconds=4,wrapper_seconds=5))
        result=analyze(reports,receipts,methods=['state','clock'])
        self.assertEqual(result['mean_paired_auc_gain'],.25)
        with self.assertRaises(ValueError):analyze(reports,receipts,methods=['clock','clock'])


if __name__=='__main__':unittest.main()
