import copy
import unittest

from automanip.overnight import campaign, recipe, candidate_stats, choose_tuning, choose_efficiency
from automanip.research import validate_campaign, training_command


class OvernightTests(unittest.TestCase):
    def test_phases_have_only_registered_candidates_and_budgets(self):
        for phase in ('tune','short','verify'):
            config=campaign(phase)
            validate_campaign(config)
            bad=copy.deepcopy(config);bad['candidates'][0]['gamma']=.99
            with self.assertRaises(ValueError):validate_campaign(bad)
            bad=copy.deepcopy(config);bad['requested_transitions']=50_000_000
            with self.assertRaises(ValueError):validate_campaign(bad)
        bad=campaign('short');bad['candidates'].reverse()
        with self.assertRaises(ValueError):validate_campaign(bad)

    def test_compute_variants_preserve_environment_interface(self):
        for candidate in campaign('short')['candidates']:
            cmd=training_command('python','ppo.py',candidate,2,requested_transitions=12_500_000)
            for flag,value in [('--env-id','StackCube-v1'),('--num-envs','1024'),('--num-minibatches','32'),
                               ('--total-timesteps','12500000'),('--gamma','0.9')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertNotIn('--checkpoint',cmd)
        lr=training_command('python','ppo.py',recipe('anchor090','short','lr2'),1)
        self.assertEqual(lr[lr.index('--learning-rate')+1],'0.0006')

    def summary(self,entries):
        return dict(runs=[dict(candidate=candidate,seed=seed,status='complete',trainer_process_seconds=seconds,
                              final=dict(report=dict(success_at_end_rate=rate)))
                          for candidate,rates,seconds in entries for seed,rate in enumerate(rates,1)])

    def test_selection_requires_seed_repeats_and_handles_regressions(self):
        anchor=dict(candidate=recipe('anchor090','short'),rates=[.6,.8,.8],mean=2.2/3,mean_training_seconds=1200)
        unstable=self.summary([(recipe('gamma0875'),[.1,.99,.99],1200)])
        self.assertIs(choose_tuning(unstable,anchor),anchor)
        better=self.summary([(recipe('gamma0925'),[.7,.85,.85],1200)])
        self.assertGreater(choose_tuning(better,anchor)['mean'],anchor['mean'])
        better['runs'].pop()
        with self.assertRaises(ValueError):candidate_stats(better)

    def test_efficiency_requires_measured_gain_not_just_a_new_setting(self):
        control=recipe('anchor090','short');lr=recipe('anchor090','short','lr2');epochs=recipe('anchor090','short','epochs4')
        flat=self.summary([(control,[.6,.6,.6],700),(lr,[.6,.6,.6],710),(epochs,[.59,.59,.59],680)])
        self.assertIsNone(choose_efficiency(flat))
        faster=self.summary([(control,[.6,.6,.6],700),(lr,[.6,.6,.6],710),(epochs,[.59,.59,.59],600)])
        self.assertEqual(choose_efficiency(faster)['candidate'],epochs)


if __name__=='__main__':unittest.main()
