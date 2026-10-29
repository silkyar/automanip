import ast
import copy
from pathlib import Path
import unittest

from automanip.research import validate_campaign, training_command
from automanip.settling import campaign, entropy_coefficient, trainer_bytes, strip_instrumentation, upgrade_legacy


class SettlingTests(unittest.TestCase):
    def test_exact_recipe_registry_and_fixed_interface(self):
        config=campaign();validate_campaign(config)
        for key,value in [('requested_transitions',50_000_000),('training_seeds',[1])]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):validate_campaign(bad)
        bad=copy.deepcopy(config);bad['candidates'][0]['reward_bonus']=1
        with self.assertRaises(ValueError):validate_campaign(bad)
        for candidate in config['candidates']:
            command=training_command('python','ppo.py',candidate,1)
            self.assertEqual('--no-partial-reset' in command,candidate['partial_reset'] is False)
            self.assertNotIn('--checkpoint',command)
            self.assertEqual(command[command.index('--total-timesteps')+1],'25000000')
            self.assertEqual(candidate['max_training_seconds'],1320)

    def test_entropy_schedule_has_fixed_transition_boundaries(self):
        total=24_985_600
        self.assertEqual(entropy_coefficient(0,total),.003)
        self.assertEqual(entropy_coefficient(15_000_000,total),.003)
        self.assertAlmostEqual(entropy_coefficient((total+15_000_000)/2,total),.0015)
        self.assertEqual(entropy_coefficient(total,total),0.)
        self.assertEqual(entropy_coefficient(total+1,total),0.)
        self.assertEqual(entropy_coefficient(400,400),.003)

    def test_derivative_is_exactly_schedule_insertion(self):
        official=(Path(__file__).resolve().parents[1]/'third_party/maniskill/ppo.py').read_bytes()
        candidates=campaign()['candidates']
        for candidate in candidates:
            modified=trainer_bytes(official,candidate)
            ast.parse(modified)
            self.assertEqual(strip_instrumentation(modified.decode()).encode(),official)
            self.assertIn(b'AUTOMANIP_TRAINING_DEADLINE',modified)
            self.assertIn(b'audit/actor_gradient_norm',modified)
            self.assertEqual(b'logger.add_scalar("charts/training_entropy_coef"' in modified,candidate['entropy_schedule']!='constant')
        with self.assertRaises(ValueError):trainer_bytes(official+b'\n',candidates[1])

    def test_legacy_amendment_is_exact_and_adds_timed_control(self):
        legacy=campaign()
        legacy['campaign_id']='stackcube-settling-20261027'
        legacy['hypothesis']='Test full-horizon training, late entropy decay, and discount/entropy interaction against the 83.6% development reference.'
        legacy['candidates']=[{k:v for k,v in c.items() if k!='max_training_seconds'} for c in legacy['candidates'][1:]]
        amended=upgrade_legacy(legacy)
        self.assertEqual(amended,campaign())
        self.assertEqual(len(amended['candidates']),4)
        legacy['candidates'][0]['gamma']=.99
        with self.assertRaises(ValueError):upgrade_legacy(legacy)


if __name__=='__main__':unittest.main()
