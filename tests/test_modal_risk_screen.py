import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/modal'))
spec=importlib.util.spec_from_file_location('risk_screen_contract',ROOT/'deploy/modal/risk_screen_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class RiskContinuationContracts(unittest.TestCase):
    def test_exact_checkpoint_command_and_visible_prefix(self):
        with patch.object(c,'REVIEWED',{arm:'1'*64 for arm in c.ARMS}):
            for arm in c.ARMS:
                args=c.command('python','trainer.py',c.job(arm))
                self.assertEqual(args[args.index('--checkpoint')+1],'/trainer-output/reference.pt')
                self.assertEqual(args[args.index('--total-timesteps')+1],'2048000')
                self.assertIn('--no-partial-reset',args)
        self.assertTrue(c.PREFIX['optimizer_state_restarted'])
        self.assertEqual(c.PREFIX['training_steps'],24985600)
        self.assertEqual(c.PREFIX['trainer_seconds'],1088.7667)
        self.assertEqual(c.PREFIX['training_seed'],1)

    def test_provenance_and_combined_input_bound_fail_closed(self):
        program=b'reviewed source';reference=b'root reference'
        with patch.object(c,'REVIEWED',{'risk-ppo-control':hashlib.sha256(program).hexdigest()}),patch.object(c,'REFERENCE_BYTES',len(reference)),patch.object(c,'REFERENCE_SHA256',hashlib.sha256(reference).hexdigest()):
            item=c.job('risk-ppo-control');c.validate_inputs(program,reference,item)
            with self.assertRaises(ValueError):c.validate_inputs(program,reference+b'x',item)
            with self.assertRaises(ValueError):c.validate_inputs(program,b'wrongreference',item)
            with self.assertRaises(ValueError):c.validate_inputs(program+b'x',reference,item)
            large=b'x'*(2*1024*1024-len(reference))
            with patch.object(c,'REVIEWED',{'risk-ppo-control':hashlib.sha256(large).hexdigest()}):
                with self.assertRaises(ValueError):c.validate_inputs(large,reference,c.job('risk-ppo-control'))

    def test_lease_consumes_only_current_dispatch_and_final_only(self):
        self.assertEqual(c.remaining_lease(1300.,1100.,5.),(200.,185.))
        with self.assertRaises(RuntimeError):c.remaining_lease(1300.,1280.,5.)
        self.assertEqual(c.select_checkpoints({'ckpt_1.pt','ckpt_26.pt','final_ckpt.pt'},2048000),[{'steps':2048000,'filename':'final_ckpt.pt','point':'final-continuation'}])
        with self.assertRaises(ValueError):c.select_checkpoints({'ckpt_51.pt','final_ckpt.pt'},2048000)

    def test_common_constant_audits_and_optional_zero_failure_tags(self):
        import copy
        values={'audit/optimizer_steps':[4.,7.],'audit/training_entropy_coef':[.003,.003],'charts/learning_rate':[.0003,.0003]}
        values.update({f'audit/logstd_dim_{i}':[-.5,-.4] for i in range(8)})
        values.update({'audit/risk/'+name:[.1,.1] for name in ('actor_alpha','actor_var_baseline','actor_raw_weight_rms','mean_discounted_return','score_weight_rms')})
        values['audit/risk/training_failures']=[0.,0.]
        scalars={tag:[{'steps':51200*(i+1),'value':v} for i,v in enumerate(vals)] for tag,vals in values.items()}
        result=c.audit_accounting(scalars,'risk-episode-cvar10',2)
        self.assertEqual(result['optimizer_step_count'],7)
        self.assertTrue(result['training_success_labels_diagnostic_only'])
        scalars['audit/risk/failure_enrichment']=[{'steps':51200,'value':None}]
        c.audit_accounting(scalars,'risk-episode-cvar10',2)
        missing=copy.deepcopy(scalars);del missing['audit/risk/actor_alpha']
        with self.assertRaises(ValueError):c.audit_accounting(missing,'risk-episode-cvar10',2)
        wrong=copy.deepcopy(scalars);wrong['audit/risk/actor_alpha'][0]['value']=1.
        with self.assertRaises(ValueError):c.audit_accounting(wrong,'risk-episode-cvar10',2)

    def test_reviewed_three_arm_budget_under_one_dollar(self):
        plan=json.loads((ROOT/'deploy/modal/risk-screen-plan.json').read_text())
        self.assertEqual(plan['jobs'],[c.job(arm) for arm in c.ARMS])
        if plan['launch_enabled']:
            self.assertIn('root_review',plan)
            self.assertTrue(plan['root_review']['checks'])
        self.assertEqual(plan['max_parallel_trainers'],1)
        self.assertEqual(len(c.ARMS),3)
        self.assertEqual(c.conservative_cost()['gpu_container_seconds_bound'],1530)
        self.assertLess(c.conservative_cost()['subtotal_usd'],.976)
        self.assertGreater(c.conservative_cost()['reserve_usd'],.024)

if __name__=='__main__':unittest.main()
