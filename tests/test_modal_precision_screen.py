import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('precision_screen_contract',Path(__file__).resolve().parents[1]/'deploy/modal/precision_screen_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class PrecisionScreenContracts(unittest.TestCase):
    def test_exact_three_recipes_and_random_seed_one(self):
        reviewed={arm:str(index)*64 for index,arm in enumerate(c.ARMS,1)}
        with patch.object(c,'REVIEWED',reviewed):
            for arm in c.ARMS:
                item=c.job(arm);args=c.command('python','source.py',item)
                self.assertEqual(item['seed'],1)
                self.assertEqual(item['trainer_seconds_cap'],2700)
                self.assertEqual(args[args.index('--total-timesteps')+1],'50000000')
                self.assertEqual(args[args.index('--num-envs')+1],'1024')
                self.assertNotIn('--checkpoint',args)
                self.assertIn('--no-partial-reset',args)
                with self.assertRaises(ValueError):c.command('python','source.py',dict(item,seed=2))
            with self.assertRaises(ValueError):c.job('unreviewed')

    def test_original_absolute_lease_after_queue_and_restart(self):
        self.assertEqual(c.remaining_lease(4000.,2000.,7.),(2000.,1987.))
        self.assertEqual(c.remaining_lease(4000.,3000.,7.),(1000.,987.))
        with self.assertRaises(RuntimeError):c.remaining_lease(4000.,3980.,7.)

    def test_prespecified_three_point_cadence_and_no_future_checkpoint(self):
        files={'ckpt_501.pt','ckpt_751.pt','final_ckpt.pt'}
        self.assertEqual([p['steps'] for p in c.select_checkpoints(files,c.MAX_STEPS)],[25600000,38400000,49971200])
        self.assertEqual(c.select_checkpoints({'final_ckpt.pt'},25600000),[{'steps':25600000,'filename':'final_ckpt.pt','point':'prespecified-final-coincidence'}])
        self.assertEqual(len(c.select_checkpoints({'ckpt_501.pt','final_ckpt.pt'},30003200)),2)
        with self.assertRaises(ValueError):c.select_checkpoints({'final_ckpt.pt'},38400000)
        with self.assertRaises(ValueError):c.select_checkpoints(files,25600000)
        with self.assertRaises(ValueError):c.select_checkpoints({'ckpt_1001.pt','final_ckpt.pt'},c.MAX_STEPS)

    def test_exact_optimizer_and_per_dimension_trace_validation(self):
        import copy
        tags=['audit/optimizer_steps','audit/training_entropy_coef','charts/learning_rate']+[f'audit/logstd_dim_{i}' for i in range(8)]
        values={tag:[-.5,-.4] for tag in tags}
        values.update({'audit/optimizer_steps':[4.,7.],'audit/training_entropy_coef':[.003,.003],'charts/learning_rate':[.0003,.0003]})
        scalars={tag:[{'steps':51200*(i+1),'value':v} for i,v in enumerate(vals)] for tag,vals in values.items()}
        audit=c.audit_accounting(scalars,'precision-constant',2)
        self.assertEqual(audit['optimizer_step_count'],7)
        self.assertEqual(audit['final_logstd_per_dimension'],[-.4]*8)
        for mutate in ('missing','nonfinite','nonmonotonic','wrong_schedule','wrong_steps'):
            broken=copy.deepcopy(scalars)
            if mutate=='missing':del broken['audit/logstd_dim_7']
            elif mutate=='nonfinite':broken['audit/logstd_dim_1'][1]['value']=None
            elif mutate=='nonmonotonic':broken['audit/optimizer_steps'][1]['value']=3.
            elif mutate=='wrong_schedule':broken['audit/training_entropy_coef'][1]['value']=.002
            else:broken['audit/optimizer_steps'][1]['steps']=51200
            with self.assertRaises(ValueError,msg=mutate):c.audit_accounting(broken,'precision-constant',2)

    def test_conservative_all_resource_bound_under_seven_dollars(self):
        cost=c.conservative_cost()
        self.assertEqual(cost['gpu_container_seconds_bound'],9720)
        self.assertLess(cost['subtotal_usd'],6.2)
        self.assertGreater(cost['reserve_usd'],.8)

if __name__=='__main__':unittest.main()
