import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/modal'))
spec=importlib.util.spec_from_file_location('precision_replication_contract',ROOT/'deploy/modal/precision_replication_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def admitted(recipe):
    jobs=c.selected_jobs(recipe)
    counts={'precision-constant':126,'precision-entropy-anneal':127,'precision-lr-anneal':128} if recipe=='precision-constant' else {'precision-constant':125,'precision-entropy-anneal':126,'precision-lr-anneal':127}
    return {'experiment_id':c.EXPERIMENT,'launch_enabled':True,'paid_launch_authorized':True,'root_review_required':False,'selected_recipe':recipe,'jobs':jobs,'training_seeds':[2,3],'confirmation_queries':0,'max_parallel_trainers':len(jobs),'max_parallel_evaluators':len(jobs),'budget_authorization_usd':c.BRANCH_CAPS[len(jobs)],'cost':c.conservative_cost(recipe),'protocol_sha256':'fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764','development_episodes':128,'selected_steps':[25600000,38400000,'final_actual_steps'],'max_development_evaluations':3*len(jobs),'outcome_qualification':{'selected_recipe':recipe,'initial_campaign':'stackcube-precision-screen-20261027','root_reviewed':True,'final_development_counts':counts,'episodes':128}}


class PrecisionReplicationContracts(unittest.TestCase):
    def test_unadmitted_plan_cannot_launch(self):
        plan=json.loads((ROOT/'deploy/modal/precision-replication-plan.json').read_text())
        # Admission files become enabled only after separate root outcome review.
        # The guard must still reject any unadmitted copy regardless of job slots.
        blocked=dict(plan,launch_enabled=False,paid_launch_authorized=False,root_review_required=True)
        with self.assertRaises(PermissionError):c.validate_admission(blocked)

    def test_only_two_complete_branches_and_exact_seed_source_cli(self):
        for recipe in c.ARMS:
            plan=admitted(recipe);jobs=c.validate_admission(plan)
            self.assertEqual(len(jobs),2 if recipe=='precision-constant' else 4)
            self.assertEqual({j['seed'] for j in jobs},{2,3})
            self.assertEqual(len({j['job_id'] for j in jobs}),len(jobs))
            self.assertEqual(len({c.volume_name(j) for j in jobs}),len(jobs))
            for item in jobs:
                args=c.command('python','trainer.py',item)
                self.assertEqual(args[args.index('--seed')+1],str(item['seed']))
                self.assertEqual(args[args.index('--exp-name')+1],item['job_id'])
                self.assertEqual(item['trainer_sha256'],c.REVIEWED[item['method']])
                self.assertNotIn('--checkpoint',args)
        for seed in (1,4,True):
            with self.assertRaises(ValueError):c.job('precision-constant',seed)

    def test_root_outcome_admission_and_branch_binding_fail_closed(self):
        valid=admitted('precision-entropy-anneal')
        for mutation in ('paid','root','below_threshold','constant_preference','duplicate','hash','extra_seed','concurrency','budget','protocol','eval_count'):
            plan=copy.deepcopy(valid)
            if mutation=='paid':plan['paid_launch_authorized']=False
            elif mutation=='root':plan['outcome_qualification']['root_reviewed']=False
            elif mutation=='below_threshold':plan['outcome_qualification']['final_development_counts']['precision-entropy-anneal']=125
            elif mutation=='constant_preference':plan['outcome_qualification']['final_development_counts']['precision-constant']=126
            elif mutation=='duplicate':plan['jobs'][1]=copy.deepcopy(plan['jobs'][0])
            elif mutation=='hash':plan['jobs'][0]['trainer_sha256']='0'*64
            elif mutation=='extra_seed':plan['jobs'][0]['seed']=4
            elif mutation=='concurrency':plan['max_parallel_trainers']=3
            elif mutation=='budget':plan['budget_authorization_usd']=10
            elif mutation=='protocol':plan['protocol_sha256']='0'*64
            else:plan['max_development_evaluations']=13
            with self.assertRaises((PermissionError,ValueError),msg=mutation):c.validate_admission(plan)

    def test_branch_bounds_and_original_queue_consumed_lease(self):
        constant=c.conservative_cost('precision-constant');refinement=c.conservative_cost('precision-lr-anneal')
        self.assertLess(constant['subtotal_usd'],4.12);self.assertGreater(constant['reserve_usd'],.38)
        self.assertLess(refinement['subtotal_usd'],8.21);self.assertGreater(refinement['reserve_usd'],.79)
        self.assertEqual(c.remaining_lease(5000.,4000.,9.),(1000.,989.))
        with self.assertRaises(RuntimeError):c.remaining_lease(5000.,4980.,9.)

if __name__=='__main__':unittest.main()
