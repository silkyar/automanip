import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/modal'))
spec=importlib.util.spec_from_file_location('kl_value_screen_contract',ROOT/'deploy/modal/kl_value_screen_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture(method,deadline_skips=0):
    records=[];totals={key:0 for key in c.CUMULATIVE};scalars={}
    for index in range(2):
        stopped=index==0;joint=20 if stopped else 256
        only=256-joint-deadline_skips if stopped and method==c.ARMS[1] else 0
        skips=deadline_skips if stopped and method==c.ARMS[1] else 0
        row=dict(method=method,base_source_sha256=c.BASE_SHA,patch_source_sha256=c.PATCH_SHA,
                 new_environment_transitions=(index+1)*51200,configured_epochs=8,
                 reached_epoch_count=1 if stopped and method==c.ARMS[0] else 8,
                 kl_stopped=stopped,kl_stop_epoch_0based=0 if stopped else None,
                 kl_stop_minibatch_0based=20 if stopped else None,observed_kl=.2 if stopped else .01,
                 joint_actor_critic_optimizer_calls=joint,actor_optimizer_calls=joint,
                 critic_only_optimizer_calls=only,critic_optimizer_calls=joint+only,optimizer_calls_total=joint+only,
                 critic_only_training_rows=only*1600,critic_only_seconds=.25 if stopped and method==c.ARMS[1] else 0.,
                 full_optimizer_loop_seconds=1.,critic_only_deadline_skips=skips,
                 critic_only_mean_value_loss=.5 if only else None,actor_updates_after_kl_stop=0,
                 critic_only_actor_gradients_none=True,target_refreshes_in_critic_only_phase=0,new_models_or_teacher_data=False)
        for key,cumulative in c.CUMULATIVE.items():totals[key]+=row[key];row[cumulative]=totals[key]
        records.append(row)
    def trace(tag,value,step):scalars.setdefault(tag,[]).append(dict(steps=step,value=value))
    for row in records:
        step=row['new_environment_transitions']
        for tag,value in {'time/step':step,'audit/optimizer_steps':row['cumulative_joint_actor_critic_calls'],
            'audit/training_entropy_coef':.003,'charts/learning_rate':.0003,**{f'audit/logstd_dim_{i}':-.5 for i in range(8)}}.items():trace(tag,value,step)
        for key,value in row.items():
            if type(value) in (int,float):trace('audit/kl_value/'+key,value,step)
    return scalars,records


def logs(records):return ''.join('AUTOMANIP_KL_VALUE_UPDATES '+json.dumps(row)+'\n' for row in records)


class KLValueScreenContracts(unittest.TestCase):
    def test_image_helpers_import_in_isolation_without_host_repository(self):
        tree=ast.parse((ROOT/'deploy/modal/kl_value_screen.py').read_bytes())
        assignment=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HELPER_FILES' for t in node.targets))
        helpers=ast.literal_eval(assignment.value)
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary)
            for name in helpers:(stage/name).write_bytes((ROOT/'deploy/modal'/name).read_bytes())
            script="import sys,pathlib; sys.path.insert(0,sys.argv[1]); import kl_value_screen_contract as c; import precision_screen_contract as p; assert pathlib.Path(c.__file__).parent==pathlib.Path(sys.argv[1]); assert pathlib.Path(p.__file__).parent==pathlib.Path(sys.argv[1]); assert c.MAX_STEPS==10240000"
            result=subprocess.run([sys.executable,'-I','-S','-c',script,str(stage)],cwd=stage,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_exact_random_initialization_sources_cli_and_bounded_checkpoint_selection(self):
        for method in c.ARMS:
            raw=(ROOT/'research/discovery/from-scratch-kl-value-v2'/(method+'.py')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),c.REVIEWED[method])
            args=c.command('python','source.py',c.job(method))
            self.assertEqual(args[args.index('--total-timesteps')+1],'10240000')
            self.assertEqual(args[args.index('--target-kl')+1],'0.1')
            self.assertIn('--no-clip-vloss',args);self.assertIn('--no-partial-reset',args)
            self.assertNotIn('--checkpoint',args);self.assertFalse(any('assist' in x for x in args))
            with self.assertRaises(ValueError):c.command('python','source.py',dict(c.job(method),seed=2))
        files={'ckpt_101.pt','final_ckpt.pt'}
        self.assertEqual([r['steps'] for r in c.select_checkpoints(files,10240000)],[5120000,10240000])
        self.assertEqual(len(c.select_checkpoints({'final_ckpt.pt'},5120000)),1)
        self.assertEqual(c.select_checkpoints({'final_ckpt.pt'},2560000)[0]['steps'],2560000)
        with self.assertRaises(ValueError):c.select_checkpoints({'final_ckpt.pt'},10240000)
        with self.assertRaises(ValueError):c.select_checkpoints({'ckpt_226.pt','final_ckpt.pt'},10240000)

    def test_primary_total_counts_include_critic_only_and_preserve_legacy_joint(self):
        for method,total,only in [(c.ARMS[0],276,0),(c.ARMS[1],512,236)]:
            scalars,records=fixture(method);report=c.audit_accounting(scalars,method,2,logs(records))
            self.assertEqual(report['legacy_joint_optimizer_step_count'],276)
            self.assertEqual(report['optimizer_step_count'],total)
            self.assertEqual(report['measured_actor_optimizer_calls'],276)
            self.assertEqual(report['measured_joint_optimizer_calls'],276)
            self.assertEqual(report['measured_critic_optimizer_calls'],total)
            self.assertEqual(report['measured_critic_only_optimizer_calls'],only)
            self.assertEqual(report['critic_only_training_rows'],only*1600)
        scalars,records=fixture(c.ARMS[1],deadline_skips=136)
        self.assertEqual(c.audit_accounting(scalars,c.ARMS[1],2,logs(records))['optimizer_step_count'],376)

    def test_role_source_rng_and_epoch_stop_receipts_fail_closed(self):
        scalars,records=fixture(c.ARMS[1])
        for key,value in [('actor_optimizer_calls',21),('optimizer_calls_total',255),('cumulative_critic_optimizer_calls',999),
                          ('configured_epochs',9),('reached_epoch_count',7),('kl_stop_minibatch_0based',19),
                          ('critic_only_training_rows',1),('critic_only_deadline_skips',1),('actor_updates_after_kl_stop',1),
                          ('critic_only_actor_gradients_none',False),('target_refreshes_in_critic_only_phase',1),
                          ('new_models_or_teacher_data',True),('patch_source_sha256','0'*64),('observed_kl',float('nan'))]:
            changed=copy.deepcopy(records);changed[0][key]=value
            with self.assertRaises(ValueError,msg=key):c.audit_accounting(scalars,c.ARMS[1],2,logs(changed))
        changed=copy.deepcopy(records);changed[1]['critic_only_optimizer_calls']=1
        with self.assertRaises(ValueError):c.audit_accounting(scalars,c.ARMS[1],2,logs(changed))

    def test_numeric_event_traces_and_json_cannot_disagree_or_go_missing(self):
        scalars,records=fixture(c.ARMS[1])
        bad=copy.deepcopy(scalars);bad['audit/kl_value/cumulative_optimizer_calls_total'][-1]['value']=511
        with self.assertRaises(ValueError):c.audit_accounting(bad,c.ARMS[1],2,logs(records))
        bad=copy.deepcopy(scalars);del bad['audit/kl_value/critic_only_training_rows']
        with self.assertRaises(ValueError):c.audit_accounting(bad,c.ARMS[1],2,logs(records))
        with self.assertRaises(ValueError):c.audit_accounting(scalars,c.ARMS[1],2,logs(records[:1]))
        bad=copy.deepcopy(scalars);bad['audit/optimizer_steps'][-1]['value']=512
        with self.assertRaises(ValueError):c.audit_accounting(bad,c.ARMS[1],2,logs(records))

    def test_leases_shared_ledger_admission_and_resource_reserve(self):
        self.assertEqual(c.remaining_lease(1600,1400,10),(200.,190.))
        with self.assertRaises(RuntimeError):c.remaining_lease(1600,1580,10)
        plan=json.loads((ROOT/'deploy/modal/kl-value-screen-plan.json').read_bytes())
        with self.assertRaises(PermissionError):c.validate_admission(dict(plan,launch_enabled=False,paid_launch_authorized=False,root_review=None))
        admitted=dict(plan,launch_enabled=True,paid_launch_authorized=True,root_review={'fixture':True});c.validate_admission(admitted)
        wrong=copy.deepcopy(admitted);wrong['outcome_rule']['minimum_candidate_final_success_gain']=0
        with self.assertRaises(ValueError):c.validate_admission(wrong)
        ledger=dict(within_shared_cap=True,user_total_cap_usd='100',committed_worst_case_plus_pending_reservations_usd='14.24',
                    reservations=[{'campaign':c.EXPERIMENT,'reserved_cap_usd':'3'}]);c.validate_budget(ledger)
        ledger['reservations']*=2
        with self.assertRaises(PermissionError):c.validate_budget(ledger)
        cost=c.conservative_cost();self.assertEqual(cost['gpu_container_seconds_bound'],1980)
        self.assertLess(cost['subtotal_usd'],1.27);self.assertGreater(cost['reserve_usd'],1.73)

if __name__=='__main__':unittest.main()
