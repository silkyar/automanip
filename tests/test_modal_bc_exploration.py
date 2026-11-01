import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/modal'))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

c=module('bc_exploration_contract',ROOT/'deploy/modal/bc_exploration_contract.py')
# Public synthetic fixtures only; no benchmark data or production episode output.
base=module('amp_public_fixtures',ROOT/'tests/test_modal_amp_pilot.py')


def fixture(stage):
    scalars,log,bindings,metadata,bc=base.student_fixture('assist-bc-ppo')
    metadata['public_training_seed_start']=c.TRAINING_SEED_START
    receipt=None
    if stage==c.STUDENTS[1]:
        rms=[0.,.06,.1,.2,.3,.5,.7,2.]
        initial=[min(max(v,.05),math.exp(-.5)) for v in rms]
        initial[0]=.04999999701976776  # Real float32 exp(log(.05)) rounding.
        receipt=dict(mechanism='training-demo residual RMS initializes existing per-coordinate Gaussian std only',
            dataset_sha256=metadata['sha256'],bc_actor_weights_sha256=bc['final']['weights_sha256'],
            residual_patch_source_sha256=c.PATCH_SHA,frozen_assisted_base_sha256=c.BASE_SHA,
            training_episode_indices=bc['train_episode_indices'],excluded_validation_episode_indices=bc['validation_episode_indices'],
            residual_forward_rows=50*len(bc['train_episode_indices']),residual_rms=rms,original_std=[.6065306663513184]*8,
            initial_std=initial,floor=.05,ceiling=math.exp(-.5),trainable_after_initialization=True,
            new_environment_transitions=0,new_optimizer_updates=0,mean_critic_reward_architecture_unchanged=True,
            calibration_wall_seconds=.25,note='Synthetic contract fixture, not a production std vector.')
        for i in range(8):
            scalars[f'audit/assist_init/original_std_dim_{i}']=[dict(steps=0,value=receipt['original_std'][i])]
            scalars[f'audit/assist_init/residual_std_dim_{i}']=[dict(steps=0,value=receipt['initial_std'][i])]
        scalars['audit/assist_init/calibration_forward_rows']=[dict(steps=0,value=receipt['residual_forward_rows'])]
        scalars['audit/assist_init/calibration_wall_seconds']=[dict(steps=0,value=.25)]
        log+='AUTOMANIP_EXPLORATION_INITIALIZATION '+json.dumps(receipt)+'\n'
    return scalars,log,bindings,metadata,bc,receipt


class BCExplorationContracts(unittest.TestCase):
    def test_exact_v2_sources_new_training_seeds_and_no_teacher_or_shaping_cli(self):
        self.assertEqual(c.RESET_SEEDS,tuple(range(90281000,90281008)))
        core=(ROOT/'research/discovery/amp_v1.py').read_bytes()
        for stage in c.STAGES:
            payload=(ROOT/c.SOURCES[stage][0]).read_bytes();c.verify_sources(payload,stage,core)
            with self.assertRaises(ValueError):c.verify_sources(payload+b'\n',stage,core)
            args=c.command(stage,'python','source.py')
            self.assertNotIn('--checkpoint',args);self.assertNotIn('--local-cpu-dataflow-smoke',args)
        self.assertEqual(c.SOURCES[c.STUDENTS[1]][1],'c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200')
        for stage in c.STUDENTS:
            self.assertEqual(c.readonly_inputs(stage),{'/demo':('collect','/trial/dataset'),'/bc':('bc','/trial/published')})
            args=c.command(stage,'python','source.py');self.assertEqual(args[args.index('--total-timesteps')+1],'5120000')

    def test_both_students_bind_shared_bc_and_calibration_has_exact_eight_coordinates(self):
        for stage in c.STUDENTS:
            scalars,log,bindings,m,bc,receipt=fixture(stage)
            result=c.student_accounting(stage,scalars,log,bindings,m,bc)
            self.assertEqual(result['actual_steps'],51200);self.assertEqual(result['optimizer_step_count'],20)
            self.assertFalse(result['assistance']['amp_enabled']);self.assertEqual(result['assistance']['amp_style_weight'],0.)
            self.assertNotIn('discriminator_updates',result)
            if receipt:
                self.assertEqual(result['exploration_initialization']['initial_std'],receipt['initial_std'])
                self.assertEqual(receipt['residual_forward_rows'],22500)
            with self.assertRaises(ValueError):c.student_accounting(stage,scalars,log.replace('b'*64,'f'*64),bindings,m,bc)

    def test_calibration_partitions_hashes_rows_bounds_and_time_fail_closed(self):
        scalars,log,bindings,m,bc,receipt=fixture(c.STUDENTS[1])
        for field,value in [('dataset_sha256','0'*64),('bc_actor_weights_sha256','0'*64),('residual_patch_source_sha256','0'*64),
                            ('frozen_assisted_base_sha256','0'*64),('training_episode_indices',[450]),
                            ('excluded_validation_episode_indices',[0]),('residual_forward_rows',22501),
                            ('new_environment_transitions',1),('new_optimizer_updates',1),('trainable_after_initialization',False),
                            ('initial_std',[.1]*7),('original_std',[.5]*8),('calibration_wall_seconds',float('inf'))]:
            changed=copy.deepcopy(receipt);changed[field]=value
            bad='AUTOMANIP_EXPLORATION_INITIALIZATION '+json.dumps(changed)+'\n'
            with self.assertRaises(ValueError,msg=field):c.validate_initialization(c.STUDENTS[1],scalars,bad,m,bc)
        changed=copy.deepcopy(receipt);changed['initial_std'][0]=.04
        bad='AUTOMANIP_EXPLORATION_INITIALIZATION '+json.dumps(changed)+'\n'
        with self.assertRaises(ValueError):c.validate_initialization(c.STUDENTS[1],scalars,bad,m,bc)
        with self.assertRaises(ValueError):c.validate_initialization(c.STUDENTS[1],scalars,'',m,bc)

    def test_step_zero_evidence_required_and_both_arms_reject_discriminator_traces(self):
        for stage in c.STUDENTS:
            scalars,log,bindings,m,bc,receipt=fixture(stage)
            scalars['audit/amp/discriminator_updates']=[dict(steps=51200,value=32)]
            with self.assertRaises(ValueError):c.student_accounting(stage,scalars,log,bindings,m,bc)
        scalars,log,bindings,m,bc,receipt=fixture(c.STUDENTS[1])
        scalars['audit/assist_init/residual_std_dim_0'][0]['steps']=1
        with self.assertRaises(ValueError):c.student_accounting(c.STUDENTS[1],scalars,log,bindings,m,bc)
        scalars,log,bindings,m,bc,receipt=fixture(c.STUDENTS[1]);del scalars['audit/assist_init/original_std_dim_7']
        with self.assertRaises(ValueError):c.student_accounting(c.STUDENTS[1],scalars,log,bindings,m,bc)
        with self.assertRaises(ValueError):c.validate_initialization(c.STUDENTS[0],scalars,log,m,bc)

    def test_cpu_demo_rejection_original_leases_default_denial_and_reservation_replacement(self):
        m=base.metadata();m['public_training_seed_start']=c.TRAINING_SEED_START;c.validate_collection(m,base.native('collect'))
        m['simulation_backend']='physx_cpu'
        with self.assertRaises(ValueError):c.validate_collection(m,base.native('collect'))
        self.assertEqual(c.remaining_lease(c.STUDENTS[1],1600,1000,1400,10),(200.,190.))
        with self.assertRaises(RuntimeError):c.remaining_lease(c.STUDENTS[1],1600,1000,1580,10)
        plan=json.loads((ROOT/'deploy/modal/bc-exploration-plan.json').read_bytes())
        blocked=dict(plan,launch_enabled=False,paid_launch_authorized=False,root_review=None)
        with self.assertRaises(PermissionError):c.validate_admission(blocked)
        admitted=copy.deepcopy(plan);admitted.update(launch_enabled=True,paid_launch_authorized=True,root_review={'fixture':True})
        c.validate_admission(admitted)
        ledger={'within_shared_cap':True,'user_total_cap_usd':'100','committed_worst_case_plus_pending_reservations_usd':'14.57',
                'reservations':[{'campaign':'stackcube-amp-pilot-20261028','reserved_cap_usd':'3'}]}
        with self.assertRaises(PermissionError):c.validate_budget(ledger)
        ledger['reservations'][0]['campaign']=c.EXPERIMENT;c.validate_budget(ledger)
        self.assertLess(c.conservative_cost()['subtotal_usd'],1.44)

if __name__=='__main__':unittest.main()
