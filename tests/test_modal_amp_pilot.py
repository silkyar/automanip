import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/modal'))
spec=importlib.util.spec_from_file_location('amp_pilot_contract',ROOT/'deploy/modal/amp_pilot_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def native(stage):
    return dict(state='complete',versions=dict(c.PINS),gpu_actual_name='NVIDIA L40S',source_sha256=c.SOURCES[stage][1],core_source_sha256=c.CORE_SHA)


def metadata():
    return dict(format='automanip-public-training-demos-v1',env_id='StackCube-v1',observation_dim=48,action_dim=8,horizon=50,
                control_mode='pd_joint_delta_pos',obs_mode='state',reward_mode='normalized_dense',simulation_backend='physx_cuda',
                source_partition='public-training',versions=dict(c.PINS),teacher_actor_weights_sha256=c.TEACHER_SHA,
                teacher_prefix_transitions=c.PREFIX['training_steps'],teacher_prefix_trainer_seconds=c.PREFIX['trainer_seconds'],
                teacher_prefix_dollars=None,collector_source_sha256=c.SOURCES['collect'][1],public_training_seed_start=c.TRAINING_SEED_START,
                num_envs=64,requested_episodes=512,completed_episodes=512,successful_episodes=500,failed_completed_episodes=12,
                status='completed',error=None,sha256='d'*64,environment_transitions=25600,vector_step_calls=400,
                incomplete_transition_rows=0,explicit_reset_calls=8,explicit_reset_episodes=512)


def bc_report():
    return dict(teacher_weights_loaded=False,student_initialization='random',comparison_eligible=True,device='cuda',seed=1,
                dataset_sha256='d'*64,bc_source_sha256=c.SOURCES['bc'][1],core_source_sha256=c.CORE_SHA,
                training_torch_version=c.PINS['torch'],training_numpy_version=c.PINS['numpy'],
                teacher_prefix_transitions=c.PREFIX['training_steps'],teacher_prefix_trainer_seconds=c.PREFIX['trainer_seconds'],
                teacher_prefix_dollars=None,teacher_generation_transitions=25600,student_environment_transitions=0,
                total_dataset_episodes=512,failed_teacher_episodes_preserved=12,optimizer_updates=5000,
                demonstration_minibatch_rows=5120000,train_episode_indices=list(range(450)),validation_episode_indices=list(range(450,500)),
                training_demo_episodes=450,validation_demo_episodes=50,validation_action_mse=.01,final={'weights_sha256':'b'*64})


def student_fixture(stage):
    m=metadata();bc=bc_report();bindings={'demo':{'dataset.json':'a'*64,'trajectories.npz':'d'*64},'bc':{'bc-report.json':'c'*64}}
    assistance=dict(teacher_weights_loaded=False,dataset_sha256='d'*64,bc_report_sha256='c'*64,bc_student_weights_sha256='b'*64,
                    bc_prefix_optimizer_updates=5000,bc_prefix_demonstration_minibatch_rows=5120000,
                    teacher_prefix_transitions=c.PREFIX['training_steps'],teacher_prefix_trainer_seconds=c.PREFIX['trainer_seconds'],
                    teacher_prefix_dollars=None,teacher_generation_transitions=25600,all_teacher_episodes=512,failed_teacher_episodes_preserved=12,
                    amp_enabled=stage==c.STUDENTS[1],amp_style_weight=.25 if stage==c.STUDENTS[1] else 0.,task_reward_weight=1.,
                    discriminator_models_in_agent_serialization=False)
    values={'time/step':51200,'audit/optimizer_steps':20,'audit/training_entropy_coef':.003,'charts/learning_rate':.0003,
            **{f'audit/logstd_dim_{i}':-.5 for i in range(8)}}
    if stage==c.STUDENTS[1]:
        values.update({'audit/amp/'+k:v for k,v in dict(discriminator_updates=32,discriminator_sample_rows=16384,
            learner_transition_rows=51200,replay_rows=51200,discriminator_update_seconds=.1,mean_style_reward=.3,
            loss=.1,least_squares=.1,gradient_penalty=.1,expert_score=.2,policy_score=-.2).items()})
    scalars={tag:[{'steps':51200,'value':value}] for tag,value in values.items()}
    return scalars,'AUTOMANIP_ASSISTANCE '+json.dumps(assistance)+'\n',bindings,m,bc


class AmpPilotContracts(unittest.TestCase):
    def test_exact_sources_seeds_cli_and_readonly_input_surfaces(self):
        self.assertEqual(c.RESET_SEEDS,tuple(range(90280000,90280008)))
        core=(ROOT/'research/discovery/amp_v1.py').read_bytes()
        for stage in c.STAGES:
            program=(ROOT/c.SOURCES[stage][0]).read_bytes();c.verify_sources(program,stage,core)
            with self.assertRaises(ValueError):c.verify_sources(program+b'\n',stage,core)
            args=c.command(stage,'python','source.py')
            self.assertNotIn('--checkpoint',args);self.assertNotIn('--local-cpu-dataflow-smoke',args)
        self.assertEqual(c.readonly_inputs('collect'),{})
        self.assertEqual(c.readonly_inputs('bc'),{'/demo':('collect','/trial/dataset')})
        for stage in c.STUDENTS:
            self.assertEqual(c.readonly_inputs(stage),{'/demo':('collect','/trial/dataset'),'/bc':('bc','/trial/published')})
            args=c.command(stage,'python','source.py')
            self.assertEqual(args[args.index('--total-timesteps')+1],'5120000')
            self.assertEqual(args[args.index('--assist-bc-actor')+1],'/bc/final-actor')
        self.assertEqual(c.command('bc','python','source.py')[-1],'cuda')

    def test_collection_cpu_private_data_and_accounting_are_rejected(self):
        c.validate_collection(metadata(),native('collect'))
        for key,value in [('simulation_backend','physx_cpu'),('source_partition','development'),('purpose','local-cpu-dataflow-smoke'),
                          ('comparison_eligible',False),('public_training_seed_start',0),('environment_transitions',25601),
                          ('explicit_reset_calls',9),('teacher_prefix_dollars',0.),('successful_episodes',3),('incomplete_transition_rows',1)]:
            data=metadata();data[key]=value
            with self.assertRaises(ValueError,msg=key):c.validate_collection(data,native('collect'))
        data=metadata();data['versions']['sapien']='3.0.4.dev20261009+7dba23f1'
        with self.assertRaises(ValueError):c.validate_collection(data,native('collect'))
        data=metadata();data.update(status='budget-stopped-before-next-complete-batch',completed_episodes=64,successful_episodes=60,
            failed_completed_episodes=4,environment_transitions=3200,vector_step_calls=50,explicit_reset_calls=1,explicit_reset_episodes=64)
        c.validate_collection(data,native('collect'))
        data['status']='failed'
        with self.assertRaises(ValueError):c.validate_collection(data,native('collect'))

    def test_bc_random_init_provenance_and_episode_splits_are_required(self):
        c.validate_bc(bc_report(),metadata(),native('bc'))
        for key,value in [('comparison_eligible',False),('teacher_weights_loaded',True),('device','cpu'),('dataset_sha256','0'*64),
                          ('student_environment_transitions',50),('demonstration_minibatch_rows',1),('core_source_sha256','0'*64)]:
            report=bc_report();report[key]=value
            with self.assertRaises(ValueError,msg=key):c.validate_bc(report,metadata(),native('bc'))
        report=bc_report();report['validation_episode_indices'][0]=0
        with self.assertRaises(ValueError):c.validate_bc(report,metadata(),native('bc'))
        receipt=native('bc');receipt['gpu_actual_name']='CPU'
        with self.assertRaises(ValueError):c.validate_bc(bc_report(),metadata(),receipt)

    def test_shared_actor_and_exact_discriminator_accounting(self):
        for stage in c.STUDENTS:
            scalars,log,bindings,m,bc=student_fixture(stage);result=c.student_accounting(stage,scalars,log,bindings,m,bc)
            self.assertEqual(result['actual_steps'],51200);self.assertEqual(result['optimizer_step_count'],20)
            self.assertEqual(result['discriminator_updates'],32 if stage==c.STUDENTS[1] else 0)
            with self.assertRaises(ValueError):c.student_accounting(stage,scalars,log.replace('b'*64,'f'*64),bindings,m,bc)
        scalars,log,bindings,m,bc=student_fixture(c.STUDENTS[1]);scalars['audit/amp/discriminator_updates'][0]['value']=31
        with self.assertRaises(ValueError):c.student_accounting(c.STUDENTS[1],scalars,log,bindings,m,bc)
        scalars,log,bindings,m,bc=student_fixture(c.STUDENTS[0]);scalars['audit/amp/mean_style_reward']=[{'steps':51200,'value':.3}]
        with self.assertRaises(ValueError):c.student_accounting(c.STUDENTS[0],scalars,log,bindings,m,bc)

    def test_absolute_lease_admission_and_shared_budget_fail_closed(self):
        self.assertEqual(c.remaining_lease('collect',1120,1000,1050,10),(70.,60.))
        self.assertEqual(c.remaining_lease(c.STUDENTS[0],1600,1000,1400,10),(200.,190.))
        with self.assertRaises(RuntimeError):c.remaining_lease(c.STUDENTS[0],1600,1000,1580,10)
        with self.assertRaises(ValueError):c.remaining_lease('collect',1200,1000,1050,10)
        plan=json.loads((ROOT/'deploy/modal/amp-pilot-plan.json').read_bytes())
        blocked=dict(plan,launch_enabled=False,paid_launch_authorized=False,root_review=None)
        with self.assertRaises(PermissionError):c.validate_admission(blocked)
        admitted=copy.deepcopy(plan);admitted.update(launch_enabled=True,paid_launch_authorized=True,root_review={'fixture':True})
        c.validate_admission(admitted)
        admitted['training_reset_seeds'][0]=0
        with self.assertRaises(ValueError):c.validate_admission(admitted)
        ledger={'within_shared_cap':True,'user_total_cap_usd':'100','committed_worst_case_plus_pending_reservations_usd':'14.57',
                'reservations':[{'campaign':c.EXPERIMENT,'reserved_cap_usd':'3'}]}
        c.validate_budget(ledger)
        ledger['reservations']*=2
        with self.assertRaises(PermissionError):c.validate_budget(ledger)
        cost=c.conservative_cost();self.assertEqual(cost['gpu_container_seconds_bound'],2250)
        self.assertLess(cost['subtotal_usd'],1.44);self.assertGreater(cost['reserve_usd'],1.56)

if __name__=='__main__':unittest.main()
