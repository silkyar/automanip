"""Fixed admission and accounting for one demonstration-assisted AMP pilot."""
import hashlib
import json
import math
from pathlib import Path
import re

from precision_screen_contract import read_regular, audit_accounting as ppo_audit

EXPERIMENT='stackcube-amp-pilot-20261028'
STUDENTS=('assist-bc-ppo','assist-bc-ppo-amp')
STAGES=('collect','bc',*STUDENTS)
PROTOCOL='fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
TASK='b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f'
PINS={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'}
CORE_SHA='463303be5809392e7ea9597eb47c0e45ccd256438750d4a0b219c2abd9dee5bd'
SOURCES={
    'collect':('research/discovery/amp_collect_v1.py','523459ce91a59200970d66a7dd83fa8dc06c7c96879e72ab541f2f263403ef3b'),
    'bc':('research/discovery/amp_bc_v1.py','1e8b1a6ea1e2007bf43adb24fc0d96008c54d01d3af5d6650548c35f6392f7f6'),
    'assist-bc-ppo':('research/discovery/amp-prior-assisted-v1/assist-bc-ppo.py','7a4de8deecf0adc15d9f8beaef1bf6b9874ca9eb7b0122efbab5d9b1bc80d427'),
    'assist-bc-ppo-amp':('research/discovery/amp-prior-assisted-v1/assist-bc-ppo-amp.py','c2af05677a3a3652b0bd6b27f482a4946a51cad72b0ce0106e6b2b7b90f457ad'),
}
TEACHER_SHA='ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16'
PREFIX={'training_steps':49_971_200,'trainer_seconds':2295.408300588,'training_seed':1,
        'actor_weights_sha256':TEACHER_SHA,'dollars':None,
        'dollar_attribution':'Teacher belongs to completed three-arm quality app; separate exact per-teacher dollar allocation unavailable, not assumed free.'}
# Fresh public TRAINING reset seeds; each seed resets one batch of64 environments.
# Chosen before launch without reading benchmark episodes or private seed files.
TRAINING_SEED_START=90_280_000
RESET_SEEDS=tuple(range(TRAINING_SEED_START,TRAINING_SEED_START+8))
EPISODES=512
NUM_DEMO_ENVS=64
MAX_RL_STEPS=5_120_000
BATCH_STEPS=51_200
OUTCOME_RULE={'matching_new_rl_steps_required':MAX_RL_STEPS,'minimum_amp_paired_success_gain':3,
              'amp_must_not_regress_vs_bc_only':True,'censored_runs_no_sample_efficiency_claim':True,
              'automatic_expansion':False,'confirmation_queries':0,
              'promotion_requires_fresh_training_seeds_and_independent_confirmation':True}
INPUT_LIMITS={'dataset.json':64*1024,'trajectories.npz':16*1024*1024,
              'bc-report.json':2*1024*1024,'final-actor/manifest.json':8192,
              'final-actor/weights.json':32*1024*1024}


def job(stage):
    if stage not in STAGES:raise ValueError('unknown AMP stage')
    return {'stage':stage,'job_id':EXPERIMENT+'-'+stage+'-seed-1','seed':1,
            'source_path':SOURCES[stage][0],'source_sha256':SOURCES[stage][1],
            'lease_seconds':600 if stage in STUDENTS else 120,
            'wrapper_seconds':660 if stage in STUDENTS else 180,
            'requested_rl_steps':MAX_RL_STEPS if stage in STUDENTS else 0,'gpu':'L40S'}


def volume_name(stage):
    job(stage)
    return 'automanip-modal-amp-'+stage+'-20261028'


def readonly_inputs(stage):
    job(stage)
    if stage=='collect':return {}
    result={'/demo':('collect','/trial/dataset')}
    if stage in STUDENTS:result['/bc']=('bc','/trial/published')
    return result


def command(stage,python,source):
    item=job(stage)
    if stage=='collect':
        return [python,source,'--teacher-actor','/trainer-output/teacher','--expected-teacher-sha256',TEACHER_SHA,
                '--output','/trainer-output/run','--seed-start',str(TRAINING_SEED_START),
                '--episodes',str(EPISODES),'--num-envs',str(NUM_DEMO_ENVS),
                '--teacher-prefix-transitions',str(PREFIX['training_steps']),
                '--teacher-prefix-trainer-seconds',str(PREFIX['trainer_seconds']),'--max-seconds','120']
    if stage=='bc':
        return [python,source,'--dataset','/demo','--output','/trainer-output/run','--seed','1',
                '--updates','5000','--batch-size','1024','--max-seconds','120','--device','cuda']
    return [python,source,'--env-id','StackCube-v1','--seed','1','--num-envs','1024','--num-eval-envs','8',
            '--num-minibatches','32','--update-epochs','8','--total-timesteps',str(MAX_RL_STEPS),
            '--exp-name',item['job_id'],'--gamma','0.9','--ent-coef','0.003','--learning-rate','0.0003',
            '--no-partial-reset','--no-capture-video','--no-track','--assist-dataset','/demo',
            '--assist-bc-actor','/bc/final-actor','--assist-bc-report','/bc/bc-report.json']


def remaining_lease(stage,expires_at,dispatched_at,now_epoch,now_monotonic):
    cap=job(stage)['lease_seconds']
    if not cap-.1<=expires_at-dispatched_at<=cap+.1:raise ValueError('fixed original stage lease required')
    remaining=min(float(cap),expires_at-now_epoch)
    if not math.isfinite(remaining) or remaining<=20:raise RuntimeError('original stage lease expired; no refresh')
    return remaining,now_monotonic+remaining-20


def verify_sources(program,stage,core):
    if hashlib.sha256(program).hexdigest()!=job(stage)['source_sha256'] or hashlib.sha256(core).hexdigest()!=CORE_SHA:
        raise ValueError('reviewed AMP source changed')


def verify_runtime(record):
    if record.get('state')!='complete' or record.get('versions')!=PINS or record.get('gpu_actual_name')!='NVIDIA L40S':
        raise ValueError('production GPU runtime receipt required')


def validate_collection(metadata,record):
    verify_runtime(record)
    if metadata.get('purpose')=='local-cpu-dataflow-smoke' or metadata.get('comparison_eligible') is False:raise ValueError('CPU smoke data is never production eligible')
    expected={'format':'automanip-public-training-demos-v1','env_id':'StackCube-v1','observation_dim':48,'action_dim':8,
              'horizon':50,'control_mode':'pd_joint_delta_pos','obs_mode':'state','reward_mode':'normalized_dense',
              'simulation_backend':'physx_cuda','source_partition':'public-training','versions':PINS,
              'teacher_actor_weights_sha256':TEACHER_SHA,'teacher_prefix_transitions':PREFIX['training_steps'],
              'teacher_prefix_trainer_seconds':PREFIX['trainer_seconds'],'teacher_prefix_dollars':None,
              'collector_source_sha256':SOURCES['collect'][1],'public_training_seed_start':TRAINING_SEED_START,
              'num_envs':NUM_DEMO_ENVS,'requested_episodes':EPISODES}
    if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('production demonstration provenance mismatch')
    if record.get('source_sha256')!=SOURCES['collect'][1] or record.get('core_source_sha256')!=CORE_SHA:
        raise ValueError('collector source receipt mismatch')
    completed=metadata.get('completed_episodes');success=metadata.get('successful_episodes')
    if type(completed) is not int or completed<=0 or completed>EPISODES or completed%NUM_DEMO_ENVS:
        raise ValueError('invalid complete demonstration count')
    if type(success) is not int or not 4<=success<=completed:raise ValueError('insufficient successful training demonstrations')
    if metadata.get('status') not in ('completed','budget-stopped-before-next-complete-batch') or metadata.get('error') is not None:
        raise ValueError('failed demonstration collection is ineligible')
    if metadata['status']=='completed' and completed!=EPISODES:raise ValueError('incomplete uncensored collection')
    expected_counts={'failed_completed_episodes':completed-success,'environment_transitions':completed*50,
                     'vector_step_calls':completed*50//NUM_DEMO_ENVS,'incomplete_transition_rows':0,
                     'explicit_reset_calls':completed//NUM_DEMO_ENVS,'explicit_reset_episodes':completed}
    if any(type(metadata.get(k)) is not int or metadata[k]!=v for k,v in expected_counts.items()):
        raise ValueError('collection simulator/reset accounting mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}',metadata.get('sha256','')):raise ValueError('dataset hash missing')
    return metadata


def validate_bc(report,metadata,record):
    verify_runtime(record)
    if report.get('teacher_weights_loaded') is not False or report.get('comparison_eligible') is not True:raise ValueError('BC production eligibility flags must be explicit booleans')
    expected={'teacher_weights_loaded':False,'student_initialization':'random','comparison_eligible':True,'device':'cuda',
              'seed':1,'dataset_sha256':metadata['sha256'],'bc_source_sha256':SOURCES['bc'][1],'core_source_sha256':CORE_SHA,
              'training_torch_version':PINS['torch'],'training_numpy_version':PINS['numpy'],
              'teacher_prefix_transitions':PREFIX['training_steps'],'teacher_prefix_trainer_seconds':PREFIX['trainer_seconds'],
              'teacher_prefix_dollars':None,'teacher_generation_transitions':metadata['environment_transitions'],
              'student_environment_transitions':0,'total_dataset_episodes':metadata['completed_episodes'],
              'failed_teacher_episodes_preserved':metadata['failed_completed_episodes']}
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('BC production/random-student provenance mismatch')
    if record.get('source_sha256')!=SOURCES['bc'][1] or record.get('core_source_sha256')!=CORE_SHA:raise ValueError('BC runtime source mismatch')
    count=report.get('optimizer_updates')
    if type(count) is not int or not 1<=count<=5000 or report.get('demonstration_minibatch_rows')!=count*1024:
        raise ValueError('invalid BC optimizer/minibatch accounting')
    train=report.get('train_episode_indices',[]);validation=report.get('validation_episode_indices',[])
    if any(type(v) is not int or not 0<=v<metadata['completed_episodes'] for v in train+validation):raise ValueError('invalid demo split index')
    if not train or not validation or len(set(train+validation))!=len(train+validation):raise ValueError('overlapping or empty episode split')
    if len(train)+len(validation)!=metadata['successful_episodes'] or len(validation)!=max(1,metadata['successful_episodes']//10):raise ValueError('success episode split mismatch')
    if report.get('training_demo_episodes')!=len(train) or report.get('validation_demo_episodes')!=len(validation):raise ValueError('BC split accounting mismatch')
    if not math.isfinite(report.get('validation_action_mse',float('nan'))) or report['validation_action_mse']<0:raise ValueError('invalid BC diagnostic')
    return report


def student_accounting(stage,scalars,log,bindings,metadata,bc_report):
    if stage not in STUDENTS:raise ValueError('unknown student')
    rows=scalars.get('time/step',[]);rollouts=len(rows);actual=rollouts*BATCH_STEPS
    if not rows or not 0<actual<=MAX_RL_STEPS or rows[-1]['value']!=actual:raise ValueError('invalid student transition count')
    audit=ppo_audit(scalars,'precision-constant',rollouts)
    messages=re.findall(r'^AUTOMANIP_ASSISTANCE (.+)$',log,re.M)
    if len(messages)!=1:raise ValueError('single assistance provenance row required')
    assistance=json.loads(messages[0])
    expected={'teacher_weights_loaded':False,'dataset_sha256':metadata['sha256'],
              'bc_report_sha256':bindings['bc']['bc-report.json'],'bc_student_weights_sha256':bc_report['final']['weights_sha256'],
              'bc_prefix_optimizer_updates':bc_report['optimizer_updates'],'bc_prefix_demonstration_minibatch_rows':bc_report['demonstration_minibatch_rows'],
              'teacher_prefix_transitions':PREFIX['training_steps'],'teacher_prefix_trainer_seconds':PREFIX['trainer_seconds'],
              'teacher_prefix_dollars':None,'teacher_generation_transitions':metadata['environment_transitions'],
              'all_teacher_episodes':metadata['completed_episodes'],'failed_teacher_episodes_preserved':metadata['failed_completed_episodes'],
              'amp_enabled':stage==STUDENTS[1],'amp_style_weight':.25 if stage==STUDENTS[1] else 0.,'task_reward_weight':1.,
              'discriminator_models_in_agent_serialization':False}
    if any(assistance.get(k)!=v for k,v in expected.items()):raise ValueError('student assistance provenance mismatch')
    amp={k:v for k,v in scalars.items() if k.startswith('audit/amp/')}
    if stage==STUDENTS[0] and amp:raise ValueError('control unexpectedly used motion prior')
    if stage==STUDENTS[1]:
        required=('discriminator_updates','discriminator_sample_rows','learner_transition_rows','replay_rows',
                  'discriminator_update_seconds','mean_style_reward','loss','least_squares','gradient_penalty','expert_score','policy_score')
        steps=[(i+1)*BATCH_STEPS for i in range(rollouts)]
        for tag in required:
            trace=amp.get('audit/amp/'+tag,[])
            if len(trace)!=rollouts or [r['steps'] for r in trace]!=steps or any(r['value'] is None or not math.isfinite(r['value']) for r in trace):raise ValueError('missing/nonfinite motion-prior trace '+tag)
        for i,step in enumerate(steps):
            fixed={'discriminator_updates':32*(i+1),'discriminator_sample_rows':32*(i+1)*512,
                   'learner_transition_rows':step,'replay_rows':min(100000,step)}
            if any(amp['audit/amp/'+k][i]['value']!=v for k,v in fixed.items()):raise ValueError('discriminator accounting mismatch')
            if not 0<=amp['audit/amp/mean_style_reward'][i]['value']<=1 or amp['audit/amp/discriminator_update_seconds'][i]['value']<0:raise ValueError('invalid motion-prior diagnostic')
        audit.update(discriminator_updates=32*rollouts,discriminator_minibatch_rows=32*rollouts*512,
                     discriminator_update_seconds=sum(r['value'] for r in amp['audit/amp/discriminator_update_seconds']))
    else:audit.update(discriminator_updates=0,discriminator_minibatch_rows=0,discriminator_update_seconds=0.)
    audit.update(actual_steps=actual,completed_rollouts=rollouts,assistance=assistance,stopped_early=actual<MAX_RL_STEPS)
    return audit


def conservative_cost():
    seconds=2*(180+30)+2*(660+30)+3*(120+30)
    gpu=seconds*(.000542+4*.0000131+16*.00000222)
    conversion=2*120*(.0000131+2*.00000222);coordinator=3600*(.125*.0000131+.5*.00000222)
    subtotal=gpu+conversion+coordinator
    return {'gpu_container_seconds_bound':seconds,'gpu_cpu_ram_usd':gpu,'conversion_cpu_usd':conversion,
            'coordinator_usd':coordinator,'subtotal_usd':subtotal,'pilot_cap_usd':3.,'reserve_usd':3.-subtotal}


def validate_admission(plan):
    if not plan.get('launch_enabled') or not plan.get('paid_launch_authorized') or not plan.get('root_review'):
        raise PermissionError('AMP pilot requires separate root reviewed paid admission')
    expected={'experiment_id':EXPERIMENT,'jobs':[job(stage) for stage in STAGES],'training_reset_seeds':list(RESET_SEEDS),
              'teacher_prefix':PREFIX,'protocol_sha256':PROTOCOL,'budget_cap_usd':3.,'cost':conservative_cost(),
              'confirmation_queries':0,'development_episodes':128,'maximum_unique_development_evaluations':3,'outcome_rule':OUTCOME_RULE}
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('exact frozen AMP admission required')
    return expected['jobs']


def validate_budget(ledger):
    from decimal import Decimal
    if not ledger.get('within_shared_cap') or Decimal(str(ledger.get('user_total_cap_usd',0)))!=100 or Decimal(str(ledger.get('committed_worst_case_plus_pending_reservations_usd',101)))>100:
        raise PermissionError('shared Modal100 ledger must be within cap')
    reservations=[r for r in ledger.get('reservations',[]) if r.get('campaign')==EXPERIMENT]
    if len(reservations)!=1 or Decimal(str(reservations[0]['reserved_cap_usd']))!=3:raise PermissionError('AMP needs one shared-ledger3 reservation')
