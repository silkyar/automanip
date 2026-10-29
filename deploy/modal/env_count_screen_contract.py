"""Configuration-only PPO cadence pilot; fixed control payload and dynamic batch accounting."""
import hashlib
import json
import math
import re
from pathlib import Path

from precision_screen_contract import read_regular, audit_accounting as base_audit

EXPERIMENT='stackcube-env-count-pilot-20261028'
ARMS=('ppo-env512','ppo-env256')
SOURCE_METHOD='scratch-kl-stop-control'
SOURCE_SHA='de067415f6895f7b4fcc534cff20fe9cd628fef6faae3846c3b49687ff526632'
REVIEWED={arm:SOURCE_SHA for arm in ARMS}
ENV_COUNTS=dict(zip(ARMS,(512,256)))
PROTOCOL='fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
REFERENCE_PROVENANCE_SHA='da54212e66847688b2981ef6a29e9295dd9e555b969a386f6e3936e0a9b7af24'
REFERENCE_BINDING_SHA='0e35e9fb3e4f40a1231c1e24ffa02fd19242924d89011e8ea1d49ff364a7f53e'
BASE_SHA='cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082'
PATCH_SHA='b817a73a8c39d506ab8cf37970b64685488986e6e5a0ed62738f2b160568768a'
MAX_STEPS=10_240_000
SELECTED_STEPS=5_120_000

def batch_steps(method):return 50*ENV_COUNTS[method]
def selected(method):return {SELECTED_STEPS//batch_steps(method)+1:SELECTED_STEPS}
OUTCOME_RULE={'common_5120000_minimum_gain':3,'maximum_final_regression':3,'alternative_final_gain':3,'pareto_screen_only_unless_equal_steps':True,'censoring_reported':True,'interpolation':False,'automatic_retry':False,'automatic_expansion':False,'confirmation_queries':0}


def job(method):
    if method not in ARMS:raise ValueError('unreviewed KL/value method')
    return {'job_id':EXPERIMENT+'-'+method+'-seed-1','method':method,'source_method':SOURCE_METHOD,'num_envs':ENV_COUNTS[method],'batch_steps':batch_steps(method),'minibatch_size':batch_steps(method)//32,'seed':1,
            'trainer_sha256':REVIEWED[method],'gpu':'L40S','trainer_seconds_cap':600,
            'requested_steps':MAX_STEPS,'actual_steps_ceiling':MAX_STEPS}


def command(python,source,item):
    if item!=job(item['method']):raise ValueError('job contract changed')
    return [str(python),str(source),'--env-id','StackCube-v1','--seed','1',
            '--num-envs',str(item['num_envs']),'--num-eval-envs','8','--num-minibatches','32','--update-epochs','8',
            '--total-timesteps',str(MAX_STEPS),'--exp-name',item['job_id'],'--gamma','0.9',
            '--ent-coef','0.003','--learning-rate','0.0003','--target-kl','0.1','--no-clip-vloss',
            '--no-partial-reset','--no-capture-video','--no-track']


def remaining_lease(expires_at,now_epoch,now_monotonic):
    remaining=min(600.,expires_at-now_epoch)
    if not math.isfinite(remaining) or remaining<=20:raise RuntimeError('original training lease expired; no fresh allowance')
    return remaining,now_monotonic+remaining-20


def select_checkpoints(files,actual_steps,method):
    BATCH_STEPS=batch_steps(method)
    SELECTED=selected(method)
    if type(actual_steps) is not int or actual_steps<=0 or actual_steps>MAX_STEPS or actual_steps%BATCH_STEPS:
        raise ValueError('invalid completed transition count')
    for filename in files:
        match=re.fullmatch(r'ckpt_(\d+)\.pt',filename)
        if match and (int(match[1])-1)*BATCH_STEPS>actual_steps:
            raise ValueError('persisted checkpoint exceeds actual completed steps')
    result=[]
    for iteration,steps in SELECTED.items():
        filename='ckpt_'+str(iteration)+'.pt'
        if filename in files and steps>actual_steps:raise ValueError('checkpoint exceeds measured actual steps')
        if steps==actual_steps and 'final_ckpt.pt' in files:
            result.append({'steps':steps,'filename':'final_ckpt.pt','point':'prespecified-final-coincidence'})
        elif filename in files:result.append({'steps':steps,'filename':filename,'point':'prespecified'})
        elif steps<actual_steps:raise ValueError('expected cadence checkpoint missing')
    if 'final_ckpt.pt' not in files:raise ValueError('final checkpoint missing')
    if not any(r['steps']==actual_steps for r in result):
        result.append({'steps':actual_steps,'filename':'final_ckpt.pt','point':'final'})
    return result


# Per-rollout measures paired with their measured cumulative counters.
CUMULATIVE={'joint_actor_critic_optimizer_calls':'cumulative_joint_actor_critic_calls',
            'actor_optimizer_calls':'cumulative_actor_optimizer_calls',
            'critic_only_optimizer_calls':'cumulative_critic_only_optimizer_calls',
            'critic_optimizer_calls':'cumulative_critic_optimizer_calls',
            'optimizer_calls_total':'cumulative_optimizer_calls_total',
            'critic_only_training_rows':'cumulative_critic_only_training_rows'}
FIELDS=set(CUMULATIVE)|set(CUMULATIVE.values())|{'method','base_source_sha256','patch_source_sha256',
       'new_environment_transitions','configured_epochs','reached_epoch_count','kl_stopped',
       'kl_stop_epoch_0based','kl_stop_minibatch_0based','observed_kl','critic_only_seconds',
       'full_optimizer_loop_seconds','critic_only_deadline_skips','critic_only_mean_value_loss',
       'actor_updates_after_kl_stop','critic_only_actor_gradients_none',
       'target_refreshes_in_critic_only_phase','new_models_or_teacher_data'}


def audit_accounting(scalars,method,rollouts,log):
    if method not in ARMS:raise ValueError('unknown KL/value audit method')
    BATCH_STEPS=batch_steps(method);receipt_method=SOURCE_METHOD
    # Validate observed batch-size step indices before using the unchanged scalar validator.
    normalized={}
    expected=[(i+1)*BATCH_STEPS for i in range(rollouts)]
    for tag in ['audit/optimizer_steps','audit/training_entropy_coef','charts/learning_rate']+[f'audit/logstd_dim_{i}' for i in range(8)]:
        rows=scalars.get(tag,[])
        if [r['steps'] for r in rows]!=expected:raise ValueError('dynamic batch audit alignment')
        normalized[tag]=[dict(r,steps=(i+1)*51200) for i,r in enumerate(rows)]
    common=base_audit(normalized,'precision-constant',rollouts)
    legacy=common.pop('optimizer_step_count')
    messages=re.findall(r'^AUTOMANIP_KL_VALUE_UPDATES (.+)$',log,re.M)
    if len(messages)!=rollouts:raise ValueError('one KL/value receipt per completed rollout required')
    totals={key:0 for key in CUMULATIVE};numeric={};receipts=[]
    for index,message in enumerate(messages):
        row=json.loads(message);step=(index+1)*BATCH_STEPS
        if set(row)!=FIELDS or row['method']!=receipt_method or row['base_source_sha256']!=BASE_SHA or row['patch_source_sha256']!=PATCH_SHA:
            raise ValueError('KL/value source or receipt schema mismatch')
        for key in set(CUMULATIVE)|set(CUMULATIVE.values())|{'new_environment_transitions','configured_epochs','reached_epoch_count',
                'critic_only_deadline_skips','actor_updates_after_kl_stop','target_refreshes_in_critic_only_phase'}:
            if type(row[key]) is not int or row[key]<0:raise ValueError('noninteger/negative optimizer receipt count')
        if row['configured_epochs']!=8 or row['new_environment_transitions']!=step or not 1<=row['reached_epoch_count']<=8:raise ValueError('rollout/configuration mismatch')
        if row['actor_updates_after_kl_stop']!=0 or row['target_refreshes_in_critic_only_phase']!=0 or row['critic_only_actor_gradients_none'] is not True or row['new_models_or_teacher_data'] is not False:
            raise ValueError('actor/target/initialization isolation receipt mismatch')
        for key in ('observed_kl','critic_only_seconds','full_optimizer_loop_seconds'):
            if type(row[key]) not in (int,float) or not math.isfinite(row[key]):raise ValueError('nonfinite KL/time diagnostic')
        j=row['joint_actor_critic_optimizer_calls'];only=row['critic_only_optimizer_calls'];skips=row['critic_only_deadline_skips']
        if row['actor_optimizer_calls']!=j or row['critic_optimizer_calls']!=j+only or row['optimizer_calls_total']!=j+only or j+only>256:
            raise ValueError('optimizer role identities/bound mismatch')
        if row['critic_only_training_rows']!=only*(BATCH_STEPS//32):raise ValueError('critic-only minibatch row mismatch')
        if type(row['kl_stopped']) is not bool:raise ValueError('KL-stop flag must be boolean')
        if row['kl_stopped']:
            epoch=row['kl_stop_epoch_0based'];batch=row['kl_stop_minibatch_0based']
            if type(epoch) is not int or not 0<=epoch<8 or type(batch) is not int or not 0<=batch<32 or j!=epoch*32+batch or row['observed_kl']<=.1:
                raise ValueError('first KL-stop indices/count mismatch')
            if receipt_method==SOURCE_METHOD:
                if only or skips or row['reached_epoch_count']!=epoch+1:raise ValueError('control continued after KL stop')
            elif row['reached_epoch_count']!=8 or j+only+skips!=256:raise ValueError('candidate did not account for configured suffix')
        elif row['kl_stop_epoch_0based'] is not None or row['kl_stop_minibatch_0based'] is not None or only or skips or j!=256 or row['reached_epoch_count']!=8:
            raise ValueError('no-KL behavior differs from ordinary eight-epoch PPO')
        if receipt_method==SOURCE_METHOD and (only or row['critic_only_training_rows'] or row['critic_only_seconds']):raise ValueError('control has critic-only work')
        if row['full_optimizer_loop_seconds']<0 or not 0<=row['critic_only_seconds']<=row['full_optimizer_loop_seconds']+1e-5:raise ValueError('critic-only/full loop timing mismatch')
        loss=row['critic_only_mean_value_loss']
        if only:
            if type(loss) not in (int,float) or not math.isfinite(loss) or loss<0:raise ValueError('invalid critic-only loss')
        elif loss is not None:raise ValueError('loss exists without critic-only updates')
        for key,cumulative in CUMULATIVE.items():
            totals[key]+=row[key]
            if row[cumulative]!=totals[key]:raise ValueError('cumulative optimizer identity mismatch')
        if row['cumulative_joint_actor_critic_calls']!=scalars['audit/optimizer_steps'][index]['value']:raise ValueError('legacy joint counter mismatch')
        for key,value in row.items():
            if type(value) in (int,float):numeric.setdefault('audit/kl_value/'+key,[]).append({'steps':step,'value':value})
        receipts.append(row)
    actual_tags={key for key in scalars if key.startswith('audit/kl_value/')}
    if actual_tags!=set(numeric):raise ValueError('KL/value scalar tag coverage mismatch')
    for tag,expected in numeric.items():
        rows=scalars[tag]
        if len(rows)!=len(expected):raise ValueError('KL/value scalar row count mismatch')
        for actual,wanted in zip(rows,expected):
            value=actual['value'];v=wanted['value']
            if actual['steps']!=wanted['steps'] or value is None or not math.isfinite(value) or (value!=v if type(v) is int else not math.isclose(value,v,rel_tol=1e-5,abs_tol=1e-7)):
                raise ValueError('KL/value JSON/scalar mismatch')
    if totals['joint_actor_critic_optimizer_calls']!=legacy:raise ValueError('legacy total mismatch')
    common.update(optimizer_step_count=totals['optimizer_calls_total'],legacy_joint_optimizer_step_count=legacy,
                  measured_actor_optimizer_calls=totals['actor_optimizer_calls'],measured_critic_optimizer_calls=totals['critic_optimizer_calls'],
                  measured_joint_optimizer_calls=totals['joint_actor_critic_optimizer_calls'],measured_critic_only_optimizer_calls=totals['critic_only_optimizer_calls'],
                  critic_only_training_rows=totals['critic_only_training_rows'],critic_only_seconds=sum(r['critic_only_seconds'] for r in receipts),
                  full_optimizer_loop_seconds=sum(r['full_optimizer_loop_seconds'] for r in receipts),kl_stop_rollouts=sum(r['kl_stopped'] for r in receipts),
                  private_rng_evidence='Pinned reviewed source and CPU actor/Adam/NumPy/Torch parity tests; no untrusted source execution in scorer.')
    return common


def conservative_cost():
    seconds=2*(660+30)+4*(120+30)
    gpu=seconds*(.000542+4*.0000131+16*.00000222)
    conversion=4*120*(.0000131+2*.00000222);coordinator=3600*(.125*.0000131+.5*.00000222)
    subtotal=gpu+conversion+coordinator
    return {'gpu_container_seconds_bound':seconds,'gpu_cpu_ram_usd':gpu,'conversion_cpu_usd':conversion,
            'coordinator_usd':coordinator,'subtotal_usd':subtotal,'pair_cap_usd':3.,'reserve_usd':3.-subtotal}


def validate_admission(plan):
    if not plan.get('launch_enabled') or not plan.get('paid_launch_authorized') or not plan.get('root_review'):
        raise PermissionError('separate reviewed environment-count pilot admission required')
    expected={'experiment_id':EXPERIMENT,'jobs':[job(m) for m in ARMS],'budget_cap_usd':3.,'cost':conservative_cost(),
              'protocol_sha256':'fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764',
              'selected_steps':[5_120_000,'final_actual_steps'],'reference_binding_sha256':REFERENCE_BINDING_SHA,'reference_provenance_sha256':REFERENCE_PROVENANCE_SHA,'max_development_evaluations':4,
              'development_episodes':128,'confirmation_queries':0,'training_seeds':[1],'outcome_rule':OUTCOME_RULE}
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('exact environment-count pilot plan required')


def validate_budget(ledger):
    from decimal import Decimal
    if not ledger.get('within_shared_cap') or Decimal(str(ledger.get('user_total_cap_usd',0)))!=100 or Decimal(str(ledger.get('committed_worst_case_plus_pending_reservations_usd',101)))>100:
        raise PermissionError('shared Modal100 cap must be preserved')
    entries=[r for r in ledger.get('reservations',[]) if r.get('campaign')==EXPERIMENT]
    if len(entries)!=1 or Decimal(str(entries[0]['reserved_cap_usd']))!=3:raise PermissionError('one shared-ledger3 reservation required')


def existing_report(durable,digest):
    """Validate reusable global report before considering any GPU dispatch."""
    if durable is None:return None
    report=durable.get('report',{})
    if durable.get('state')!='complete' or durable.get('protocol_sha256')!=PROTOCOL or durable.get('weights_sha256')!=digest or durable.get('split')!='development':raise ValueError('global durable report identity')
    if report.get('protocol_sha256')!=PROTOCOL or report.get('weights_sha256')!=digest or report.get('split')!='development' or report.get('episode_count')!=128 or report.get('failed_episodes')!=0 or report.get('smoke') is not False:raise ValueError('global durable report validity')
    return dict(report=report,durable_path='frozen-evaluation-attempts/'+PROTOCOL+'/'+digest+'/development/result.json',evaluation_seconds=durable['evaluation_seconds'],gpu_actual_name=durable['gpu_actual_name'],queue_seconds=durable['queue_seconds'],reused_global_report=True)
