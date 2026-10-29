"""Bounded, sequential three-arm continuation pilot; root admission pending."""
import hashlib
import math
from pathlib import Path
import re

from precision_screen_contract import read_regular, audit_accounting as precision_audit_accounting

ARMS=('risk-ppo-control','risk-episode-mean','risk-episode-cvar10')
REVIEWED={'risk-ppo-control': '5f802501efb026560774d795f16bacb0b8fc7767a17822f5c191b1431fc7988b', 'risk-episode-mean': '3c5fbf9334a16a1cdb584000ff74c49e05f645e9177488f079ca2e917142c2f0', 'risk-episode-cvar10': 'ebf5469003849787e5d8469a48a76f94eb232f0554b8c8f0a07e0c5b85f3495c'}
EXPERIMENT='stackcube-risk-continuation-20261028'
MAX_STEPS=2_048_000
BATCH_STEPS=51_200
REFERENCE_SHA256='307a2bc5996d5def9f696ab35b27a9cf96a5cf7401c678ecc981b78170d1084b'
REFERENCE_BYTES=1_168_418
REFERENCE_ACTOR_SHA256='39ba198d0c98af14a3d2965ef9a3be859f31aa3c4889db3da311da910c2da153'
PREFIX={'training_steps':24_985_600,'trainer_seconds':1088.7667,'training_seed':1,
        'native_checkpoint_sha256':REFERENCE_SHA256,'actor_weights_sha256':REFERENCE_ACTOR_SHA256,
        'optimizer_state_restarted':True,'initialization':'continuation from own confirmed PPO policy; not fresh random initialization'}


def job(method):
    if method not in ARMS or not re.fullmatch(r'[0-9a-f]{64}',REVIEWED.get(method,'')):raise ValueError('unreviewed continuation method')
    return {'job_id':EXPERIMENT+'-'+method+'-seed-1','method':method,'seed':1,
            'trainer_sha256':REVIEWED[method],'reference_sha256':REFERENCE_SHA256,
            'gpu':'L40S','trainer_seconds_cap':300,'requested_steps':MAX_STEPS,'actual_steps_ceiling':MAX_STEPS}


def validate_inputs(program,reference,item):
    if item!=job(item['method']) or hashlib.sha256(program).hexdigest()!=item['trainer_sha256']:raise ValueError('unreviewed trainer input')
    if len(reference)!=REFERENCE_BYTES or hashlib.sha256(reference).hexdigest()!=REFERENCE_SHA256:raise ValueError('reference checkpoint provenance mismatch')
    if len(program)+len(reference)>=2*1024*1024:raise ValueError('combined program/checkpoint input must stay below2MiB')


def command(python,source,item):
    if item!=job(item['method']):raise ValueError('continuation job contract changed')
    return [str(python),str(source),'--env-id','StackCube-v1','--seed','1',
            '--num-envs','1024','--num-eval-envs','8','--num-minibatches','32','--update-epochs','8',
            '--total-timesteps',str(MAX_STEPS),'--exp-name',item['job_id'],'--gamma','0.9',
            '--ent-coef','0.003','--learning-rate','0.0003','--no-partial-reset','--no-capture-video','--no-track',
            '--checkpoint','/trainer-output/reference.pt']


def remaining_lease(expires_at,now_epoch,now_monotonic):
    remaining=min(300.,expires_at-now_epoch)
    if remaining<=20:raise RuntimeError('original continuation lease expired; no fresh allowance')
    return remaining,now_monotonic+remaining-20


def select_checkpoints(files,actual_steps):
    if type(actual_steps) is not int or not 0<actual_steps<=MAX_STEPS or actual_steps%BATCH_STEPS:raise ValueError('invalid continuation step count')
    for filename in files:
        match=re.fullmatch(r'ckpt_(\d+)\.pt',filename)
        if match and (int(match[1])-1)*BATCH_STEPS>actual_steps:raise ValueError('future checkpoint')
    if 'final_ckpt.pt' not in files:raise ValueError('final checkpoint missing')
    return [{'steps':actual_steps,'filename':'final_ckpt.pt','point':'final-continuation'}]


def conservative_cost():
    seconds=3*(330+30)+3*(120+30)
    gpu=seconds*(.000542+4*.0000131+16*.00000222)
    conversions=3*120*(.0000131+2*.00000222)
    coordinator=1800*(.125*.0000131+.5*.00000222)
    subtotal=gpu+conversions+coordinator
    return {'gpu_container_seconds_bound':seconds,'gpu_cpu_ram_usd':gpu,'conversion_cpu_usd':conversions,
            'coordinator_usd':coordinator,'subtotal_usd':subtotal,'pilot_cap_usd':1.,'reserve_usd':1.-subtotal}


def audit_accounting(scalars,method,rollouts):
    if method not in ARMS:raise ValueError('unknown risk audit method')
    result=precision_audit_accounting(scalars,'precision-constant',rollouts)
    required=('actor_alpha','actor_var_baseline','actor_raw_weight_rms','mean_discounted_return','training_failures','score_weight_rms')
    expected_steps=[(i+1)*BATCH_STEPS for i in range(rollouts)]
    for name in required:
        rows=scalars.get('audit/risk/'+name,[])
        if len(rows)!=rollouts or [r['steps'] for r in rows]!=expected_steps:raise ValueError('missing or misaligned risk trace '+name)
        if any(r['value'] is None or not math.isfinite(r['value']) for r in rows):raise ValueError('nonfinite required risk trace '+name)
    alpha=.1 if method=='risk-episode-cvar10' else 1.
    if any(not math.isclose(r['value'],alpha,rel_tol=1e-5,abs_tol=1e-7) for r in scalars['audit/risk/actor_alpha']):raise ValueError('wrong actor alpha')
    if any(r['value']!=int(r['value']) or not 0<=r['value']<=1024 for r in scalars['audit/risk/training_failures']):raise ValueError('invalid training failure diagnostic')
    # Tail failure/capture/enrichment diagnostics are optional when no failures.
    result.update(actor_objective=method,actor_alpha=alpha,training_success_labels_diagnostic_only=True)
    return result
