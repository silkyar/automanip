"""Narrow contracts for the three reviewed final-quality PPO jobs; standard library only."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

ARMS=('precision-constant','precision-entropy-anneal','precision-lr-anneal')
REVIEWED={'precision-constant': 'cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082', 'precision-entropy-anneal': '4618b428c35838417b95996a4245530f3b24fd5011b674ef9666a1ae6319c761', 'precision-lr-anneal': '878c99d186f2bd554d36999e437302c52c768204246c5d4c35ecb68a34116a24'}

EXPERIMENT='stackcube-precision-screen-20261027'
MAX_STEPS=49_971_200
BATCH_STEPS=51_200
SELECTED={501:25_600_000,751:38_400_000}


def job(method):
    if method not in ARMS or method not in REVIEWED or not re.fullmatch(r'[0-9a-f]{64}',REVIEWED[method]):raise ValueError('unreviewed method')
    return {'job_id':EXPERIMENT+'-'+method+'-seed-1','method':method,'seed':1,
            'trainer_sha256':REVIEWED[method],'gpu':'L40S','trainer_seconds_cap':2700,
            'requested_steps':50_000_000,'actual_steps_ceiling':MAX_STEPS}


def command(python,source,item):
    if item!=job(item['method']):raise ValueError('job contract changed')
    return [str(python),str(source),'--env-id','StackCube-v1','--seed','1',
            '--num-envs','1024','--num-eval-envs','8','--num-minibatches','32','--update-epochs','8',
            '--total-timesteps','50000000','--exp-name',item['job_id'],'--gamma','0.9',
            '--ent-coef','0.003','--learning-rate','0.0003','--no-partial-reset','--no-capture-video','--no-track']


def remaining_lease(expires_at,now_epoch,now_monotonic):
    remaining=min(2700.,expires_at-now_epoch)
    if remaining<=20:raise RuntimeError('original training lease expired; no fresh allowance')
    return remaining,now_monotonic+remaining-20


def read_regular(root,relative,limit):
    """Read one bounded regular file through directory descriptors; no symlink traversal."""
    parts=Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in ('.','..') for p in parts):raise ValueError('unsafe relative path')
    flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
    directory=os.open(root,flags|getattr(os,'O_DIRECTORY',0))
    try:
        for part in parts[:-1]:
            child=os.open(part,flags|getattr(os,'O_DIRECTORY',0),dir_fd=directory)
            os.close(directory);directory=child
        fd=os.open(parts[-1],flags|os.O_NONBLOCK,dir_fd=directory)
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size>limit:raise ValueError('bounded regular file required')
            with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(limit+1)
            after=os.fstat(fd)
            if len(raw)>limit or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
                raise ValueError('oversized or changing artifact')
            return raw
        finally:os.close(fd)
    finally:os.close(directory)


def select_checkpoints(files,actual_steps):
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


def conservative_cost():
    # Three fixed trainer wrappers, nine frozen GPU evaluations, startup reserves.
    gpu_seconds=3*(2760+30)+9*(120+30)
    gpu_rate=.000542+4*.0000131+16*.00000222
    conversions=9*120*(.0000131+2*.00000222)
    coordinator=7200*(.125*.0000131+.5*.00000222)
    subtotal=gpu_seconds*gpu_rate+conversions+coordinator
    return {'gpu_container_seconds_bound':gpu_seconds,'gpu_cpu_ram_usd':gpu_seconds*gpu_rate,
            'conversion_cpu_usd':conversions,'coordinator_usd':coordinator,
            'subtotal_usd':subtotal,'batch_cap_usd':7.,'reserve_usd':7.-subtotal}


def audit_accounting(scalars,method,rollouts):
    """Require complete measured optimizer/std/schedule traces; never infer steps."""
    if method not in ARMS or type(rollouts) is not int or rollouts<=0:raise ValueError('invalid audit request')
    expected_steps=[(index+1)*BATCH_STEPS for index in range(rollouts)]
    tags=['audit/optimizer_steps','audit/training_entropy_coef','charts/learning_rate']+[f'audit/logstd_dim_{index}' for index in range(8)]
    for tag in tags:
        rows=scalars.get(tag,[])
        if len(rows)!=rollouts or [r['steps'] for r in rows]!=expected_steps:raise ValueError('missing or misaligned audit '+tag)
        if any(r['value'] is None or not math.isfinite(r['value']) for r in rows):raise ValueError('nonfinite audit '+tag)
    previous=0
    for index,step in enumerate(expected_steps):
        count=scalars['audit/optimizer_steps'][index]['value']
        if count!=int(count) or not 1<=count-previous<=8*32:raise ValueError('invalid exact optimizer accounting')
        previous=int(count)
        fraction=max(0.,min(1.,(step-25_600_000)/(MAX_STEPS-25_600_000)))
        entropy=.003*(1-fraction) if method=='precision-entropy-anneal' else .003
        learning_rate=.0003*(1-.9*fraction) if method=='precision-lr-anneal' else .0003
        if not math.isclose(scalars['audit/training_entropy_coef'][index]['value'],entropy,rel_tol=1e-5,abs_tol=1e-9):raise ValueError('entropy schedule audit mismatch')
        if not math.isclose(scalars['charts/learning_rate'][index]['value'],learning_rate,rel_tol=1e-5,abs_tol=1e-10):raise ValueError('learning rate schedule audit mismatch')
    return {'optimizer_step_count':previous,'final_logstd_per_dimension':[scalars[f'audit/logstd_dim_{index}'][-1]['value'] for index in range(8)],'audit_trace_rows':rollouts}
