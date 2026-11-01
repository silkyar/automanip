"""Narrow contracts for the two reviewed CAPG screen jobs; standard library only."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

REVIEWED={
    'full-horizon-control':'68e09965db1378616f380eb763862149d8c28232075e06741426caba3dfb5064',
    'full-horizon-capg':'8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971',
}
EXPERIMENT='stackcube-capg-screen-20261027'
MAX_STEPS=24_985_600
BATCH_STEPS=51_200
SELECTED={101:5_120_000,201:10_240_000,301:15_360_000,401:20_480_000}


def job(method):
    if method not in REVIEWED:raise ValueError('unreviewed method')
    return {'job_id':EXPERIMENT+'-'+method+'-seed-1','method':method,'seed':1,
            'trainer_sha256':REVIEWED[method],'gpu':'L40S','trainer_seconds_cap':1320,
            'requested_steps':25_000_000,'actual_steps_ceiling':MAX_STEPS}


def command(python,source,item):
    if item!=job(item['method']):raise ValueError('job contract changed')
    return [str(python),str(source),'--env-id','StackCube-v1','--seed','1',
            '--num-envs','1024','--num-eval-envs','8','--num-minibatches','32','--update-epochs','8',
            '--total-timesteps','25000000','--exp-name',item['job_id'],'--gamma','0.9',
            '--ent-coef','0.003','--learning-rate','0.0003','--no-partial-reset','--no-capture-video','--no-track']


def remaining_lease(expires_at,now_epoch,now_monotonic):
    remaining=min(1320.,expires_at-now_epoch)
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
    # Two 1380s trainer wrappers and ten 120s evaluator wrappers; each has 30s
    # startup reserve. CPU uses its 4-core ceiling and all 16 GiB throughout.
    gpu_seconds=2*(1380+30)+10*(120+30)
    gpu_rate=.000542+4*.0000131+16*.00000222
    conversions=10*120*(.0000131+2*.00000222)
    coordinator=5400*(.125*.0000131+.5*.00000222)
    return {'gpu_container_seconds_bound':gpu_seconds,'gpu_cpu_ram_usd':gpu_seconds*gpu_rate,
            'conversion_cpu_usd':conversions,'coordinator_usd':coordinator,
            'subtotal_usd':gpu_seconds*gpu_rate+conversions+coordinator,
            'pair_cap_usd':3.,'reserve_usd':3.-gpu_seconds*gpu_rate-conversions-coordinator}
