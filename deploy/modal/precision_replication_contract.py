"""Separate, default-denied replication admission; no initial-batch changes."""
import hashlib
import json
from pathlib import Path

from precision_screen_contract import (
    ARMS, REVIEWED, MAX_STEPS, BATCH_STEPS, SELECTED, read_regular,
    remaining_lease, select_checkpoints, audit_accounting,
)

EXPERIMENT='stackcube-precision-replication-20261027'
SEEDS=(2,3)
BRANCH_CAPS={2:4.5,4:9.}


def job(method,seed):
    if method not in ARMS or type(seed) is not int or seed not in SEEDS:raise ValueError('unreviewed replication slot')
    return {'job_id':EXPERIMENT+'-'+method+'-seed-'+str(seed),'method':method,'seed':seed,
            'trainer_sha256':REVIEWED[method],'gpu':'L40S','trainer_seconds_cap':2700,
            'requested_steps':50_000_000,'actual_steps_ceiling':MAX_STEPS}


def selected_jobs(recipe):
    if recipe not in ARMS:raise ValueError('one registered recipe selection required')
    methods=('precision-constant',) if recipe=='precision-constant' else ('precision-constant',recipe)
    return [job(method,seed) for method in methods for seed in SEEDS]


def command(python,source,item):
    if item!=job(item['method'],item['seed']):raise ValueError('replication job contract changed')
    return [str(python),str(source),'--env-id','StackCube-v1','--seed',str(item['seed']),
            '--num-envs','1024','--num-eval-envs','8','--num-minibatches','32','--update-epochs','8',
            '--total-timesteps','50000000','--exp-name',item['job_id'],'--gamma','0.9',
            '--ent-coef','0.003','--learning-rate','0.0003','--no-partial-reset','--no-capture-video','--no-track']


def volume_name(item):
    if item!=job(item['method'],item['seed']):raise ValueError('invalid Volume slot')
    return 'automanip-modal-replicate-'+item['method']+'-seed'+str(item['seed'])+'-20261027'


def conservative_cost(recipe):
    count=len(selected_jobs(recipe))
    gpu_seconds=count*(2760+30)+3*count*(120+30)
    gpu=gpu_seconds*(.000542+4*.0000131+16*.00000222)
    conversions=3*count*120*(.0000131+2*.00000222)
    coordinator=7200*(.125*.0000131+.5*.00000222)
    subtotal=gpu+conversions+coordinator;cap=BRANCH_CAPS[count]
    return {'job_count':count,'gpu_container_seconds_bound':gpu_seconds,'gpu_cpu_ram_usd':gpu,
            'conversion_cpu_usd':conversions,'coordinator_usd':coordinator,
            'subtotal_usd':subtotal,'branch_cap_usd':cap,'reserve_usd':cap-subtotal}


def validate_admission(plan):
    """Root admission must name exactly one complete, outcome-qualified branch."""
    if not plan.get('launch_enabled') or not plan.get('paid_launch_authorized') or plan.get('root_review_required',True):
        raise PermissionError('separate outcome-based root admission required')
    recipe=plan.get('selected_recipe');expected=selected_jobs(recipe)
    if plan.get('experiment_id')!=EXPERIMENT or plan.get('jobs')!=expected:raise ValueError('exact branch jobs required')
    if plan.get('training_seeds')!=list(SEEDS) or plan.get('confirmation_queries')!=0:raise ValueError('invalid replication query scope')
    if plan.get('max_parallel_trainers')!=len(expected) or plan.get('max_parallel_evaluators')!=len(expected):raise ValueError('dedicated branch concurrency required')
    if plan.get('budget_authorization_usd')!=BRANCH_CAPS[len(expected)] or plan.get('cost')!=conservative_cost(recipe):raise ValueError('exact reviewed branch cost required')
    if plan.get('protocol_sha256')!='fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764':raise ValueError('frozen protocol changed')
    qualification=plan.get('outcome_qualification') or {}
    if qualification.get('selected_recipe')!=recipe or qualification.get('initial_campaign')!='stackcube-precision-screen-20261027' or not qualification.get('root_reviewed'):
        raise PermissionError('root-reviewed initial outcome qualification required')
    counts=qualification.get('final_development_counts',{})
    if set(counts)!=set(ARMS) or any(type(v) is not int or not 0<=v<=128 for v in counts.values()):raise ValueError('complete initial final counts required')
    if counts[recipe]<126 or qualification.get('episodes')!=128:raise ValueError('selected recipe below preregistered replication screening level')
    if recipe!='precision-constant' and counts['precision-constant']>=126 and counts[recipe]-counts['precision-constant']<3:raise ValueError('constant-preference decision rule violated')
    if plan.get('development_episodes')!=128 or plan.get('selected_steps')!=[25_600_000,38_400_000,'final_actual_steps'] or plan.get('max_development_evaluations')!=3*len(expected):raise ValueError('frozen development scope changed')
    return expected
