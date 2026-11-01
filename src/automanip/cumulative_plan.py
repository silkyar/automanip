"""Compile an exact, unlaunched transfer plan. No simulator/cloud execution."""
import hashlib
from pathlib import Path

from automanip.cumulative import (ROOT,SOURCE,NORTH,asset_status,compatible,load_asset,
    load_policy,register_asset,cost,lifetime_cost,relative_path,write_json,_json,_read)

SOURCE_POLICY = 'runs/stackcube-bc-exploration-pilot-20261028/teacher-input'
SOURCE_WEIGHTS = 'ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16'
SOURCE_MANIFEST = 'cumulative/assets/stackcube-50m-seed1-v1.json'
SOURCE_DATA = 'runs/stackcube-bc-exploration-pilot-20261028/collect/native/run'


def register_existing_source(*,root=ROOT,destination=None):
    root=Path(root);destination=root/SOURCE_MANIFEST if destination is None else Path(destination)
    if destination.exists():
        asset=load_asset(destination,root=root)
        if asset.policy.weights_sha256!=SOURCE_WEIGHTS:raise ValueError('Existing source registry differs')
        return asset
    policy=root/SOURCE_POLICY
    actor=load_policy(policy,SOURCE.schema)
    if actor.weights_sha256!=SOURCE_WEIGHTS:raise ValueError('Exact source actor unavailable')
    history=_json(_read(root/'runs/stackcube-bc-exploration-pilot-20261028/summary.json',4*1024*1024))
    prior=history['teacher_prefix']
    if prior['actor_weights_sha256']!=SOURCE_WEIGHTS or prior['training_steps']!=49971200 or prior['training_seed']!=1:
        raise ValueError('Source acquisition provenance differs')
    confirmations=_json(_read(root/'runs/private-confirmation/stackcube-precision-20261028/summary.json',4*1024*1024))
    point=next(r for r in confirmations['records'] if r['weights_sha256']==SOURCE_WEIGHTS)
    evidence=dict(task_id=SOURCE.task_id,partition='historical_confirmation_not_new_feedback',
        successes=point['report']['successes_at_end'],episodes=point['report']['episode_count'],
        report='runs/private-confirmation/stackcube-precision-20261028/summary.json',
        note='Whole stacking policy applicability only; target north variation unassessed')
    metadata=_json(_read(root/SOURCE_DATA/'dataset.json',1024*1024))
    raw_path=root/SOURCE_DATA/'trajectories.npz'
    if metadata['source_partition']!='public-training' or hashlib.sha256(raw_path.read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('Source experience digest/partition differs')
    experience=dict(format=metadata['format'],path=SOURCE_DATA,sha256=metadata['sha256'],partition='training',
        task_id=SOURCE.task_id,episodes=metadata['completed_episodes'],failures=metadata['failed_completed_episodes'],
        used_by_this_PPO_strategy=False,
        acquisition=cost(metadata['environment_transitions'],history['stages']['collect']['trainer_process_seconds'],None),
        note='Available prior experience; no source reward/critic/replay used for the new goal')
    return register_asset(destination,policy,SOURCE,cost(prior['training_steps'],prior['trainer_seconds'],prior['dollars']),
                          root=root,evidence=(evidence,),data_assets=(experience,))


def make_plan(source_asset, *,root=ROOT):
    code_paths=('train.py','research/discovery/amp_v1.py','src/automanip/cumulative.py','src/automanip/cumulative_ppo.py','src/automanip/cumulative_episodes.py','src/automanip/cumulative_env.py','src/automanip/cumulative_plan.py')
    pins={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_paths}
    return dict(source_sha256=pins,format='automanip-cumulative-transfer-plan-v1',version=1,execution_authorized=False,
        scope='Local preparation only. No cloud/GPU/collection/scoring; no historical admission authority.',
        source_asset=relative_path(source_asset.manifest_path,root),source_asset_id=source_asset.asset_id,
        source_weights_sha256=source_asset.policy.weights_sha256,
        source_task=SOURCE.to_dict(),target_task=NORTH.to_dict(),
        variation='A.x*=.4,A.y=.12+.1*y; B.x*=.25,B.y=.34+y/15; preserve z/quaternion; reset-only, no RNG/physics step',
        arms=['unchanged-source-reference','scratch-context','ordinary-actor-warmstart'],
        learner_seeds=dict(pilot=41,replication_unlaunched=[42,43]),
        environment_seed_bases=dict(pilot=93440000,replication_unlaunched=[93441000,93442000]),
        reset_batches_max=640,development_seeds=list(range(93450000,93450128)),
        final_seeds=list(range(93550000,93550512)),
        development_at=[0,128000,512000,1024000,2048000],
        budget_per_trainable_arm=dict(training_interactions=2048000,optimizer_calls=163840,seconds=600,development_interactions=64000),
        ppo=dict(num_envs=64,num_steps=50,hidden_sizes=[256,256,256],gamma=.9,gae_lambda=.9,learning_rate=.0003,
            ent_coef=.003,clip_coef=.2,vf_coef=.5,max_grad_norm=.5,target_kl=.1,update_epochs=8,num_minibatches=32,
            partial_reset=False,bootstrap_on_termination=False,anneal_lr=False,initial_logstd=-.5),
        gate=dict(target_success_min=.9,source_retention_min=.9,max_source_drop=.05,
            development_counts=dict(target_min=116,source_min=116,max_drop=6),
            final_counts=dict(target_min=461,source_min=461,max_drop=25)),
        selection='Earliest fully evaluated declared development checkpoint passing BOTH domains; no interpolation/replacement. If none, describe failure. Source already passes B => report zero-shot; no automatic training.',
        retention='SAME adapted controller in A and B; archived source/router competence does not satisfy retention',
        prior_cost=lifetime_cost(source_asset,root=root),
        feedback_clock='600s total includes outer lookup/import/factory/init/training/resets/export/development; future hard process supervisor required; current adapt() only covers inner clock',
        final_budget=dict(max_controller_digests=3,episodes_per_domain=512,domains=2,environment_transitions=153600),
        unresolved=['Real north-strip feasibility/reachability untested','New-track effective simulator clocks/throughput unmeasured',
            'Production outer hard deadline, isolated final-assessment admission and durable cross-process handoff not validated'],
        claim='Variation adaptation only; no cumulative A-to-B-to-C or new-mechanism benefit demonstrated')


def compile_plan(plan,destination,*,root=ROOT):
    if plan.get('format')!='automanip-cumulative-transfer-plan-v1' or plan.get('execution_authorized') is not False:
        raise PermissionError('This compiler cannot authorize or launch an experiment')
    status=asset_status(Path(root)/plan['source_asset'],root=root)
    result=dict(state='prepared_not_executed',execution_authorized=False,source_asset=status,arms=[],
        missing_assets=[],unvalidated=list(plan['unresolved']))
    if status['state']!='available':
        result.update(state='missing_or_invalid_assets',missing_assets=[status]);write_json(destination,result);return result
    asset=load_asset(Path(root)/plan['source_asset'],root=root);compatible(asset,NORTH)
    if asset.asset_id!=plan['source_asset_id'] or asset.policy.weights_sha256!=plan['source_weights_sha256']:
        raise ValueError('Plan source identity differs')
    if plan['source_task']!=SOURCE.to_dict() or plan['target_task']!=NORTH.to_dict():raise ValueError('Goal/interface contract differs')
    if plan!=make_plan(asset,root=root):raise ValueError('Plan differs from the closed registered recipe/seeds/gates/source pins')
    from automanip.cumulative_ppo import Budget,default_config
    if list(asset.policy.hidden_sizes)!=plan['ppo']['hidden_sizes']:raise ValueError('Source architecture differs from pilot recipe')
    if plan['ppo']!={k:getattr(default_config(),k) for k in plan['ppo'] if k!='initial_logstd'}|{'initial_logstd':-.5}:
        # JSON hidden-size arrays are equivalent to config tuples, handled below.
        expected={k:(list(getattr(default_config(),k)) if k=='hidden_sizes' else getattr(default_config(),k)) for k in plan['ppo'] if k!='initial_logstd'}|{'initial_logstd':-.5}
        if plan['ppo']!=expected:raise ValueError('PPO recipe differs from registered implementation')
    Budget(**plan['budget_per_trainable_arm']).validate()
    train_sets=[set(range(base,base+plan['reset_batches_max'])) for base in [plan['environment_seed_bases']['pilot'],*plan['environment_seed_bases']['replication_unlaunched']]]
    dev,final=set(plan['development_seeds']),set(plan['final_seeds'])
    if len(dev)!=128 or len(final)!=512 or dev&final or any(s&(dev|final) for s in train_sets):raise ValueError('Episode partitions overlap or counts differ')
    if any(a&b for i,a in enumerate(train_sets) for b in train_sets[i+1:]):raise ValueError('Training reset namespaces overlap')
    for arm in plan['arms']:
        result['arms'].append(dict(method=arm,parent_asset=None if arm=='scratch-context' else asset.asset_id,
            source_weights=None if arm=='scratch-context' else asset.policy.weights_sha256,
            adaptation='none' if arm=='unchanged-source-reference' else 'PPO',
            budget=None if arm=='unchanged-source-reference' else plan['budget_per_trainable_arm'],
            assessment_tasks=[SOURCE.task_id,NORTH.task_id],development_at=[0] if arm=='unchanged-source-reference' else plan['development_at']))
    result.update(plan=plan,ready_for_local_synthetic_tests=True,ready_for_robot_execution=False)
    write_json(destination,result)
    return result
