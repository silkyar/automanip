"""Aggregate only existing full-task learner/scorer records; no training or queries."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'runs/full-stacking-curve-20261029'
launch=json.loads((ROOT/'.runtime/modal-full-stacking-curve-score-launch.json').read_bytes())
if launch['state']!='complete':raise RuntimeError('All originally selected scores required')
original=json.loads((root/'trusted-summary.json').read_bytes())
summary=dict(state='complete',experiment='full-stacking-curve-20261029',arms={},confirmation_queries=0,
    plan_sha256=original['plan_sha256'],primary_transitions=512000,training_seed=2,
    historical_PPO_context_only=True,tenfold_or_algorithm_superiority_established=False)
for route in ('baseline','coordinate'):
    receipt=json.loads((root/route/'receipt.json').read_bytes());report=receipt['report']
    curve=[]
    for row in launch['points'].values():
        if row['route']!=route:continue
        p=row['checkpoint'];expected=max(0,(p['transitions']-4992)//16)
        curve.append(dict(steps=p['transitions'],label=p['label'],trainer_seconds=p['elapsed_seconds'],
            critic_updates=p['critic_updates'],actor_updates=p['actor_updates'],
            expected_critic_updates=expected,update_censored=p['critic_updates']!=expected,
            successes=row['successes'],episodes=128,weights_sha256=row['weights_sha256'],
            diagnostics_rates=row['diagnostics_rates'],report_path=row['durable_report_path']))
    curve.sort(key=lambda p:p['steps'])
    terminal_checkpoint=next(p for p in report['checkpoint_exports'] if p['label']=='actor-final')
    terminal_hash=terminal_checkpoint['weights_sha256']
    terminal=next(p for p in curve if p['weights_sha256']==terminal_hash)
    def crossing(threshold):
        for index,p in enumerate(curve):
            if p['successes']>=threshold:
                prior=curve[index-1] if index else None
                return dict(status='first_observed_crossing',steps=p['steps'],
                    trainer_elapsed_seconds=p['trainer_seconds'],
                    previous_observed_steps=prior['steps'] if prior else None,
                    previous_observed_trainer_seconds=prior['trainer_seconds'] if prior else None,
                    qualification='Observed checkpoint bracket only; unsampled earlier or temporary crossings remain possible')
        return dict(status='no_observed_crossing_through_last_checkpoint',
            last_observed_steps=curve[-1]['steps'],last_observed_trainer_seconds=curve[-1]['trainer_seconds'])
    summary['arms'][route]=dict(state='complete',training=receipt,development=terminal,terminal_checkpoint=terminal_checkpoint,learning_curve=curve,
        first_observed_T80=next((p['steps'] for p in curve if p['successes']>=103),None),
        first_observed_T90=next((p['steps'] for p in curve if p['successes']>=116),None),
        observed_thresholds=dict(T80=crossing(103),T90=crossing(116)),
        checkpoint_clock='Elapsed train() time through export; module imports excluded; full child-process wall separately recorded')
primary={r:next((p for p in a['learning_curve'] if p['steps']==512000),None) for r,a in summary['arms'].items()}
if not all(primary.values()):verdict='missing_or_censored_primary'
else:
    b,c=primary['baseline']['successes'],primary['coordinate']['successes']
    verdict=('promising_coordinate_full_task' if c>=64 and c-b>=20 else
             'weakens_full_task_sufficiency' if c<=8 else
             'weakens_collector_benefit' if c<=b else 'partial_or_inconclusive')
summary.update(primary_verdict=verdict,primary=primary,scored_points=len(launch['points']),
               new_development_queries=sum(not p['reused_existing_report'] for p in launch['points'].values()))
(root/'reconciled-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest=[]
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name!='final-archive-manifest.json':
        raw=path.read_bytes();manifest.append(dict(path=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
(root/'final-archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(primary_verdict=verdict,arms={r:dict(steps=a['development']['steps'],successes=a['development']['successes'],curve=[(p['steps'],p['successes'],round(128*p['diagnostics_rates']['lift'])) for p in a['learning_curve']]) for r,a in summary['arms'].items()})))
