"""Extract aggregate-only publication data from the parent research archive.
No simulator, pickle/checkpoint loading, scorer, network or private episode reads.
"""
import argparse,hashlib,json,re,tarfile
from pathlib import Path
FAMILIES=[('StackCube official PPO','Official PPO',True),('gamma090','Discount 0.9',True),('gamma095','Discount 0.95',False),('tune-gamma0875-control','Discount 0.875',False),('tune-gamma0925-control','Discount 0.925',False),('tune-entropy001-control','Entropy 0.001',False),('tune-entropy003-control','Entropy 0.003',True),('settle-full-horizon','Full horizon',True),('settle-entropy-decay','Entropy decay',False),('settle-gamma0925-entropy003','Discount 0.925 + entropy',False)]
STEPS=24985600

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);args=p.parse_args();root=args.source_root
 out=Path(__file__).resolve().parents[1]/'assets';aggregate=root/'docs/experiment-runs.json';doc=json.loads(aggregate.read_bytes());rows=doc['runs'];families=[]
 for i,(name,label,kept) in enumerate(FAMILIES,1):
  runs=[r for r in rows if r['experiment']==name]
  if len(runs)!=3 or {r['seed'] for r in runs}!={1,2,3} or any(r['environment_transitions']!=STEPS or r['initialization']!='random' or r['task']!='StackCube-v1' or r['gpu']!='L4' for r in runs):raise ValueError('comparison mismatch')
  public=[{k:r[k] for k in ('seed','trainer_seconds','environment_transitions','development_successes','development_episodes','confirmation_successes','confirmation_episodes','weights_sha256')} for r in runs]
  families.append(dict(index=i,experiment=name,label=label,selected=kept,runs=public,development_mean_percent=sum(r['development_successes']/r['development_episodes'] for r in runs)/3*100,confirmation_mean_percent=None if any(r['confirmation_successes'] is None for r in runs) else sum(r['confirmation_successes']/r['confirmation_episodes'] for r in runs)/3*100))
 data=dict(task='StackCube-v1',metric='Frozen terminal StackCube success',training_transitions_per_seed=STEPS,training_seeds=[1,2,3],development_episodes_per_seed=128,confirmation_episodes_per_seed=512,families=families,source='docs/experiment-runs.json',source_sha256=hashlib.sha256(aggregate.read_bytes()).hexdigest(),exclusions=['PickCube task','12.5M and shorter budgets','50M and longer budgets','assisted/continuation','SAC','CAPG on L40S','settle-timed-control duplicates identical entropy-policy digests'],path_note='Retrospective selected recipe path, not a formal running maximum or strict serial experiment clock.',timing_note='Primary comparison holds transitions fixed; elapsed training time differs. No interpolation to22 minutes.')
 (out/'progress-data.json').write_text(json.dumps(data,indent=2)+'\n')
 # Existing independent development checkpoints and existing scalar timestamps only.
 # Never read confirmation reports. Read the early started_at metadata and stop.
 archive_specs=[('.runtime/stackcube-20261026-results.tar.gz','automanip-stackcube-20261026','StackCube official PPO','stackcube-seed-','runs'),('.runtime/overnight-20261027-results.tar.gz','automanip-research-20261026-credit-v7','gamma090','gamma090-seed-','records'),('.runtime/overnight-20261027-results.tar.gz','automanip-research-20261026-credit-v7/overnight/tune','tune-entropy003-control','tune-entropy003-control-seed-','records'),('.runtime/settling-20261027-results.tar.gz','automanip-settling-20261027/experiments','settle-full-horizon','settle-full-horizon-seed-','records'),('.runtime/settling-20261027-results.tar.gz','automanip-settling-20261027/experiments','settle-timed-control','settle-timed-control-seed-','records')]
 points=[];assessment=[]
 for archive,prefix,experiment,nameprefix,records_dir in archive_specs:
  canonical='tune-entropy003-control' if experiment=='settle-timed-control' else experiment
  with tarfile.open(root/archive) as t:
   for seed in (1,2,3):
    name=nameprefix+str(seed);record_name=prefix+'/'+records_dir+'/'+name+'/run.json'
    started=None
    for line in t.extractfile(record_name):
     match=re.search(rb'"started_at":\s*([\d.]+)',line)
     if match:started=float(match[1]);break
    if started is None:raise ValueError('start timestamp missing')
    training=prefix+('/runs/' if records_dir=='runs' else '/training/runs/')+name
    scalar_member=training+'/learning-curves.json';logged=json.load(t.extractfile(scalar_member));step_times={int(e['value']):e['wall_time']-started for e in logged['time/step']}
    curve_name=prefix+'/'+records_dir+'/'+name+'/independent-learning-curve.json';curve=json.load(t.extractfile(curve_name))
    run=next(r for r in rows if r['experiment']==experiment and r['seed']==seed);valid=[]
    if experiment=='settle-timed-control':
     original=next(r for r in rows if r['experiment']==canonical and r['seed']==seed)
     if run['weights_sha256']!=original['weights_sha256']:raise ValueError('timed rerun final actor differs from equivalent original recipe')
    for q in curve:
     step=q['environment_steps'];report=q['report'];seconds=0. if step==0 else run['trainer_seconds'] if step==STEPS else step_times[step]
     if report['split']!='development' or report['episode_count']!=128 or report['failed_episodes'] or report['smoke']:raise ValueError('unexpected checkpoint scorer')
     if step==STEPS and report['weights_sha256']!=run['weights_sha256']:raise ValueError('final report digest differs from recorded actor')
     if seconds<0:raise ValueError('negative checkpoint timestamp')
     point=dict(experiment=canonical,source_experiment=experiment,seed=seed,environment_transitions=step,training_seconds=seconds,successes=report['successes_at_end'],episodes=128,weights_sha256=report['weights_sha256'],checkpoint_sha256=q.get('checkpoint_sha256'),timestamp_basis='final measured trainer process duration' if step==STEPS else 'initial checkpoint' if step==0 else 'time/step TensorBoard wall_time minus recorded pre-process started_at',archive=archive,curve_member=curve_name,scalar_member=scalar_member)
     points.append(point)
     if seconds<=1320:valid.append(point)
    chosen=max(valid,key=lambda r:r['environment_transitions']) if valid else None
    assessment.append(dict(experiment=canonical,source_experiment=experiment,seed=seed,latest_independently_scored_checkpoint_available_by1320s=chosen,final_trainer_seconds=run['trainer_seconds'],full25M_completed_by1320s=run['trainer_seconds']<=1320))
 merged=[]
 for family in (f for f in families if f['selected']):
  for seed in (1,2,3):
   records=[r for r in assessment if r['experiment']==family['experiment'] and r['seed']==seed]
   eligible=[r['latest_independently_scored_checkpoint_available_by1320s'] for r in records if r['latest_independently_scored_checkpoint_available_by1320s'] is not None]
   chosen=max(eligible,key=lambda r:r['environment_transitions']) if eligible else None
   merged.append(dict(experiment=family['experiment'],seed=seed,latest_independently_scored_checkpoint_available_by1320s=chosen,equivalent_records_considered=records))
 report=dict(cutoff_seconds=1320,selection='Latest-step existing independently scored checkpoint available by1320s across equivalent recipe records; timed-control final entropy actors must match original weights digests exactly. No interpolation, new scoring, or exact22-minute endpoint claim.',equivalent_recipe_note='The timed-control re-execution is the same entropy.003 recipe/seed and produces identical final safe actor digests; it supplies valid full-budget timing evidence, not additional independent policies.',timing_caveat='Intermediate time/step timestamps occur after the logged update; started_at precedes process launch. Conservative availability snapshots, not exact budget-terminated reruns. Final time uses measured child duration. Initialization/evaluation times are not silently included as training.',points=points,per_run=merged,primary_25M_comparison_not_fixed_wall_time=True)
 (out/'timing-checkpoints.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'families':len(families),'training_runs':30,'checkpoint_points':len(points),'assets':str(out)}))
if __name__=='__main__':main()
