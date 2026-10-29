"""Local trusted handoff of archived, preregistered full-task curve checkpoints."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT
from evaluate import frozen_evaluate
from transport import pack_weights


@app.local_entrypoint()
def score():
    import time
    import modal
    import numpy as np
    sys.path.insert(0,str(ROOT/'src'))
    from automanip.policy_io import PolicySchema,load_policy
    root=ROOT/'runs/full-stacking-curve-20261029'
    raw=(root/'source/plan.json').read_bytes();plan=json.loads(raw)
    summary=json.loads((root/'trusted-summary.json').read_bytes())
    digest=hashlib.sha256(raw).hexdigest()
    if not plan['admitted'] or digest!=summary['plan_sha256']:raise PermissionError('Plan differs')
    proof=json.loads((ROOT/'.runtime/full-stacking-curve-archive-verification.json').read_bytes())
    if proof['state']!='verified':raise PermissionError('Archive not verified')
    budget=json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd'])>100:raise PermissionError('Cap exceeded')
    if time.time()-budget['observed_at']>300:raise PermissionError('Ledger stale')
    if not any(r['campaign']=='full-stacking-curve-20261029' and float(r['reserved_cap_usd'])==3 for r in budget['reservations']):raise PermissionError('Reservation absent')
    schema=PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8)
    payloads=[];initials=[]
    for route in ('baseline','coordinate'):
        inventory=json.loads((root/route/'inventory.json').read_bytes())
        for entry in inventory:
            rel=Path(entry['path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path')
            data=(root/route/rel).read_bytes()
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Archive drift')
        receipt=json.loads((root/route/'receipt.json').read_bytes());report=receipt['report']
        if receipt['state']!='complete' or report['state']!='complete':raise ValueError('Training incomplete')
        if receipt['route']!=route or receipt['plan_sha256']!=digest:raise ValueError('Receipt differs')
        if report['config']['seed']!=2 or report['config']['total_transitions']!=1024000 or report['config']['max_seconds']!=600.:raise ValueError('Configuration differs')
        if report['counts']['transitions']>1024000 or receipt['trainer_process_seconds']>600.5:raise ValueError('Budget differs')
        if report['study_arm']!=route or report['behavior_mode']!='coordinate' or report['behavior_epsilon']!=(0. if route=='baseline' else .1):raise ValueError('Behavior differs')
        core={n:plan['source_sha256'][n] for n in ('replay_robot.py','replay_learner.py','sac_baseline.py','joint_exploration.py')}
        if report['source_files_sha256']!=core:raise ValueError('Core differs')
        admission=report['admission']
        if admission['route']!=route or admission['launcher_sha256']!=plan['source_sha256']['full_stacking_curve.py'] or admission['collector_sha256']!=plan['source_sha256']['joint_exploration.py']:raise ValueError('Admission differs')
        initial=load_policy(root/route/'run/actor-step-0',schema)
        if initial.weights_sha256!=plan['initial_actor_sha256']:raise ValueError('Initial actor differs')
        values=np.load(root/route/'run/actor-step-0-state.npz',allow_pickle=False)
        initials.append({k:values[k].copy() for k in values.files})
        for k in values.files:
            if k.startswith('q1.') and not np.array_equal(values[k],values['target1.'+k[3:]]):raise ValueError('Initial target1 differs')
            if k.startswith('q2.') and not np.array_equal(values[k],values['target2.'+k[3:]]):raise ValueError('Initial target2 differs')
        points=report['checkpoint_exports']
        selected=[p for p in points if p['label']=='actor-step-'+str(p['transitions']) and p['transitions'] in (102400,256000,512000)]
        terminal=next(p for p in points if p['label']=='actor-final')
        selected.append(terminal)
        seen=set()
        for point in selected:
            policy=root/route/'run'/point['label'];actor=load_policy(policy,schema)
            if actor.weights_sha256!=point['weights_sha256']:raise ValueError('Checkpoint differs')
            if actor.weights_sha256 in seen:continue
            seen.add(actor.weights_sha256)
            payloads.append((route,policy,actor.weights_sha256,point))
    if set(initials[0])!=set(initials[1]) or any(not np.array_equal(initials[0][k],initials[1][k]) for k in initials[0]):raise ValueError('Paired random initialization differs')
    if len(payloads)>8:raise ValueError('Query bound exceeded')
    launch=ROOT/'.runtime/modal-full-stacking-curve-score-launch.json'
    record=dict(state='started',plan_sha256=digest,points={},new_training_runs=0,new_confirmation_queries=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with launch.open('x') as f:json.dump(record,f,indent=2)
    volume=modal.Volume.from_name('automanip-modal-validation')
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=1)
    def persist():launch.write_text(json.dumps(record,indent=2)+'\n')
    def existing(path):
        data=bytearray()
        try:
            for block in volume.read_file(path):
                data.extend(block)
                if len(data)>4*1024*1024:raise ValueError('Score size exceeded')
        except FileNotFoundError:return None
        return json.loads(data) # malformed/incomplete attempts are not absent
    for route,policy,weights,point in payloads:
        path='frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+weights+'/development/result.json'
        row=dict(route=route,checkpoint=point,weights_sha256=weights,durable_report_path=path)
        record['points'][route+'/'+point['label']]=row
        rawscore=existing(path)
        row['reused_existing_report']=rawscore is not None
        if rawscore is None:
            row.update(state='submitting',submitted_at=time.time());persist()
            call=evaluator.spawn((policy/'manifest.json').read_bytes(),pack_weights((policy/'weights.json').read_bytes()),'development','L40S',time.time(),100)
            row.update(state='submitted',function_call_id=call.object_id);persist()
            try:row['transport_result']=call.get(timeout=160)
            except Exception as exc:
                row['transport_error']=type(exc).__name__+': '+str(exc)
                persist()
            rawscore=existing(path)
        if rawscore is None:raise RuntimeError('Score absent; no repeat')
        protocol='fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764'
        if (rawscore['state']!='complete' or rawscore['weights_sha256']!=weights
            or rawscore['protocol_sha256']!=protocol or rawscore['split']!='development'
            or rawscore['expected_episodes']!=128):raise RuntimeError('Outer frozen score differs; no repeat')
        checked=rawscore['report']
        if (checked['weights_sha256']!=weights or checked['protocol_sha256']!=protocol
            or checked['split']!='development' or checked['smoke'] is not False
            or checked['episode_count']!=128 or checked['failed_episodes']!=0
            or checked['task_fingerprint']!=schema.task_fingerprint):
            raise RuntimeError('Frozen report differs; no repeat')
        dest=root/'development'/(route+'-'+point['label']+'.json');dest.parent.mkdir(exist_ok=True)
        data=(json.dumps(rawscore,indent=2)+'\n').encode()
        if dest.exists() and dest.read_bytes()!=data:raise ValueError('Existing local score differs')
        dest.write_bytes(data)
        row.update(state='complete',successes=rawscore['report']['successes_at_end'],episodes=128,
                   diagnostics_rates=rawscore['report']['diagnostics_rates'],local_report=str(dest.relative_to(ROOT)))
        persist();print(json.dumps(dict(route=route,steps=point['transitions'],successes=row['successes'],diagnostics=row['diagnostics_rates'])))
    record['state']='complete';persist()
