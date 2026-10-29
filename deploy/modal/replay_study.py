"""Root-admitted single-mechanism replay study; trainer never sees scorer files."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,validation_image,results
from smoke_capg import runtime_mount
from evaluate import frozen_evaluate

EXPERIMENT='stackcube-replay-moments-20261028'
METHODS=('target-sac','joint-moments','split-moments')
PLAN=ROOT/'deploy/modal/replay-study-plan.json'
candidate_image=trainer_image
for name in ('replay_learner.py','replay_robot.py','sac_baseline.py'):
    candidate_image=candidate_image.add_local_file(ROOT/'research/discovery'/name,'/opt/replay/'+name,copy=True)
candidate_image=candidate_image.add_local_file(PLAN,'/opt/replay/plan.json',copy=True)
coordinator_image=validation_image.add_local_file(PLAN,'/opt/replay-plan.json',copy=True)
outputs=modal.Volume.from_name('automanip-replay-moments-20261028',create_if_missing=True,version=2)


@app.function(image=candidate_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),
    timeout=680,startup_timeout=30,retries=0,max_containers=3,scaledown_window=2,
    restrict_modal_access=True,block_network=True,single_use_containers=True,
    include_source=False,serialized=True,_experimental_restrict_output=True)
def train_replay(phase:str,method:str,seed:int,plan_sha:str,expires:float):
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    import traceback
    root=Path('/candidate-output');started=time.monotonic()
    with (root/'attempt.json').open('x') as f:json.dump(dict(phase=phase,method=method,seed=seed,expires=expires),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    record=dict(state='started',phase=phase,method=method,seed=seed,plan_sha256=plan_sha,trials=[])
    try:
        raw=Path('/opt/replay/plan.json').read_bytes();plan=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=plan_sha or phase not in plan['admitted_phases']:raise PermissionError('Exact phase not admitted')
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('Trainer isolation failed')
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((Path('/opt/replay')/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source differs')
        import torch
        if torch.cuda.get_device_name(0)!='NVIDIA L40S':raise RuntimeError('GPU pin differs')
        cache=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest()!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':
            raise RuntimeError('Pinned PhysX runtime cache differs')
        if phase=='smoke':
            if method!='all' or seed!=1:raise ValueError('One fixed smoke required')
            methods=('target-sac','joint-moments','split-moments')
        else:
            if method not in ('target-sac','joint-moments','split-moments') or seed not in plan['training_seeds']:
                raise ValueError('Unregistered scientific trial')
            methods=(method,)
        for selected in methods:
            remaining=min(60. if phase=='smoke' else 600.,expires-time.time()-15)
            if remaining<25:raise RuntimeError('Original submission lease expired; no refresh')
            directory=root/selected;directory.mkdir()
            seconds=45 if phase=='smoke' else 600
            command=[sys.executable,'/opt/replay/replay_robot.py','--method',selected,'--seed',str(seed),
                '--total-transitions','5120' if phase=='smoke' else '1024000',
                '--max-seconds',str(seconds),'--output',str(directory/'run')]
            env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
            env['AUTOMANIP_TRAINING_DEADLINE']=str(time.monotonic()+remaining-15)
            mark=time.monotonic()
            with (directory/'training.log').open('w') as stream:
                process=subprocess.Popen(command,cwd='/opt/replay',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Hard trial time cap exceeded')
            seconds=time.monotonic()-mark
            if code:raise RuntimeError('Candidate subprocess failed; preserve log')
            report=json.loads((directory/'run/report.json').read_bytes())
            if report['state']!='complete' or report['counts']['transitions']>(5120 if phase=='smoke' else 1024000):
                raise RuntimeError('Incomplete or over-budget report')
            if report['source_files_sha256']!=plan['source_sha256']:raise ValueError('Runtime source mismatch')
            record['trials'].append(dict(method=selected,trainer_process_seconds=seconds,report=report))
        record['state']='complete'
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        inventory=[]
        for path in sorted(root.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink output refused')
            if path.is_file() and path.name not in ('receipt.json','inventory.json'):
                raw=path.read_bytes()
                if len(raw)>64*1024*1024:raise ValueError('Individual output limit')
                inventory.append(dict(path=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if sum(r['bytes'] for r in inventory)>256*1024*1024:raise ValueError('Aggregate output limit')
        record['wrapper_seconds']=time.monotonic()-started
        (root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        (root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=10)
    return dict(state=record['state'],error=record.get('error'),receipt='receipt.json')


@app.function(image=coordinator_image,cpu=1,memory=2048,timeout=3600,retries=0,max_containers=1,
    single_use_containers=True,include_source=False,serialized=True,_experimental_restrict_output=True,
    volumes={'/candidate':outputs.with_mount_options(read_only=True),'/results':results})
def coordinate_replay(workers:dict,evaluator:modal.Function,plan_sha:str):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/automanip/src');sys.path.insert(0,'/opt/automanip-modal')
    from automanip.policy_io import load_policy,PolicySchema
    from transport import pack_weights
    plan_raw=Path('/opt/replay-plan.json').read_bytes();plan=json.loads(plan_raw)
    if hashlib.sha256(plan_raw).hexdigest()!=plan_sha or 'screen' not in plan['admitted_phases']:raise PermissionError('Screen not admitted')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    campaign='stackcube-replay-moments-20261028-screen-seed-1'
    if not claims.put(campaign,dict(plan_sha256=plan_sha,started=time.time()),skip_if_exists=True):raise RuntimeError('Scientific screen already claimed')
    out=Path('/results')/campaign;out.mkdir(exist_ok=False)
    summary=dict(state='started',experiment=campaign,plan_sha256=plan_sha,training=[],development_curves={},confirmation_queries=0)
    def persist():
        target=out/'summary.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');tmp.replace(target)
        modal.Volume.from_name('automanip-modal-validation').commit()
    persist();calls={}
    try:
        dispatched=time.time();expires=min(dispatched+650,plan['window_deadline_epoch']-200)
        if expires-dispatched<620:raise RuntimeError('Insufficient autonomous window for a fresh screen')
        for method,worker in workers.items():
            key=campaign+'-'+method
            if not claims.put(key,dict(dispatched_at=dispatched,expires_at=expires),skip_if_exists=True):raise RuntimeError('Trial already claimed')
            calls[method]=worker.spawn('screen',method,1,plan_sha,expires)
        summary['calls']={method:call.object_id for method,call in calls.items()};persist()
        for method,call in calls.items():
            try:call.get()
            except Exception as e:summary.setdefault('transport_errors',{})[method]=type(e).__name__+': '+str(e)
            modal.Volume.from_name('automanip-replay-moments-20261028').reload()
            root=Path('/candidate/screen')/method/'seed-1'
            receipt=json.loads((root/'receipt.json').read_bytes())
            if receipt['state']!='complete':raise RuntimeError('Trial failed: '+str(receipt.get('error')))
            trial=receipt['trials'][0];summary['training'].append(trial);persist()
            inventory={r['path']:r for r in json.loads((root/'inventory.json').read_bytes())}
            report=trial['report'];actual=report['counts']['transitions']
            selected=[p for p in report['checkpoint_exports'] if p['transitions'] in (51200,102400,256000,512000,1024000) and p['label']!='actor-final']
            final=next(p for p in report['checkpoint_exports'] if p['label']=='actor-final')
            if not any(p['transitions']==actual for p in selected):selected.append(final)
            summary['development_curves'][method]=[]
            for point in selected:
                import re
                if not re.fullmatch(r'actor-(?:step-[0-9]+|final)',point['label']):raise ValueError('Unexpected candidate export label')
                if type(point['transitions']) is not int or not 0<point['transitions']<=actual:raise ValueError('Invalid checkpoint step')
                policy=out/method/('step-'+str(point['transitions']));policy.mkdir(parents=True)
                for filename in ('manifest.json','weights.json'):
                    name=method+'/run/'+point['label']+'/'+filename
                    raw=(root/name).read_bytes();entry=inventory[name]
                    if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Candidate output changed')
                    (policy/filename).write_bytes(raw)
                actor=load_policy(policy,PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8))
                if actor.weights_sha256!=point['weights_sha256']:raise ValueError('Actor digest mismatch')
                # Commit safe artifacts before any evaluator-side commit/reload.
                persist()
                path='frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+actor.weights_sha256+'/development/result.json'
                durable=Path('/results')/path
                if durable.exists():
                    result=json.loads(durable.read_bytes())
                    if result['state']!='complete':raise RuntimeError('Prior incomplete global score; no repeat')
                    reused=True
                else:
                    try:
                        evaluator.remote((policy/'manifest.json').read_bytes(),pack_weights((policy/'weights.json').read_bytes()),'development','L40S',time.time(),100)
                    except Exception as exc:
                        summary.setdefault('evaluation_transport_errors',[]).append(dict(method=method,steps=point['transitions'],error=type(exc).__name__+': '+str(exc)))
                    modal.Volume.from_name('automanip-modal-validation').reload()
                    if not durable.exists():raise RuntimeError('No durable evaluator result; preserve attempt, never auto-rescore')
                    result=json.loads(durable.read_bytes());reused=False
                if result['weights_sha256']!=actor.weights_sha256 or result['state']!='complete':raise ValueError('Frozen score provenance differs')
                summary['development_curves'][method].append(dict(steps=point['transitions'],
                    trainer_seconds=point['elapsed_seconds'],weights_sha256=actor.weights_sha256,
                    successes=result['report']['successes_at_end'],episodes=128,
                    reused_existing_report=reused,report_path=path))
                persist()
        summary['state']='complete';persist()
    except Exception as exc:summary.update(state='failed',error=type(exc).__name__+': '+str(exc));persist();raise
    return dict(state='complete',summary_path=str((out/'summary.json').relative_to('/results')))


@app.local_entrypoint()
def launch(phase:str='smoke'):
    import time
    raw=PLAN.read_bytes();plan=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if phase not in plan['admitted_phases']:raise PermissionError('Phase not root-admitted')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['adapter_sha256']:raise ValueError('Adapter changed')
    for name,sha in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/'research/discovery'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Source changed')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:raise ValueError('Scientific protocol changed')
    budget=json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd'])>100:raise PermissionError('Shared budget exceeded')
    target=ROOT/'.runtime'/('modal-replay-'+phase+'-launch.json')
    if target.exists():raise RuntimeError('Phase already submitted; read durable attempt')
    if phase=='smoke':
        worker=train_replay.with_options(timeout=220,volumes={**runtime_mount,'/candidate-output':outputs.with_mount_options(sub_path='/smoke')})
        worker.hydrate();call=worker.spawn('smoke','all',1,digest,time.time()+200)
        path='smoke/receipt.json'
    else:
        workers={}
        for method in METHODS:
            worker=train_replay.with_options(volumes={**runtime_mount,'/candidate-output':outputs.with_mount_options(sub_path='/screen/'+method+'/seed-1')})
            worker.hydrate();workers[method]=worker
        evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=3);evaluator.hydrate()
        call=coordinate_replay.spawn(workers,evaluator,digest);path=EXPERIMENT+'-screen-seed-1/summary.json'
    record=dict(phase=phase,function_call_id=call.object_id,plan_sha256=digest,receipt_path=path,
        output_volume='automanip-replay-moments-20261028',submitted_at=time.time(),cap_usd=1 if phase=='smoke' else 6)
    with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
