"""Root-reviewed one-factor early-grasp test, independent scorer unchanged."""
import hashlib,json
from pathlib import Path
import sys
import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,validation_image,results
from evaluate import frozen_evaluate

PLAN=ROOT/'deploy/modal/gamma-sufficiency-plan.json'
image=trainer_image.add_local_file(PLAN,'/opt/gamma/plan.json',copy=True)
for name in ('gamma_sufficiency.py','replay_robot.py','replay_learner.py','sac_baseline.py'):
    image=image.add_local_file(ROOT/'research/discovery'/name,'/opt/gamma/'+name,copy=True)
coordinator_image=validation_image.add_local_file(PLAN,'/opt/gamma-plan.json',copy=True)
outputs=modal.Volume.from_name('automanip-gamma-sufficiency-20261028',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=300,startup_timeout=30,
    retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,restrict_modal_access=True,
    block_network=True,include_source=False,serialized=True,_experimental_restrict_output=True,
    volumes={'/candidate-output':outputs,'/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def train_gamma(plan_sha:str,expires_at:float):
    import hashlib,json,os,signal,subprocess,sys,time,traceback
    from pathlib import Path
    root=Path('/candidate-output');mark=time.monotonic()
    with (root/'attempt.json').open('x') as f:json.dump(dict(plan_sha256=plan_sha,expires_at=expires_at),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    record=dict(state='started',plan_sha256=plan_sha)
    try:
        raw=Path('/opt/gamma/plan.json').read_bytes();plan=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=plan_sha or not plan['admitted']:raise PermissionError('Exact screen not admitted')
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('Trainer isolation differs')
        for name,expected in plan['source_sha256'].items():
            if hashlib.sha256((Path('/opt/gamma')/name).read_bytes()).hexdigest()!=expected:raise ValueError('Reviewed source differs')
        import torch
        if torch.cuda.get_device_name(0)!='NVIDIA L40S':raise RuntimeError('GPU differs')
        cache=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest()!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':raise RuntimeError('Pinned physics cache differs')
        left=min(240.,expires_at-time.time()-15)
        if left<45:raise RuntimeError('Original lease expired; no refresh')
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(time.monotonic()+left-15)
        with (root/'training.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'/opt/gamma/gamma_sufficiency.py','--output',str(root/'run')],
                cwd='/opt/gamma',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic()
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Hard trainer cap reached')
        record['trainer_process_seconds']=time.monotonic()-start
        if code:raise RuntimeError('Trainer failed; preserve log')
        report=json.loads((root/'run/report.json').read_bytes())
        if report['state']!='complete' or report['counts']['transitions']>256000:raise RuntimeError('Invalid/censored report')
        if report['admission']['launcher_sha256']!=plan['source_sha256']['gamma_sufficiency.py']:raise ValueError('Launcher differs')
        if report['source_files_sha256']!={k:v for k,v in plan['source_sha256'].items() if k!='gamma_sufficiency.py'}:raise ValueError('Core source differs')
        initial=next(x for x in report['checkpoint_exports'] if x['transitions']==0)
        if initial['weights_sha256']!=plan['initial_actor_sha256']:raise ValueError('Paired initialization differs')
        record.update(state='complete',report=report)
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds']=time.monotonic()-mark
        (root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        inventory=[]
        for path in sorted(root.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink output refused')
            if path.is_file() and path.name!='inventory.json':
                raw=path.read_bytes()
                if len(raw)>32*1024*1024:raise ValueError('Individual output bound')
                inventory.append(dict(path=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if sum(x['bytes'] for x in inventory)>128*1024*1024:raise ValueError('Aggregate output bound')
        (root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=10)
    return dict(state=record['state'],error=record.get('error'))


@app.function(image=coordinator_image,cpu=1,memory=2048,timeout=1200,retries=0,max_containers=1,
    single_use_containers=True,include_source=False,serialized=True,_experimental_restrict_output=True,
    volumes={'/gamma-candidate':outputs.with_mount_options(read_only=True),'/results':results})
def coordinate_gamma(worker:modal.Function,evaluator:modal.Function,plan_sha:str):
    import hashlib,json,sys,time,modal
    from pathlib import Path
    sys.path.insert(0,'/opt/automanip/src');sys.path.insert(0,'/opt/automanip-modal')
    from automanip.policy_io import load_policy,PolicySchema
    from transport import pack_weights
    raw=Path('/opt/gamma-plan.json').read_bytes();plan=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=plan_sha or not plan['admitted']:raise PermissionError('Exact screen not admitted')
    experiment='gamma-sufficiency-20261028-seed-1'
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    if not claims.put(experiment,dict(started=time.time(),plan_sha256=plan_sha),skip_if_exists=True):raise RuntimeError('Attempt already claimed')
    out=Path('/results')/experiment;out.mkdir(exist_ok=False)
    summary=dict(state='started',experiment=experiment,plan_sha256=plan_sha,development_curve=[],confirmation_queries=0)
    def persist():
        p=out/'summary.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');q.replace(p)
        modal.Volume.from_name('automanip-modal-validation').commit()
    persist()
    try:
        call=worker.spawn(plan_sha,time.time()+285);summary['worker_call_id']=call.object_id;persist()
        try:call.get()
        except Exception as exc:summary['transport_error']=type(exc).__name__+': '+str(exc)
        modal.Volume.from_name('automanip-gamma-sufficiency-20261028').reload()
        root=Path('/gamma-candidate');receipt=json.loads((root/'receipt.json').read_bytes())
        if receipt['state']!='complete':raise RuntimeError('Worker failed: '+str(receipt.get('error')))
        summary['training']=receipt;persist()
        inventory={x['path']:x for x in json.loads((root/'inventory.json').read_bytes())}
        report=receipt['report']
        for filename in ('manifest.json','weights.json'):
            name='run/actor-step-0/'+filename;raw=(root/name).read_bytes();entry=inventory[name]
            if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Initial artifact changed')
        initial=load_policy(root/'run/actor-step-0',PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8))
        if initial.weights_sha256!=plan['initial_actor_sha256']:raise ValueError('Initial safe artifact differs')
        summary['initial_safe_actor_validated']=True;persist()
        for steps in (51200,102400,256000):
            point=next((x for x in report['checkpoint_exports'] if x['transitions']==steps and x['label']=='actor-step-'+str(steps)),None)
            if point is None:
                summary.setdefault('missing_checkpoints',[]).append(steps);persist();continue
            policy=out/('step-'+str(steps));policy.mkdir()
            for filename in ('manifest.json','weights.json'):
                name='run/'+point['label']+'/'+filename;raw=(root/name).read_bytes();entry=inventory[name]
                if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Candidate artifact changed')
                (policy/filename).write_bytes(raw)
            actor=load_policy(policy,PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8))
            if actor.weights_sha256!=point['weights_sha256']:raise ValueError('Policy identity differs')
            persist()
            path='frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+actor.weights_sha256+'/development/result.json'
            durable=Path('/results')/path
            reused=durable.exists()
            if not reused:
                try:evaluator.remote((policy/'manifest.json').read_bytes(),pack_weights((policy/'weights.json').read_bytes()),'development','L40S',time.time(),100)
                except Exception as exc:summary.setdefault('evaluation_transport_errors',[]).append(str(type(exc).__name__)+': '+str(exc))
                modal.Volume.from_name('automanip-modal-validation').reload()
            if not durable.exists():raise RuntimeError('No durable score; no repeat')
            result=json.loads(durable.read_bytes())
            if result['state']!='complete' or result['weights_sha256']!=actor.weights_sha256:raise ValueError('Frozen report differs')
            summary['development_curve'].append(dict(steps=steps,trainer_seconds=point['elapsed_seconds'],
                weights_sha256=actor.weights_sha256,successes=result['report']['successes_at_end'],episodes=128,
                diagnostics_rates=result['report']['diagnostics_rates'],report_path=path,reused_existing_report=reused));persist()
        primary=next((p for p in summary['development_curve'] if p['steps']==102400),None)
        if primary is None:verdict='primary_missing'
        else:
            d=primary['diagnostics_rates'];grasp=round(128*d['grasp']);lift=round(128*d['lift'])
            verdict='supports_discount_sufficiency' if grasp>=64 and lift>=32 else 'weakens_discount_sufficiency' if grasp<=8 and lift<=8 else 'partial_or_inconclusive'
        summary.update(state='complete',primary_verdict=verdict);persist()
    except Exception as exc:summary.update(state='failed',error=type(exc).__name__+': '+str(exc));persist();raise
    return dict(state='complete',summary_path=str((out/'summary.json').relative_to('/results')))


@app.local_entrypoint()
def launch():
    import time
    raw=PLAN.read_bytes();plan=json.loads(raw)
    if not plan['admitted']:raise PermissionError('Default denied')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['adapter_sha256']:raise ValueError('Adapter drift')
    for name,expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/'research/discovery'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source drift')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:raise ValueError('Protocol drift')
    budget=json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd'])>100:raise PermissionError('Budget exceeded')
    if not any(x['campaign']=='gamma-sufficiency-20261028' and float(x['reserved_cap_usd'])==2 for x in budget['reservations']):raise PermissionError('No reservation')
    if time.time()+1100>plan['window_deadline_epoch']:raise RuntimeError('Insufficient window')
    p=ROOT/'.runtime/modal-gamma-sufficiency-launch.json'
    if p.exists():raise RuntimeError('Already submitted; no repeat')
    worker=train_gamma;worker.hydrate()
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=1);evaluator.hydrate()
    call=coordinate_gamma.spawn(worker,evaluator,hashlib.sha256(raw).hexdigest())
    record=dict(function_call_id=call.object_id,plan_sha256=hashlib.sha256(raw).hexdigest(),submitted_at=time.time(),
        candidate_volume='automanip-gamma-sufficiency-20261028',summary_path='gamma-sufficiency-20261028-seed-1/summary.json',budget_cap_usd=2)
    with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
