"""One root-admitted, bounded public diagnostic. No training or frozen scoring."""
import hashlib,json
from pathlib import Path
import sys
import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,results

INPUTS=ROOT/'.runtime/gradient-noise-inputs'
PLAN=INPUTS/'plan.json'
image=trainer_image.add_local_dir(INPUTS,'/opt/noise/inputs',copy=True)
for name in ('continuation_noise_robot.py','continuation_variance.py','sac_baseline.py'):
    image=image.add_local_file(ROOT/'research/discovery'/name,'/opt/noise/'+name,copy=True)
outputs=modal.Volume.from_name('automanip-continuation-noise-20261028',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=550,startup_timeout=30,
    retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/diagnostic-output':outputs,
        '/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def probe(plan_sha:str,expires_at:float):
    import hashlib,json,os,signal,subprocess,sys,time,traceback
    from pathlib import Path
    root=Path('/diagnostic-output');mark=time.monotonic()
    with (root/'attempt.json').open('x') as f:json.dump(dict(plan_sha256=plan_sha,expires_at=expires_at),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    record=dict(state='started',plan_sha256=plan_sha,new_training_runs=0,protected_evaluations=0)
    try:
        raw=Path('/opt/noise/inputs/plan.json').read_bytes();plan=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=plan_sha or not plan['admitted']:raise PermissionError('Exact diagnostic not admitted')
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('Diagnostic isolation differs')
        for name,expected in plan['source_sha256'].items():
            if hashlib.sha256((Path('/opt/noise')/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source differs')
        left=min(500.,expires_at-time.time()-15)
        if left<45:raise RuntimeError('Original lease expired; no refresh')
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_DIAGNOSTIC_DEADLINE']=str(time.monotonic()+left-15)
        with (root/'diagnostic.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'/opt/noise/continuation_noise_robot.py',
                '--inputs','/opt/noise/inputs','--output',str(root/'run')],cwd='/opt/noise',
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Diagnostic hard deadline reached')
        if code:raise RuntimeError('Diagnostic child failed; preserve report/log')
        report=json.loads((root/'run/report.json').read_bytes())
        if report['state']!='complete' or report['actual_public_transitions']!=409600 or report['optimizer_calls']!=0:raise RuntimeError('Incomplete or invalid diagnostic')
        record.update(state='complete',actual_public_transitions=report['actual_public_transitions'])
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds']=time.monotonic()-mark
        (root/'wrapper-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        inventory=[]
        for path in sorted(root.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink output refused')
            if path.is_file() and path.name!='inventory.json':
                raw=path.read_bytes()
                if len(raw)>16*1024*1024:raise ValueError('Individual output bound')
                inventory.append(dict(path=str(path.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if sum(x['bytes'] for x in inventory)>32*1024*1024:raise ValueError('Aggregate output bound')
        (root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=10)
    return record


@app.local_entrypoint()
def launch():
    import time
    raw=PLAN.read_bytes();plan=json.loads(raw)
    if not plan['admitted']:raise PermissionError('No root admission')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['adapter_sha256']:raise ValueError('Adapter drift')
    for name,expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/'research/discovery'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source drift')
    if hashlib.sha256((ROOT/plan['plan_path']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:raise ValueError('Scientific protocol drift')
    for policy in plan['policies']:
        if hashlib.sha256((INPUTS/policy['filename']).read_bytes()).hexdigest()!=policy['sha256']:raise ValueError('Policy drift')
    budget=json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd'])>100:raise PermissionError('Budget exceeded')
    if not any(r['campaign']=='continuation-noise-20261028' and float(r['reserved_cap_usd'])==2 for r in budget['reservations']):raise PermissionError('No diagnostic reservation')
    if time.time()+600>plan['window_deadline_epoch']:raise PermissionError('Insufficient window')
    launch_path=ROOT/'.runtime/modal-continuation-noise-launch.json'
    if launch_path.exists():raise RuntimeError('Attempt already submitted; read receipt, no repeat')
    call=probe.spawn(hashlib.sha256(raw).hexdigest(),time.time()+520)
    record=dict(function_call_id=call.object_id,plan_sha256=hashlib.sha256(raw).hexdigest(),
        volume='automanip-continuation-noise-20261028',max_public_transitions=409600,
        new_training_runs=0,protected_evaluations=0,budget_cap_usd=2,submitted_at=time.time())
    with launch_path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
