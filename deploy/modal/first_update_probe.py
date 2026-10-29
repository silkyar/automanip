"""Single approved mechanism probe. No benchmark scorer or training-loop handle."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import ROOT, trainer_image, results

EXPERIMENT='stackcube-first-update-probe-20261028'
PLAN=ROOT/'deploy/modal/first-update-probe-plan.json'
PRODUCTION=ROOT/'research/discovery/bc-residual-exploration-v2/assist-bc-ppo-residual-std.py'
ARCHIVE=ROOT/'runs/stackcube-bc-exploration-pilot-20261028'
INITIAL=ARCHIVE/'assist-bc-ppo-residual-std/native/work/runs/stackcube-bc-exploration-pilot-20261028-assist-bc-ppo-residual-std-seed-1/ckpt_1.pt'
image=trainer_image
for name in ('first_update_probe.py','run_first_update_probe.py'):
    image=image.add_local_file(ROOT/'research/discovery'/name,'/opt/probe/'+name,copy=True)
image=image.add_local_file(PRODUCTION,'/opt/probe/production.py',copy=True)
image=image.add_local_file(INITIAL,'/opt/probe/initial.pt',copy=True)
image=image.add_local_file(PLAN,'/opt/probe/plan.json',copy=True)
for name in ('manifest.json','weights.json'):
    image=image.add_local_file(ARCHIVE/'bc-only'/name,'/opt/probe/bc/'+name,copy=True)

app=modal.App('automanip-first-update-probe',include_source=False)
output=modal.Volume.from_name('automanip-first-update-probe-20261028',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),
    timeout=550,startup_timeout=30,retries=0,max_containers=1,scaledown_window=2,
    restrict_modal_access=True,block_network=True,single_use_containers=True,
    include_source=False,serialized=True,_experimental_restrict_output=True,
    volumes={'/probe-output':output,
        '/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def run_probe(expected_plan_sha:str,expires_at:float):
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    import traceback
    root=Path('/probe-output')
    # Durable exclusive marker forbids infrastructure replay of the scientific run.
    with (root/'attempt.json').open('x') as stream:
        json.dump(dict(admitted_at=time.time(),expires_at=expires_at,plan_sha256=expected_plan_sha),stream)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    started=time.monotonic()
    record=dict(state='started',experiment_id='stackcube-first-update-probe-20261028',
        expires_at=expires_at,plan_sha256=expected_plan_sha,new_training_runs=0,
        new_development_queries=0,new_confirmation_queries=0)
    try:
        raw=Path('/opt/probe/plan.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_plan_sha:raise ValueError('Reviewed plan changed')
        plan=json.loads(raw)
        if not plan['launch_enabled'] or plan['approval_scope']!='one first-update mechanism probe only; stop and reassess':
            raise PermissionError('Probe not admitted')
        for path,digest in plan['image_files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Reviewed file differs: '+path)
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('Candidate isolation failed')
        remaining=min(510.,expires_at-time.time()-20)
        if remaining<30:raise RuntimeError('Original lease expired; no refresh')
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        deadline=time.monotonic()+remaining-10
        command=[sys.executable,'/opt/probe/run_first_update_probe.py','--production','/opt/probe/production.py',
            '--initial','/opt/probe/initial.pt','--bc-actor','/opt/probe/bc','--output',str(root/'run'),
            '--deadline',str(deadline)]
        with (root/'probe.log').open('w') as stream:
            process=subprocess.Popen(command,env=env,cwd='/opt/probe',stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Original hard probe lease exceeded')
        record['return_code']=code
        if code:raise RuntimeError('Mechanism probe failed; inspect durable log, no automatic retry')
        report=json.loads((root/'run/report.json').read_bytes())
        if report['state']!='complete' or report['progress']['new_simulator_transitions']>201600:
            raise RuntimeError('Incomplete or over-budget probe')
        record.update(state='complete',progress=report['progress'])
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        record['container_function_seconds']=time.monotonic()-started
        files=[]
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.name not in ('inventory.json','receipt.json'):
                files.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (root/'inventory.json').write_text(json.dumps(files,indent=2)+'\n')
        (root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=10)
    return {k:v for k,v in record.items() if k!='traceback'}


@app.local_entrypoint()
def launch_first_update_probe():
    import time
    raw=PLAN.read_bytes();plan=json.loads(raw)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['adapter_sha256']:
        raise ValueError('Reviewed Modal adapter differs')
    preflight=(ROOT/'.runtime/first-update-cpu-preflight.json').read_bytes()
    if hashlib.sha256(preflight).hexdigest()!=plan['cpu_preflight_sha256'] or json.loads(preflight)['state']!='passed':
        raise PermissionError('Exact CPU reconstruction prerequisite missing')
    if not plan['launch_enabled'] or plan['maximum_gpu_workers']!=1 or plan['budget_cap_usd']!=1:
        raise PermissionError('Exact reviewed one-worker scope required')
    ledger=json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not ledger['within_shared_cap'] or float(ledger['committed_worst_case_plus_pending_reservations_usd'])>100:
        raise PermissionError('Shared $100 budget exceeded')
    for local,remote in plan['local_to_image'].items():
        if hashlib.sha256((ROOT/local).read_bytes()).hexdigest()!=plan['image_files'][remote]:
            raise ValueError('Reviewed local file differs: '+local)
    launch=ROOT/'.runtime/modal-first-update-probe-launch.json'
    if launch.exists():raise RuntimeError('Probe already submitted; read durable result, do not relaunch')
    dispatched=time.time()
    call=run_probe.spawn(hashlib.sha256(raw).hexdigest(),dispatched+580)
    record=dict(experiment_id=EXPERIMENT,function_call_id=call.object_id,dispatched_at=dispatched,
        expires_at=dispatched+580,plan_sha256=hashlib.sha256(raw).hexdigest(),
        output_volume='automanip-first-update-probe-20261028',gpu='L40S',maximum_gpu_workers=1,
        budget_cap_usd=1,maximum_new_transitions=201600)
    with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(record))
