"""One bounded, registered instantaneous-impulse recovery probe; no learner or final split."""
import hashlib
import json
from pathlib import Path
import sys
import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import ROOT,validation_image,results

app=modal.App('automanip-recovery-impulse-probe-20261030',include_source=False)
inputs=ROOT/'.runtime/recovery-impulse-probe-20261030'
image=validation_image.add_local_file(ROOT/'src/automanip/cumulative.py','/opt/automanip/src/automanip/cumulative.py',copy=True)
image=image.add_local_file(ROOT/'research/recovery_impulse_probe.py','/opt/automanip/research/recovery_impulse_probe.py',copy=True)
image=image.add_local_file(ROOT/'research/discovery/amp_v1.py','/opt/automanip/research/discovery/amp_v1.py',copy=True)
image=image.add_local_file(ROOT/'research/recovery_impulse.py','/opt/automanip/research/recovery_impulse.py',copy=True)
source=ROOT/'runs/stackcube-bc-exploration-pilot-20261028/teacher-input'
image=image.add_local_dir(source,'/opt/automanip/runs/stackcube-bc-exploration-pilot-20261028/teacher-input',copy=True)
image=image.add_local_file(inputs/'preflight-admission.json','/opt/cumulative-preflight-admission.json',copy=True)
output_volume=modal.Volume.from_name('automanip-recovery-impulse-probe-20261030',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=300,
    startup_timeout=60,retries=0,max_containers=1,single_use_containers=True,
    scaledown_window=2,restrict_modal_access=True,block_network=True,
    include_source=False,serialized=True,_experimental_restrict_output=True,
    volumes={'/output':output_volume,'/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def preflight(plan_sha,expires_at):
    import os,signal,subprocess,time,traceback
    root=Path('/output');started=time.monotonic()
    with (root/'attempt.json').open('x') as f:json.dump(dict(plan_sha256=plan_sha,started_at=time.time()),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    record=dict(state='started',training_updates=0,final_episodes=0)
    try:
        raw=Path('/opt/cumulative-preflight-admission.json').read_bytes();plan=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=plan_sha or plan['execution_authorized'] is not True:raise PermissionError('No exact preflight admission')
        for name,digest in plan['sources'].items():
            if hashlib.sha256((Path('/opt/automanip')/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source drift: '+name)
        left=min(260.,expires_at-time.time()-15)
        if left<60:raise RuntimeError('Submission lease expired; no automatic renewal')
        child_env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        child_env['PYTHONPATH']='/opt/automanip/src:/opt/automanip'
        with (root/'preflight.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'/opt/automanip/research/recovery_impulse_probe.py','--output',str(root/'run')],env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/opt/automanip')
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Hard preflight deadline')
        if code:raise RuntimeError('Preflight child failed; see durable report/log')
        record.update(state='complete',report='run/report.json')
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds']=time.monotonic()-started
        (root/'wrapper-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=10)
    return {k:v for k,v in record.items() if k!='traceback'}


@app.local_entrypoint()
def launch():
    import time
    plan=json.loads((inputs/'preflight-admission.json').read_bytes())
    if not plan['execution_authorized']:raise PermissionError('Not admitted')
    for name,digest in plan['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Local source drift')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['adapter_sha256']:raise ValueError('Adapter drift')
    path=inputs/'preflight-launch.json'
    with path.open('x') as f:json.dump(dict(state='submitting',submitted_at=time.time()),f)
    call=preflight.spawn(hashlib.sha256((inputs/'preflight-admission.json').read_bytes()).hexdigest(),time.time()+600)
    record=dict(state='submitted',function_call_id=call.object_id,volume='automanip-recovery-impulse-probe-20261030',submitted_at=time.time(),max_gpu_seconds=300,max_public_transitions=13600,training_updates=0,final_episodes=0)
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
