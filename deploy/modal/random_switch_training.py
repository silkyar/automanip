"""Bounded training-only corrective outcome collection, no evaluator image."""
import hashlib
import json
from pathlib import Path
import sys
import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import ROOT,trainer_image,results

app=modal.App('automanip-random-switch-training-20261030',include_source=False)
inputs=ROOT/'.runtime/random-switch-training-20261030'
image=trainer_image.add_local_dir(inputs/'bundle','/opt/collector',copy=True)
image=image.add_local_dir(inputs/'inputs','/opt/collector-inputs',copy=True)
image=image.add_local_file(inputs/'admission.json','/opt/collection-admission.json',copy=True)
volume=modal.Volume.from_name('automanip-random-switch-training-20261030',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=240,startup_timeout=60,
    retries=0,max_containers=2,single_use_containers=True,scaledown_window=2,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/output':volume,'/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def collect(admission_sha,expires_at,index):
    import os,signal,subprocess,time,traceback
    if type(index) is not int or index not in [0,1]:raise ValueError('Closedsourceindex')
    root=Path('/output')/f'source-{index}';root.mkdir(exist_ok=True);started=time.monotonic()
    with (root/'attempt.json').open('x') as f:json.dump(dict(started_at=time.time(),admission_sha256=admission_sha),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10);record=dict(state='started',optimizer_calls=0)
    process=None
    try:
        raw=Path('/opt/collection-admission.json').read_bytes();admission=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=admission_sha or not admission['execution_authorized']:raise PermissionError('Exactadmissionmissing')
        if hashlib.sha256(Path('/opt/collector-inputs/plan.json').read_bytes()).hexdigest()!=admission['plan_sha256']:raise ValueError('Protocol drift')
        for path,digest in admission['bundle_files'].items():
            if hashlib.sha256((Path('/opt/collector')/path).read_bytes()).hexdigest()!=digest:raise ValueError('Collector drift')
        for forbidden in ['/opt/automanip','/opt/collector/automanip/evaluator.py','/opt/collector/benchmarks']:
            if Path(forbidden).exists():raise ValueError('Evaluator/benchmark present in training image')
        left=min(200.,expires_at-time.time()-15)
        if left<60:raise RuntimeError('Submission leaseexpired; noautorenewal')
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'};env['PYTHONPATH']='/opt/collector'
        with (root/'collection.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'/opt/collector/collect.py','--inputs','/opt/collector-inputs','--output',str(root/'run'),'--index',str(index)],cwd='/opt/collector',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=left)
            except subprocess.TimeoutExpired:raise RuntimeError('Hardcollectiondeadline')
        if code:raise RuntimeError('Collectorfailed; durablereportretained')
        record.update(state='complete',report='run/report.json')
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        record['wrapper_seconds']=time.monotonic()-started
        (root/'wrapper-receipt.json').write_text(json.dumps(record,indent=2)+'\n');subprocess.run(['sync',str(root)],check=True,timeout=10)
    return {k:v for k,v in record.items() if k!='traceback'}


@app.local_entrypoint()
def launch():
    import time
    raw=(inputs/'admission.json').read_bytes();admission=json.loads(raw)
    if not admission['execution_authorized'] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=admission['adapter_sha256']:raise ValueError('Admission/adapter differs')
    for path,digest in admission['bundle_files'].items():
        if hashlib.sha256((inputs/'bundle'/path).read_bytes()).hexdigest()!=digest:raise ValueError('Local bundle drift')
    path=inputs/'launch.json'
    with path.open('x') as f:json.dump(dict(state='submitting',submitted_at=time.time()),f)
    calls=[]
    for index in [0,1]:
        call=collect.spawn(hashlib.sha256(raw).hexdigest(),time.time()+900,index);calls.append(dict(index=index,function_call_id=call.object_id,submitted_at=time.time()))
        path.write_text(json.dumps(dict(state='submitted',calls=calls,volume='automanip-random-switch-training-20261030',max_transitions=204800,optimizer_calls=0),indent=2)+'\n')
    print(json.dumps(dict(state='submitted',calls=calls)))
