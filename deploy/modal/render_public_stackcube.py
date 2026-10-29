"""One bounded public illustration, not a protected benchmark evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,validation_image,results

SOURCE=ROOT/'research/discovery/render_stackcube_success.py'
POLICY=ROOT/'.runtime/continuation-reference-seed1/policy'
RENDER_SHA='12ca0b4e128935630d737704c1a2f5d5ab2f23b3b99db07fd47c68cd88d3f78e'
image=validation_image.add_local_file(SOURCE,'/opt/render/render.py',copy=True)
for name in ('manifest.json','weights.json'):
    image=image.add_local_file(POLICY/name,'/opt/render/policy/'+name,copy=True)
output=modal.Volume.from_name('automanip-public-stackcube-video-20261028',create_if_missing=True,version=2)


@app.function(image=image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=140,startup_timeout=30,
    retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/render-output':output,
        '/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def render_success(expires_at:float):
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    root=Path('/render-output')
    with (root/'attempt.json').open('x') as f:json.dump(dict(started_at=time.time(),expires_at=expires_at),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    if hashlib.sha256(Path('/opt/render/render.py').read_bytes()).hexdigest()!='12ca0b4e128935630d737704c1a2f5d5ab2f23b3b99db07fd47c68cd88d3f78e':
        raise ValueError('Reviewed renderer differs')
    left=min(110.,expires_at-time.time()-10)
    if left<30:raise RuntimeError('Original render lease expired; no refresh')
    env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_')}
    env.update(PYTHONPATH='/opt/automanip/src',AUTOMANIP_RENDER_DEADLINE=str(time.monotonic()+left-10))
    mark=time.monotonic()
    with (root/'render.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'/opt/render/render.py','--policy','/opt/render/policy',
            '--output',str(root/'video')],cwd='/opt/render',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();raise RuntimeError('Render hard cap reached')
    record=dict(state='complete' if code==0 else 'failed',return_code=code,process_seconds=time.monotonic()-mark,
        maximum_public_transitions=400,new_training_runs=0,protected_evaluations=0)
    (root/'wrapper-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    inventory=[]
    for path in sorted(root.rglob('*')):
        if path.is_file():inventory.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    return record


@app.local_entrypoint()
def launch_public_video():
    import time
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=RENDER_SHA:raise ValueError('Source drift')
    launch=ROOT/'.runtime/modal-public-stackcube-video-launch.json'
    if launch.exists():raise RuntimeError('Public batch already submitted; no resampling')
    manifest=json.loads((POLICY/'manifest.json').read_bytes())
    digest=hashlib.sha256((POLICY/'weights.json').read_bytes()).hexdigest()
    if manifest['weights_sha256']!=digest or digest!='39ba198d0c98af14a3d2965ef9a3be859f31aa3c4889db3da311da910c2da153':raise ValueError('Wrong25M actor')
    call=render_success.spawn(time.time()+130)
    record=dict(function_call_id=call.object_id,source_sha256=RENDER_SHA,policy_sha256=digest,
        volume='automanip-public-stackcube-video-20261028',public_episodes=8,max_public_transitions=400,
        new_training_runs=0,protected_evaluations=0,budget_cap_usd=1)
    with launch.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
