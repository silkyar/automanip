"""Single bounded branching diagnostic. No training or private evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import ROOT, validation_image, results

app = modal.App('automanip-branching-validity-r2-20261030', include_source=False)
inputs = ROOT/'.runtime/branching-validity-r2-20261030'
image = validation_image
for name in ('branching_validity', 'branching_validity_probe'):
    image = image.add_local_file(ROOT/f'research/{name}.py', f'/opt/automanip/research/{name}.py', copy=True)
image = image.add_local_dir(inputs/'inputs', '/opt/calibration-inputs', copy=True)
image = image.add_local_file(inputs/'preflight-admission.json', '/opt/cumulative-preflight-admission.json', copy=True)
output_volume = modal.Volume.from_name('automanip-branching-validity-r2-20261030', create_if_missing=True, version=2)


@app.function(image=image, gpu='L40S', cpu=(3, 4), memory=(16384, 16384), timeout=300,
              startup_timeout=60, retries=0, max_containers=1, single_use_containers=True,
              scaledown_window=2, restrict_modal_access=True, block_network=True,
              include_source=False, serialized=True, _experimental_restrict_output=True,
              volumes={'/output': output_volume,
                       '/root/.sapien': results.with_mount_options(read_only=True, sub_path='/runtime/sapien-cache')})
def preflight(plan_sha, expires_at):
    import os
    import signal
    import subprocess
    import time
    import traceback
    root = Path('/output/diagnostic')
    root.mkdir(exist_ok=True)
    started = time.monotonic()
    with (root/'attempt.json').open('x') as f:
        json.dump(dict(plan_sha256=plan_sha, started_at=time.time()), f)
    subprocess.run(['sync', str(root)], check=True, timeout=10)
    record = dict(state='started', optimizer_calls=0, confirmation_queries=0, final_queries=0)
    try:
        raw = Path('/opt/cumulative-preflight-admission.json').read_bytes()
        admission = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != plan_sha or admission['execution_authorized'] is not True:
            raise PermissionError('Exact diagnostic not admitted')
        if hashlib.sha256(Path('/opt/calibration-inputs/plan.json').read_bytes()).hexdigest() != admission['input_plan_sha256']:
            raise ValueError('Input protocol drift')
        for name, digest in admission['sources'].items():
            if hashlib.sha256((Path('/opt/automanip')/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Source drift: '+name)
        left = min(260., expires_at-time.time()-15)
        if left < 60:
            raise RuntimeError('Submission lease expired; no automatic renewal')
        child_env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL_') and k != 'RUNPOD_API_KEY'}
        child_env['PYTHONPATH'] = '/opt/automanip/src:/opt/automanip'
        with (root/'preflight.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '/opt/automanip/research/branching_validity_probe.py',
                                        '--inputs', '/opt/calibration-inputs', '--output', str(root/'run')],
                                       env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd='/opt/automanip')
            try:
                code = process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError('Hard diagnostic deadline')
        if code:
            raise RuntimeError('Diagnostic child failed; see durable report/log')
        record.update(state='complete', report='run/report.json')
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds'] = time.monotonic()-started
        (root/'wrapper-receipt.json').write_text(json.dumps(record, indent=2)+'\n')
        subprocess.run(['sync', str(root)], check=True, timeout=10)
    return {k: v for k, v in record.items() if k != 'traceback'}


@app.local_entrypoint()
def launch():
    import time
    raw = (inputs/'preflight-admission.json').read_bytes()
    admission = json.loads(raw)
    if not admission['execution_authorized']:
        raise PermissionError('Not admitted')
    for name, digest in admission['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Local source drift: '+name)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != admission['adapter_sha256']:
        raise ValueError('Adapter drift')
    destination = inputs/'preflight-launch.json'
    with destination.open('x') as f:
        json.dump(dict(state='submitting', submitted_at=time.time()), f)
    call = preflight.spawn(hashlib.sha256(raw).hexdigest(), time.time()+900)
    record = dict(state='submitted', calls=[dict(function_call_id=call.object_id, submitted_at=time.time())],
                  volume='automanip-branching-validity-r2-20261030', max_public_transitions=94720, optimizer_calls=0)
    destination.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
