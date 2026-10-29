"""One bounded public mechanical probe; excludes evaluator and training access."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app, ROOT, trainer_image, results

INPUTS = ROOT/'.runtime/lift-control-inputs'
PLAN = INPUTS/'plan.json'
image = trainer_image.add_local_dir(INPUTS, '/opt/lift/inputs', copy=True)
for name in ('lift_control_robot.py', 'panda_kinematics.py', 'lift_probe_control.py'):
    image = image.add_local_file(ROOT/'research/discovery'/name, '/opt/lift/'+name, copy=True)
for name in ('__init__.py', 'policy_io.py', 'numpy_inference.py'):
    image = image.add_local_file(ROOT/'src/automanip'/name, '/opt/safe/automanip/'+name, copy=True)
outputs = modal.Volume.from_name('automanip-lift-control-20261028', create_if_missing=True, version=2)


@app.function(image=image, gpu='L40S', cpu=(3, 4), memory=(16384, 16384),
              timeout=330, startup_timeout=30, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True, restrict_modal_access=True,
              block_network=True, include_source=False, serialized=True,
              _experimental_restrict_output=True, volumes={'/diagnostic-output': outputs,
                  '/root/.sapien': results.with_mount_options(read_only=True, sub_path='/runtime/sapien-cache')})
def probe(plan_sha: str, expires_at: float):
    import os
    import signal
    import subprocess
    import time
    import traceback

    root = Path('/diagnostic-output')
    mark = time.monotonic()
    with (root/'attempt.json').open('x') as stream:
        json.dump(dict(plan_sha256=plan_sha, expires_at=expires_at), stream)
    subprocess.run(['sync', str(root)], check=True, timeout=10)
    record = dict(state='started', plan_sha256=plan_sha, new_training_runs=0, protected_evaluations=0)
    try:
        raw = Path('/opt/lift/inputs/plan.json').read_bytes()
        plan = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != plan_sha or not plan['admitted']:
            raise PermissionError('Exact diagnostic not admitted')
        if Path('/results').exists() or Path('/opt/automanip').exists():
            raise RuntimeError('Diagnostic isolation differs')
        for name, expected in plan['source_sha256'].items():
            if hashlib.sha256((Path('/opt/lift')/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reviewed source differs')
        for name, expected in plan['safe_policy_source_sha256'].items():
            if hashlib.sha256((Path('/opt/safe/automanip')/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Safe policy reader source differs')
        left = min(270., expires_at-time.time()-15)
        if left < 45: raise RuntimeError('Original lease expired; no refresh')
        child_env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL_') and k != 'RUNPOD_API_KEY'}
        child_env['PYTHONPATH'] = '/opt/safe'
        child_env['AUTOMANIP_DIAGNOSTIC_DEADLINE'] = str(time.monotonic()+left-5)
        with (root/'diagnostic.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '/opt/lift/lift_control_robot.py',
                '--inputs', '/opt/lift/inputs', '--output', str(root/'run')], cwd='/opt/lift',
                env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError('Diagnostic hard deadline reached')
        if code: raise RuntimeError('Diagnostic child failed; preserve log and report')
        result = json.loads((root/'run/report.json').read_bytes())
        if result['state'] != 'complete' or result['actual_public_transitions'] != 1600 or result['optimizer_calls'] != 0:
            raise RuntimeError('Incomplete diagnostic')
        record.update(state='complete', actual_public_transitions=1600)
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds'] = time.monotonic()-mark
        (root/'wrapper-receipt.json').write_text(json.dumps(record, indent=2)+'\n')
        inventory = []
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError('Symlink output refused')
            if path.is_file() and path.name != 'inventory.json':
                raw = path.read_bytes()
                if len(raw) > 16*1024*1024: raise ValueError('Individual output bound')
                inventory.append(dict(path=str(path.relative_to(root)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        if sum(item['bytes'] for item in inventory) > 32*1024*1024: raise ValueError('Aggregate output bound')
        (root/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
        subprocess.run(['sync', str(root)], check=True, timeout=10)
    return record


@app.local_entrypoint()
def launch():
    import time
    raw = PLAN.read_bytes()
    plan = json.loads(raw)
    if not plan['admitted']: raise PermissionError('No root admission')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan['adapter_sha256']:
        raise ValueError('Adapter drift')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/'research/discovery'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source drift')
    for name, expected in plan['safe_policy_source_sha256'].items():
        if hashlib.sha256((ROOT/'src/automanip'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Safe policy reader drift')
    if hashlib.sha256((ROOT/plan['plan_path']).read_bytes()).hexdigest() != plan['scientific_plan_sha256']:
        raise ValueError('Protocol drift')
    for name, expected in plan['policy_files_sha256'].items():
        if hashlib.sha256((INPUTS/'policy'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Policy drift')
    budget = json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd']) > 100:
        raise PermissionError('Budget exceeded')
    if time.time()-budget['observed_at'] > 300: raise PermissionError('Budget ledger stale')
    if not any(r['campaign'] == 'lift-control-20261028' and float(r['reserved_cap_usd']) == 1 for r in budget['reservations']):
        raise PermissionError('No diagnostic reservation')
    launch_path = ROOT/'.runtime/modal-lift-control-launch.json'
    with launch_path.open('x') as stream:
        json.dump(dict(state='submitting', plan_sha256=hashlib.sha256(raw).hexdigest(),
                       submitted_at=time.time(), retry_permitted=False), stream, indent=2)
    call = probe.spawn(hashlib.sha256(raw).hexdigest(), time.time()+310)
    record = dict(function_call_id=call.object_id, plan_sha256=hashlib.sha256(raw).hexdigest(),
                  volume='automanip-lift-control-20261028', max_public_transitions=1600,
                  new_training_runs=0, protected_evaluations=0, budget_cap_usd=1, submitted_at=time.time())
    launch_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
