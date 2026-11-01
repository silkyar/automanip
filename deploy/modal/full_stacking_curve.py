"""Fresh-seed full-task study; supervisor never reads candidate artifacts."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app, ROOT, trainer_image, validation_image, results

PLAN = ROOT/'deploy/modal/full-stacking-curve-plan.json'
SOURCE_MAP = {'full_stacking_curve.py': 'full_stacking_curve.py', 'replay_robot.py': 'replay_robot_full_curve.py',
              'replay_learner.py': 'replay_learner_lowalpha.py', 'sac_baseline.py': 'sac_baseline.py',
              'joint_exploration.py': 'joint_exploration.py'}
image = trainer_image.add_local_file(PLAN, '/opt/route/plan.json', copy=True)
for target, source in SOURCE_MAP.items():
    image = image.add_local_file(ROOT/'research/discovery'/source, '/opt/route/'+target, copy=True)
coordinator_image = validation_image.add_local_file(PLAN, '/opt/route-plan.json', copy=True)
outputs = modal.Volume.from_name('automanip-full-stacking-curve-20261029', create_if_missing=True, version=2)


@app.function(image=image, gpu='L40S', cpu=(3, 4), memory=(16384, 16384), timeout=660,
              startup_timeout=30, retries=0, max_containers=2, scaledown_window=2,
              single_use_containers=True, restrict_modal_access=True, block_network=True,
              include_source=False, serialized=True, _experimental_restrict_output=True,
              volumes={'/candidate-output': outputs,
                  '/root/.sapien': results.with_mount_options(read_only=True, sub_path='/runtime/sapien-cache')})
def train_route(route: str, plan_sha: str, expires_at: float):
    import os
    import signal
    import subprocess
    import time
    import traceback

    mark = time.monotonic()
    if route not in ('baseline', 'coordinate'): raise ValueError('Unknown route')
    root = Path('/candidate-output')/route
    root.mkdir(exist_ok=False)
    with (root/'attempt.json').open('x') as stream:
        json.dump(dict(plan_sha256=plan_sha, expires_at=expires_at, route=route), stream)
    subprocess.run(['sync', str(root)], check=True, timeout=10)
    record = dict(state='started', route=route, plan_sha256=plan_sha)
    try:
        raw = Path('/opt/route/plan.json').read_bytes()
        plan = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != plan_sha or not plan['admitted']:
            raise PermissionError('Exact route experiment not admitted')
        if Path('/results').exists() or Path('/opt/automanip').exists(): raise RuntimeError('Trainer isolation differs')
        for name, expected in plan['source_sha256'].items():
            if hashlib.sha256((Path('/opt/route')/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reviewed source differs')
        import torch
        if torch.cuda.get_device_name(0) != 'NVIDIA L40S': raise ValueError('GPU differs')
        cache = Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest() != '4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':
            raise ValueError('PhysX cache differs')
        left = min(600., expires_at-time.time()-15)
        if left < 45: raise RuntimeError('Original lease expired; no refresh')
        child_env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL_') and k != 'RUNPOD_API_KEY'}
        child_env['AUTOMANIP_TRAINING_DEADLINE'] = str(time.monotonic()+left-15)
        with (root/'training.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '/opt/route/full_stacking_curve.py', '--route', route,
                '--output', str(root/'run')], cwd='/opt/route', env=child_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child_start = time.monotonic()
            try:
                code = process.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError('Hard trainer cap reached')
        record['trainer_process_seconds'] = time.monotonic()-child_start
        if code: raise RuntimeError('Trainer failed; preserve log')
        report = json.loads((root/'run/report.json').read_bytes())
        if report['state'] != 'complete' or report['counts']['transitions'] > 1024000: raise ValueError('Invalid training report')
        expected_core = {name: plan['source_sha256'][name] for name in ('replay_robot.py', 'replay_learner.py', 'sac_baseline.py', 'joint_exploration.py')}
        if report['source_files_sha256'] != expected_core: raise ValueError('Core source differs')
        admission = report['admission']
        if admission['route'] != route or admission['launcher_sha256'] != plan['source_sha256']['full_stacking_curve.py']:
            raise ValueError('Launcher or route differs')
        if admission['collector_sha256'] != plan['source_sha256']['joint_exploration.py']:
            raise ValueError('Collector source differs')
        initial = next(point for point in report['checkpoint_exports'] if point['transitions'] == 0)
        if initial['weights_sha256'] != plan['initial_actor_sha256']: raise ValueError('Paired actor initialization differs')
        
        if report['config']['seed'] != 2 or report['study_arm'] != route or report['behavior_epsilon'] != (0. if route == 'baseline' else .1):
            raise ValueError('Seed or behavior differs')
        record.update(state='complete', report=report)
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
    finally:
        record['wrapper_seconds'] = time.monotonic()-mark
        (root/'receipt.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        inventory = []
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError('Symlink output refused')
            if path.is_file() and path.name != 'inventory.json':
                data = path.read_bytes()
                if len(data) > 32*1024*1024: raise ValueError('Individual output bound')
                inventory.append(dict(path=str(path.relative_to(root)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if sum(row['bytes'] for row in inventory) > 128*1024*1024: raise ValueError('Aggregate output bound')
        (root/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
        subprocess.run(['sync', str(root)], check=True, timeout=10)
    return dict(state=record['state'], route=route, error=record.get('error'))


@app.function(image=coordinator_image, cpu=1, memory=2048, timeout=900, retries=0,
              max_containers=1, single_use_containers=True, include_source=False,
              serialized=True, _experimental_restrict_output=True,
              volumes={'/results': results})
def coordinate_routes(worker: modal.Function, plan_sha: str):
    import time
    import modal
    raw = Path('/opt/route-plan.json').read_bytes()
    plan = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != plan_sha or not plan['admitted']: raise PermissionError('Not admitted')
    experiment = 'full-stacking-curve-20261029'
    claims = modal.Dict.from_name('automanip-modal-scientific-job-claims', create_if_missing=True)
    if not claims.put(experiment, dict(started=time.time(), plan_sha256=plan_sha), skip_if_exists=True):
        raise RuntimeError('Attempt already claimed')
    out = Path('/results')/experiment
    out.mkdir(exist_ok=False)
    summary = dict(state='started', experiment=experiment, plan_sha256=plan_sha, arms={}, confirmation_queries=0,
                   note='Scalar worker outcomes only. External archive validation precedes local frozen evaluation.')
    def persist():
        temporary = out/'summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        temporary.replace(out/'summary.json')
        modal.Volume.from_name('automanip-modal-validation').commit()
    calls = {}
    persist()
    for route in ('baseline', 'coordinate'):
        call = worker.spawn(route, plan_sha, time.time()+645)
        calls[route] = call
        summary['arms'][route] = dict(state='submitted', worker_call_id=call.object_id)
        persist()
    for route, call in calls.items():
        try:
            outcome = call.get()
            if outcome['route'] != route: raise ValueError('Route identity differs')
            summary['arms'][route].update(state=outcome['state'], outcome=outcome)
        except Exception as exc:
            summary['arms'][route].update(state='transport_failed', error=type(exc).__name__+': '+str(exc))
        persist()
    summary['state'] = 'complete' if all(a['state']=='complete' for a in summary['arms'].values()) else 'partial_or_failed'
    persist()
    return dict(state=summary['state'], summary_path='full-stacking-curve-20261029/summary.json')


@app.local_entrypoint()
def launch():
    import time
    raw = PLAN.read_bytes(); plan = json.loads(raw)
    if not plan['admitted']: raise PermissionError('Default denied')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan['adapter_sha256']: raise ValueError('Adapter drift')
    for target, source in SOURCE_MAP.items():
        if hashlib.sha256((ROOT/'research/discovery'/source).read_bytes()).hexdigest() != plan['source_sha256'][target]:
            raise ValueError('Source drift')
    for local, expected in [(plan['scientific_plan'], plan['scientific_plan_sha256']),
                            ('.runtime/full-stacking-curve-CPU-gate.json', plan['cpu_gate_sha256'])]:
        if hashlib.sha256((ROOT/local).read_bytes()).hexdigest() != expected: raise ValueError('Protocol/gate drift')
    budget = json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd']) > 100:
        raise PermissionError('Shared cap exceeded')
    if time.time()-budget['observed_at'] > 300: raise PermissionError('Budget ledger stale')
    if not any(r['campaign']=='full-stacking-curve-20261029' and float(r['reserved_cap_usd'])==3 for r in budget['reservations']):
        raise PermissionError('No exact reservation')
    launch_path = ROOT/'.runtime/modal-full-stacking-curve-launch.json'
    with launch_path.open('x') as stream:
        json.dump(dict(state='submitting', plan_sha256=hashlib.sha256(raw).hexdigest(), submitted_at=time.time()), stream)
    worker = train_route; worker.hydrate()
    call = coordinate_routes.spawn(worker, hashlib.sha256(raw).hexdigest())
    record = dict(function_call_id=call.object_id, plan_sha256=hashlib.sha256(raw).hexdigest(), submitted_at=time.time(),
                  candidate_volume='automanip-full-stacking-curve-20261029',
                  summary_path='full-stacking-curve-20261029/summary.json', budget_cap_usd=3)
    launch_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
