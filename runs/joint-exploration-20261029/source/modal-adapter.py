"""Two exact joint-exploration interventions; original independent scoring reused."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app, ROOT, trainer_image, validation_image, results
from evaluate import frozen_evaluate

PLAN = ROOT/'deploy/modal/joint-exploration-plan.json'
SOURCE_MAP = {'joint_exploration_train.py': 'joint_exploration_train.py', 'replay_robot.py': 'replay_robot_exploration.py',
              'replay_learner.py': 'replay_learner_lowalpha.py', 'sac_baseline.py': 'sac_baseline.py',
              'joint_exploration.py': 'joint_exploration.py'}
image = trainer_image.add_local_file(PLAN, '/opt/route/plan.json', copy=True)
for target, source in SOURCE_MAP.items():
    image = image.add_local_file(ROOT/'research/discovery'/source, '/opt/route/'+target, copy=True)
coordinator_image = validation_image.add_local_file(PLAN, '/opt/route-plan.json', copy=True)
outputs = modal.Volume.from_name('automanip-joint-exploration-20261029', create_if_missing=True, version=2)


@app.function(image=image, gpu='L40S', cpu=(3, 4), memory=(16384, 16384), timeout=210,
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
    if route not in ('joint', 'coordinate'): raise ValueError('Unknown route')
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
        left = min(150., expires_at-time.time()-15)
        if left < 45: raise RuntimeError('Original lease expired; no refresh')
        child_env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL_') and k != 'RUNPOD_API_KEY'}
        child_env['AUTOMANIP_TRAINING_DEADLINE'] = str(time.monotonic()+left-15)
        with (root/'training.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '/opt/route/joint_exploration_train.py', '--route', route,
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
        if report['state'] != 'complete' or report['counts']['transitions'] > 102400: raise ValueError('Invalid training report')
        expected_core = {name: plan['source_sha256'][name] for name in ('replay_robot.py', 'replay_learner.py', 'sac_baseline.py', 'joint_exploration.py')}
        if report['source_files_sha256'] != expected_core: raise ValueError('Core source differs')
        admission = report['admission']
        if admission['route'] != route or admission['launcher_sha256'] != plan['source_sha256']['joint_exploration_train.py']:
            raise ValueError('Launcher or route differs')
        if admission['collector_sha256'] != plan['source_sha256']['joint_exploration.py']:
            raise ValueError('Collector source differs')
        initial = next(point for point in report['checkpoint_exports'] if point['transitions'] == 0)
        if initial['weights_sha256'] != plan['initial_actor_sha256']: raise ValueError('Paired actor initialization differs')
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
def coordinate_routes(worker: modal.Function, evaluator: modal.Function, plan_sha: str):
    import time
    import modal
    sys.path.insert(0, '/opt/automanip/src')
    sys.path.insert(0, '/opt/automanip-modal')
    from automanip.policy_io import load_policy, PolicySchema
    from transport import pack_weights
    raw = Path('/opt/route-plan.json').read_bytes()
    plan = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != plan_sha or not plan['admitted']: raise PermissionError('Not admitted')
    experiment = 'joint-exploration-20261029'
    claims = modal.Dict.from_name('automanip-modal-scientific-job-claims', create_if_missing=True)
    if not claims.put(experiment, dict(started=time.time(), plan_sha256=plan_sha), skip_if_exists=True):
        raise RuntimeError('Attempt already claimed')
    out = Path('/results')/experiment
    out.mkdir(exist_ok=False)
    summary = dict(state='started', experiment=experiment, plan_sha256=plan_sha, arms={}, confirmation_queries=0)

    def persist():
        p = out/'summary.json'; temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        temporary.replace(p)
        modal.Volume.from_name('automanip-modal-validation').commit()

    persist()
    candidate_volume = modal.Volume.from_name('automanip-joint-exploration-20261029')
    trusted_volume = modal.Volume.from_name('automanip-modal-validation')

    def read_file(volume, name, limit, attempts=1):
        # Retry only file availability, never a scientific job/evaluation.
        for attempt in range(attempts):
            try:
                data = bytearray()
                for block in volume.read_file(name):
                    data.extend(block)
                    if len(data) > limit: raise ValueError('Bounded artifact read exceeded')
                return bytes(data)
            except FileNotFoundError:
                if attempt+1 == attempts: raise
                summary.setdefault('file_visibility_retries', []).append(dict(path=name, retry=attempt+1))
                time.sleep(.2)
    calls = {}
    for route in ('joint', 'coordinate'):
        calls[route] = worker.spawn(route, plan_sha, time.time()+195)
        summary['arms'][route] = dict(state='submitted', worker_call_id=calls[route].object_id)
    persist()
    schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f', 48, 8)
    for route, call in calls.items():
        arm = summary['arms'][route]
        try:
            try: call.get()
            except Exception as exc: arm['transport_error'] = type(exc).__name__+': '+str(exc)
            receipt = json.loads(read_file(candidate_volume, route+'/receipt.json', 2*1024*1024, attempts=5))
            arm['training'] = receipt
            if receipt['state'] != 'complete': raise RuntimeError('Worker failed: '+str(receipt.get('error')))
            inventory = {row['path']: row for row in json.loads(read_file(candidate_volume, route+'/inventory.json', 512000, attempts=5))}

            def copy_policy(label, destination):
                destination.mkdir()
                for filename in ('manifest.json', 'weights.json'):
                    name = 'run/'+label+'/'+filename
                    entry = inventory[name]
                    data = read_file(candidate_volume, route+'/'+name, min(entry['bytes'], 32*1024*1024), attempts=5)
                    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                        raise ValueError('Artifact inventory differs')
                    (destination/filename).write_bytes(data)
                return load_policy(destination, schema)

            initial = copy_policy('actor-step-0', out/(route+'-initial'))
            if initial.weights_sha256 != plan['initial_actor_sha256']: raise ValueError('Initial safe policy differs')
            arm['initial_safe_actor_validated'] = True
            point = next((point for point in receipt['report']['checkpoint_exports']
                          if point['transitions'] == 102400 and point['label'] == 'actor-step-102400'), None)
            if point is None:
                arm.update(state='primary_missing', primary_verdict='missing_or_censored')
                persist()
                continue
            policy = out/(route+'-102400')
            actor = copy_policy(point['label'], policy)
            if actor.weights_sha256 != point['weights_sha256']: raise ValueError('Policy identity differs')
            path = 'frozen-evaluation-attempts/fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764/'+actor.weights_sha256+'/development/result.json'
            try: score_raw = read_file(trusted_volume, path, 4*1024*1024)
            except FileNotFoundError: score_raw = None
            reused = score_raw is not None
            persist()
            if not reused:
                try:
                    evaluator.remote((policy/'manifest.json').read_bytes(), pack_weights((policy/'weights.json').read_bytes()),
                                     'development', 'L40S', time.time(), 100)
                except Exception as exc: arm['evaluation_transport_error'] = type(exc).__name__+': '+str(exc)
                try: score_raw = read_file(trusted_volume, path, 4*1024*1024, attempts=5)
                except FileNotFoundError: raise RuntimeError('No durable score; no repeat')
            score = json.loads(score_raw)
            if score['state'] != 'complete' or score['weights_sha256'] != actor.weights_sha256:
                raise ValueError('Frozen report differs')
            rate = score['report']['diagnostics_rates']
            grasp = round(128*rate['grasp'])
            arm.update(state='complete', primary_verdict='grasp_lift_support' if grasp >= 64 and round(128*rate['lift']) >= 32 else 'stage_gate_not_met',
                development=dict(steps=102400, trainer_seconds=point['elapsed_seconds'], weights_sha256=actor.weights_sha256,
                    successes=score['report']['successes_at_end'], episodes=128, diagnostics_rates=rate,
                    report_path=path, reused_existing_report=reused))
        except Exception as exc:
            arm.update(state='failed', error=type(exc).__name__+': '+str(exc))
        persist()
    summary['state'] = 'complete' if all(arm['state'] == 'complete' for arm in summary['arms'].values()) else 'partial_or_failed'
    if summary['state'] == 'complete':
        joint = summary['arms']['joint']['development']['diagnostics_rates']
        coordinate = summary['arms']['coordinate']['development']['diagnostics_rates']
        gj, lj, lc = round(128*joint['grasp']), round(128*joint['lift']), round(128*coordinate['lift'])
        summary['primary_verdict'] = 'promising_joint_exploration' if gj >= 64 and lj >= 32 and lj-lc >= 20 else 'weakens_joint_sufficiency' if lj <= 8 else 'partial_or_inconclusive'
    else: summary['primary_verdict'] = 'missing_primary'
    persist()
    return dict(state=summary['state'], summary_path=str((out/'summary.json').relative_to('/results')))


@app.local_entrypoint()
def launch():
    import time
    raw = PLAN.read_bytes(); plan = json.loads(raw)
    if not plan['admitted']: raise PermissionError('Default denied')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan['adapter_sha256']: raise ValueError('Adapter drift')
    for target, source in SOURCE_MAP.items():
        if hashlib.sha256((ROOT/'research/discovery'/source).read_bytes()).hexdigest() != plan['source_sha256'][target]:
            raise ValueError('Source drift')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest() != plan['scientific_plan_sha256']:
        raise ValueError('Protocol drift')
    budget = json.loads((ROOT/'.runtime/modal-tonight-budget-ledger.json').read_bytes())
    if not budget['within_shared_cap'] or float(budget['committed_worst_case_plus_pending_reservations_usd']) > 100:
        raise PermissionError('Shared cap exceeded')
    if time.time()-budget['observed_at'] > 300: raise PermissionError('Budget ledger stale')
    if not any(row['campaign'] == 'joint-exploration-20261029' and float(row['reserved_cap_usd']) == 2 for row in budget['reservations']):
        raise PermissionError('No exact campaign reservation')
    launch_path = ROOT/'.runtime/modal-joint-exploration-launch.json'
    with launch_path.open('x') as stream:
        json.dump(dict(state='submitting', plan_sha256=hashlib.sha256(raw).hexdigest(), submitted_at=time.time()), stream)
    worker = train_route; worker.hydrate()
    evaluator = frozen_evaluate.with_options(gpu='L40S', timeout=120, max_containers=1); evaluator.hydrate()
    call = coordinate_routes.spawn(worker, evaluator, hashlib.sha256(raw).hexdigest())
    record = dict(function_call_id=call.object_id, plan_sha256=hashlib.sha256(raw).hexdigest(), submitted_at=time.time(),
                  candidate_volume='automanip-joint-exploration-20261029', summary_path='joint-exploration-20261029/summary.json', budget_cap_usd=2)
    launch_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))
