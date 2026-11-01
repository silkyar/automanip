"""One reviewed 800-transition SAC interface smoke, never a learning benchmark."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app, ROOT, trainer_image, validation_image, results
from smoke_capg import runtime_mount
from screen_contract import read_regular

SOURCE_SHA = '1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca'
RUN_ID = 'sac-interface-smoke-v2-20261027'
output = modal.Volume.from_name('automanip-modal-sac-smoke-v2-20261027', create_if_missing=True, version=2)
helper = ROOT / 'deploy/modal/screen_contract.py'
candidate_image = trainer_image.add_local_file(helper, '/opt/screen/screen_contract.py', copy=True)
trusted_image = validation_image.add_local_file(helper, '/opt/screen/screen_contract.py', copy=True)


@app.function(image=candidate_image, gpu='L40S', cpu=(3,4), memory=(16384,16384),
              timeout=160, startup_timeout=30, retries=0, max_containers=1,
              restrict_modal_access=True, block_network=True, single_use_containers=True,
              include_source=False, serialized=True, _experimental_restrict_output=True,
              volumes={**runtime_mount, '/trainer-output': output})
def sac_train_smoke_v2(program: bytes, expires_at: float):
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    import torch
    sys.path.insert(0, '/opt/screen')
    from screen_contract import read_regular
    if hashlib.sha256(program).hexdigest() != '1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca':
        raise ValueError('unreviewed SAC source')
    root = Path('/trainer-output')
    with (root / 'attempt.json').open('x') as stream:
        json.dump({'admitted_at':time.time(), 'expires_at':expires_at}, stream)
    subprocess.run(['sync', str(root)], check=True, timeout=10)
    record = {'state':'started', 'source_sha256':hashlib.sha256(program).hexdigest(),
              'gpu_actual_name':torch.cuda.get_device_name(0), 'training_repeated':False,
              'performance_result':False, 'confirmation_queries':0}
    try:
        if Path('/results').exists() or Path('/opt/automanip').exists():
            raise RuntimeError('candidate isolation failed')
        remaining = min(120., expires_at - time.time())
        if remaining <= 20:
            raise RuntimeError('original smoke lease expired')
        source = root / 'trainer.py'
        source.write_bytes(program)
        env = {k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k != 'RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE'] = str(time.monotonic() + remaining - 20)
        command = [sys.executable, str(source), '--seed','1','--device','cuda','--num-envs','8',
                   '--total-transitions','800','--buffer-size','512','--batch-size','32',
                   '--learning-starts','64','--training-freq','8','--utd','0.125',
                   '--gamma','0.9','--max-seconds','120','--checkpoint-every','400',
                   '--output',str(root / 'run')]
        started = time.monotonic()
        with (root / 'train.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError('120-second original smoke lease exceeded')
        record.update(trainer_process_seconds=time.monotonic()-started, return_code=code)
        if code:
            raise RuntimeError('SAC smoke subprocess failed')
        report = json.loads(read_regular(root, 'run/report.json', 1024*1024))
        expected = {'transitions':800, 'vector_step_calls':100, 'completed_episodes':16,
                    'resets':24, 'reset_calls':3, 'critic_optimizer_steps':93,
                    'actor_optimizer_steps':93, 'alpha_optimizer_steps':93,
                    'minibatch_presentations':2976, 'replay_size':512}
        if any(report.get(k) != v for k,v in expected.items()):
            raise RuntimeError('smoke accounting disagrees with prespecified contract')
        record.update(state='complete', report=report)
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__+': '+str(exc))
    finally:
        (root / 'gpu-record.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        subprocess.run(['sync', str(root)], check=True, timeout=20)
    return {'state':record['state'], 'record_path':'gpu-record.json'}


@app.function(image=trusted_image, cpu=1, memory=2048, timeout=1800, retries=0,
              single_use_containers=True, include_source=False, serialized=True,
              _experimental_restrict_output=True,
              volumes={'/candidate':output.with_mount_options(read_only=True), '/results':results})
def coordinate_sac_smoke_v2(program: bytes, trainer: modal.Function, revision: str, recovery: bool = False):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    import numpy as np
    sys.path.insert(0, '/opt/screen')
    sys.path.insert(0, '/opt/automanip/src')
    from screen_contract import read_regular
    from automanip.policy_io import load_policy, PolicySchema
    from automanip.numpy_inference import NumpyPolicy
    if hashlib.sha256(program).hexdigest() != '1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca':
        raise ValueError('unreviewed SAC source')
    run_id = 'sac-interface-smoke-v2-20261027'
    if recovery:
        # Explicit root-reviewed recovery: original caller could not dispatch
        # through its blocked regional input-plane connection. No GPU child
        # appears in its call graph, and its output Volume is entirely empty.
        # Keep its original failed summary and claim, never renew an actual
        # training attempt or erase an attempt marker.
        previous = json.loads(read_regular('/results', run_id+'/summary.json', 1024*1024))
        if previous['state'] != 'failed' or previous.get('training') is not None:
            raise RuntimeError('recovery requires original pre-dispatch failure')
        if list(Path('/candidate').iterdir()):
            raise RuntimeError('prior GPU output exists; recovery must not rerun training')
        run_id = 'sac-interface-smoke-recovery1-20261027'
    claims = modal.Dict.from_name('automanip-modal-scientific-job-claims', create_if_missing=True)
    if not claims.put(run_id, {'started_at':time.time(), 'root_reviewed_recovery':recovery}, skip_if_exists=True):
        raise RuntimeError('smoke already claimed; never rerun silently')
    destination = Path('/results') / run_id
    destination.mkdir(exist_ok=False)
    result = {'state':'started', 'gpu_training_transitions':0, 'performance_result':False,
              'confirmation_queries':0, 'evaluation_episodes':0, 'git_sha':revision,
              'budget_cap_usd':.20, 'source_sha256':hashlib.sha256(program).hexdigest(),
              'root_reviewed_recovery':recovery, 'training_repeated':False}
    try:
        error = None
        try:
            trainer.remote(program, time.time()+120)
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
            result['receipt_transport_error'] = error
        modal.Volume.from_name('automanip-modal-sac-smoke-v2-20261027').reload()
        record = json.loads(read_regular('/candidate', 'gpu-record.json', 1024*1024))
        result['training'] = record
        if error:
            result['receipt_transport_error'] = error
        if record['state'] != 'complete':
            raise RuntimeError(record.get('error', 'smoke incomplete'))
        schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8)
        checks = []
        states = np.random.default_rng(192).normal(size=(16,48))
        for label in ('actor-step-0','actor-step-400','actor-step-800','actor-final'):
            target = destination / label
            target.mkdir()
            for filename, limit in (('manifest.json',8192),('weights.json',32*1024*1024)):
                raw = read_regular('/candidate', 'run/'+label+'/'+filename, limit)
                (target / filename).write_bytes(raw)
            artifact = load_policy(target, schema)
            action = np.asarray(NumpyPolicy(artifact, schema).act(states))
            if not np.isfinite(action).all() or np.any(np.abs(action)>1):
                raise RuntimeError('invalid trusted actor actions')
            checks.append({'label':label, 'weights_sha256':artifact.weights_sha256})
        if checks[-1]['weights_sha256'] != checks[-2]['weights_sha256']:
            raise RuntimeError('coincident final export changed weights')
        result.update(state='complete', gpu_training_transitions=800, trusted_artifact_checks=checks)
    except Exception as exc:
        result.update(state='failed', error=type(exc).__name__+': '+str(exc))
    finally:
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    return {'state':result['state'], 'summary_path':run_id+'/summary.json'}


@app.local_entrypoint()
def sac_smoke_v2():
    import subprocess
    source = (ROOT / 'research/discovery/sac_baseline.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError('reviewed source changed')
    destination = ROOT / '.runtime/modal-sac-smoke-v2-launch.json'
    if destination.exists():
        raise RuntimeError('already submitted; inspect durable result')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    call = coordinate_sac_smoke_v2.spawn(source, sac_train_smoke_v2, revision)
    receipt = {'run_id':RUN_ID, 'function_call_id':call.object_id,
               'summary_path':RUN_ID+'/summary.json', 'source_sha256':SOURCE_SHA,
               'budget_cap_usd':.20, 'performance_result':False}
    destination.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
