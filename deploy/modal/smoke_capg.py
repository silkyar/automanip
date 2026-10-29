"""One reviewed 400-transition CAPG smoke; deliberately not a full-trial launcher."""
import hashlib
import json
from pathlib import Path
import sys
import uuid

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,common,trainer_image,validation_image,results,ROOT,STAGE
from bundle import build_context

CAPG_SHA='8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971'
converter_context=build_context('converter',STAGE/'converter')
converter_image=common.add_local_dir(converter_context,'/opt/converter',copy=True)
runtime_mount={'/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')}


@app.function(image=trainer_image,gpu='L4',cpu=(3,4),memory=(16384,16384),timeout=120,retries=0,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,volumes=runtime_mount,
              _experimental_restrict_output=True)
def train_smoke(program:bytes,expires_at:float):
    import hashlib
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    if hashlib.sha256(program).hexdigest()!='8a95fca4ac08a8118ecb361fb9a458e137578673a1b4fcfe2c94fdbd3033b971':
        raise ValueError('only the reviewed CAPG smoke payload is allowed')
    if len(program)>128*1024:raise ValueError('trainer size limit')
    remaining=min(120.,expires_at-time.time())
    if remaining<20:raise RuntimeError('submission deadline expired before trainer started')
    with tempfile.TemporaryDirectory() as temporary:
        work=Path(temporary);trainer=work/'train.py';trainer.write_bytes(program)
        command=[sys.executable,str(trainer),'--env-id','StackCube-v1','--seed','1',
                 '--num-envs','8','--num-eval-envs','8','--num-minibatches','2','--update-epochs','8',
                 '--total-timesteps','400','--exp-name','modal-capg-smoke',
                 '--gamma','0.9','--ent-coef','0.003','--learning-rate','0.0003',
                 '--no-partial-reset','--no-capture-video','--no-track']
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(time.monotonic()+remaining-20)
        started=time.monotonic()
        with (work/'train.log').open('w') as log:
            process=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('hard 120-second smoke budget exceeded')
        seconds=time.monotonic()-started
        if code:raise RuntimeError('trainer failed: '+(work/'train.log').read_text()[-5000:])
        checkpoint=work/'runs/modal-capg-smoke/final_ckpt.pt'
        if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size>64*1024*1024:
            raise ValueError('bounded regular final checkpoint required')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(checkpoint.parent),size_guidance={'scalars':0});events.Reload()
        transitions=int(events.Scalars('time/step')[-1].value)
        if transitions!=400:raise RuntimeError('smoke transition count incomplete')
        raw=checkpoint.read_bytes()
        return {'checkpoint':raw,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
                'trainer_process_seconds':seconds,'environment_transitions':transitions,
                'submission_expires_at':expires_at,'container_task_id':os.environ.get('MODAL_TASK_ID'),
                'trainer_sha256':hashlib.sha256(program).hexdigest(),'training_log':(work/'train.log').read_text()}


@app.function(image=converter_image,cpu=1,memory=2048,timeout=120,retries=0,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def convert_checkpoint(checkpoint:bytes):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    if len(checkpoint)>64*1024*1024:raise ValueError('checkpoint byte limit')
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary);native=path/'checkpoint.pt';native.write_bytes(checkpoint)
        env=dict(os.environ,PYTHONPATH='/opt/converter/src')
        result=subprocess.run([sys.executable,'/opt/converter/scripts/export_official_policy.py',str(native),str(path/'policy')],
                              env=env,capture_output=True,text=True,timeout=90)
        if result.returncode:raise RuntimeError(result.stderr[-3000:])
        sys.path.insert(0,'/opt/automanip-modal')
        from transport import pack_weights
        return {'manifest':(path/'policy/manifest.json').read_bytes(),'weights_gzip':pack_weights((path/'policy/weights.json').read_bytes()),
                'export':json.loads(result.stdout)}


@app.function(image=validation_image,gpu='L4',cpu=(3,4),memory=(16384,16384),timeout=180,retries=0,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,volumes=runtime_mount,
              _experimental_restrict_output=True)
def evaluate_smoke(manifest:bytes,weights_gzip:bytes):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    weights=unpack_weights(weights_gzip)
    if len(manifest)>8192 or len(weights)>32*1024*1024:raise ValueError('policy byte limit')
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary);policy=path/'policy';policy.mkdir()
        (policy/'manifest.json').write_bytes(manifest);(policy/'weights.json').write_bytes(weights)
        env=dict(os.environ,PYTHONPATH='/opt/automanip/src')
        result=subprocess.run([sys.executable,'/opt/automanip/scripts/evaluate_stackcube.py',str(policy),
                               '--split','development','--smoke','--output',str(path/'report.json')],
                              env=env,capture_output=True,text=True,timeout=150)
        if result.returncode:raise RuntimeError(result.stderr[-4000:])
        report=json.loads((path/'report.json').read_text())
        if report['episode_count']!=8 or not report['smoke'] or report['split']!='development' or report['failed_episodes']:
            raise RuntimeError('invalid smoke evaluation')
        return report


coordinator_image=modal.Image.debian_slim(python_version='3.11').add_local_file(ROOT/'deploy/modal/transport.py','/opt/automanip-modal/transport.py',copy=True)


@app.function(image=coordinator_image,cpu=.125,memory=512,
              timeout=900,retries=0,single_use_containers=True,include_source=False,
              serialized=True,volumes={'/results':results},_experimental_restrict_output=True)
def coordinate_capg(run_id:str,validation_id:str,program:bytes,train_function:modal.Function,
                    converter_function:modal.Function,evaluator_function:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import time
    import modal
    import sys
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    if not run_id.isalnum() or not validation_id.isalnum():raise ValueError('invalid run ID')
    proof=json.loads((Path('/results')/validation_id/'coordinator-summary.json').read_text())
    if not proof['passed']:raise RuntimeError('physics, parity and restricted-boundary proof required first')
    claims=modal.Dict.from_name('automanip-modal-validation-claims')
    if not claims.put('capg-'+run_id,{'started_at':time.time()},skip_if_exists=True):raise RuntimeError('attempt already claimed')
    out=Path('/results')/run_id;out.mkdir(exist_ok=False)
    summary={'run_id':run_id,'passed':False,'mode':'400-transition smoke, not a scientific result',
             'confirmation_queries':0,'validation_id':validation_id,'trainer_seconds_cap':120}
    try:
        # Stable absolute deadline survives automatic preemption retries without a fresh budget.
        native=train_function.remote(program,time.time()+120)
        raw=native.pop('checkpoint')
        if hashlib.sha256(raw).hexdigest()!=native['checkpoint_sha256']:raise RuntimeError('checkpoint digest mismatch')
        (out/'checkpoint.pt').write_bytes(raw);(out/'train.log').write_text(native.pop('training_log'))
        summary['training']=native
        converted=converter_function.remote(raw)
        policy=out/'policy';policy.mkdir();(policy/'manifest.json').write_bytes(converted['manifest']);(policy/'weights.json').write_bytes(unpack_weights(converted['weights_gzip']))
        summary['export']=converted['export']
        evaluation=evaluator_function.remote(converted['manifest'],converted['weights_gzip'])
        summary.update(passed=True,evaluation=evaluation)
    except Exception as exc:summary['error']=type(exc).__name__+': '+str(exc)
    finally:
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    return {'run_id':run_id,'passed':summary['passed'],'summary_path':run_id+'/summary.json'}


@app.function(image=coordinator_image,cpu=.125,memory=512,timeout=600,retries=0,
              single_use_containers=True,include_source=False,serialized=True,
              volumes={'/results':results},_experimental_restrict_output=True)
def recover_transport(run_id:str,converter_function:modal.Function,evaluator_function:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    if not run_id.isalnum():raise ValueError('invalid run ID')
    claims=modal.Dict.from_name('automanip-modal-validation-claims')
    if not claims.put('transport-recovery-'+run_id,{'started_at':time.time()},skip_if_exists=True):
        raise RuntimeError('transport recovery already attempted')
    out=Path('/results')/run_id;original=json.loads((out/'summary.json').read_text())
    summary=dict(original,passed=False,transport_recovery=True,training_repeated=False)
    try:
        if original['training']['environment_transitions']!=400:raise ValueError('completed smoke required')
        raw=(out/'checkpoint.pt').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=original['training']['checkpoint_sha256']:raise ValueError('checkpoint changed')
        converted=converter_function.remote(raw)
        policy=out/'policy';policy.mkdir(exist_ok=True)
        (policy/'manifest.json').write_bytes(converted['manifest']);(policy/'weights.json').write_bytes(unpack_weights(converted['weights_gzip']))
        summary['export']=converted['export']
        summary['evaluation']=evaluator_function.remote(converted['manifest'],converted['weights_gzip'])
        summary.update(passed=True,error=None)
    except Exception as exc:summary['error']=type(exc).__name__+': '+str(exc)
    finally:
        (out/'recovery-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    return {'run_id':run_id,'passed':summary['passed'],'summary_path':run_id+'/recovery-summary.json'}


@app.local_entrypoint()
def capg_smoke(validation_id:str):
    program=(ROOT/'.runtime/modal-capg-payloads/full-horizon-capg.py').read_bytes()
    if hashlib.sha256(program).hexdigest()!=CAPG_SHA:raise ValueError('reviewed CAPG source changed')
    run_id='capgsmoke'+uuid.uuid4().hex[:20]
    call=coordinate_capg.spawn(run_id,validation_id,program,train_smoke,convert_checkpoint,evaluate_smoke)
    launch={'run_id':run_id,'function_call_id':call.object_id,'results_volume':'automanip-modal-validation',
            'summary_path':run_id+'/summary.json'}
    destination=ROOT/'.runtime'/('modal-'+run_id+'-launch.json');destination.write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))


@app.local_entrypoint()
def finish_smoke(run_id:str):
    call=recover_transport.spawn(run_id,convert_checkpoint,evaluate_smoke)
    launch={'run_id':run_id,'function_call_id':call.object_id,'training_repeated':False,
            'summary_path':run_id+'/recovery-summary.json'}
    (ROOT/'.runtime'/('modal-'+run_id+'-recovery-launch.json')).write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))
