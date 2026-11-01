"""One bounded actuator-prior mechanism test; training and scoring isolated."""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import ROOT, trainer_image, validation_image, results

NAME='automanip-gripper-hold-20261031'
RUNTIME=ROOT/'.runtime/gripper-hold-20261031'
INPUTS=RUNTIME/'inputs'
app=modal.App(NAME,include_source=False)
training_image=trainer_image.add_local_dir(INPUTS,'/opt/credit',copy=True)
scores_image=validation_image.add_local_file(RUNTIME/'score-plan.json','/opt/score-plan.json',copy=True)
training_outputs=modal.Volume.from_name(NAME,create_if_missing=True,version=2)
score_outputs=modal.Volume.from_name(NAME+'-scores',create_if_missing=True,version=2)
cache=results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')


@app.function(image=training_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=360,
    startup_timeout=60,retries=0,max_containers=3,single_use_containers=True,scaledown_window=2,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/output':training_outputs,'/root/.sapien':cache})
def train_trial(index,expected_plan_sha,expires_at):
    import importlib.metadata
    import os
    import signal
    import subprocess
    import time
    import traceback

    if type(index) is not int or not 0<=index<3:raise ValueError('Closed trial index')
    root=Path('/output')/f'trial-{index}';root.mkdir(exist_ok=True)
    started=time.monotonic();record=dict(state='started',index=index,training_initialization='from_scratch',protected_queries=0)
    with (root/'attempt.json').open('x') as f:json.dump(dict(plan_sha256=expected_plan_sha,admitted_at=time.time()),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    try:
        raw=Path('/opt/credit/training-plan.json').read_bytes();plan=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=expected_plan_sha or not plan['execution_authorized']:raise ValueError('Exact training admission missing')
        if Path('/opt/automanip').exists() or Path('/results').exists():raise RuntimeError('Trainer contains scorer filesystem')
        record['versions']={name:importlib.metadata.version(name) for name in plan['production_pins']}
        if record['versions']!=plan['production_pins']:raise ValueError('Pinned versions differ')
        import torch
        record['gpu']=torch.cuda.get_device_name(0)
        if record['gpu']!='NVIDIA L40S':raise ValueError('Hardware differs')
        blob=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        record['physx_sha256']=hashlib.sha256(blob.read_bytes()).hexdigest()
        if record['physx_sha256']!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':raise ValueError('PhysX pin differs')
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!='ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440':raise ValueError('Task source differs')
        methods=['full','masked','hold'];method=methods[index];seed=41
        record.update(method=method,seed=seed)
        source=Path('/opt/credit')/(method+'.py')
        for filename,digest in plan['files_sha256'].items():
            if hashlib.sha256((Path('/opt/credit')/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Training input differs: '+filename)
        remaining=min(300.,expires_at-time.time()-15)
        if remaining<60:raise RuntimeError('Original lease expired; no refreshed allowance')
        work=root/'work';work.mkdir(exist_ok=False)
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(time.monotonic()+remaining-8)
        env['PYTHONPATH']='/opt/credit'
        run_name=f'gripper-hold-{method}-seed-{seed}'
        command=[sys.executable,str(source),'--env-id','StackCube-v1','--seed',str(seed),'--num-envs','1024',
                 '--num-eval-envs','8','--num-minibatches','32','--update-epochs','8','--total-timesteps','4096000',
                 '--exp-name',run_name,'--gamma','0.9','--ent-coef','0.003','--learning-rate','0.0003',
                 '--no-partial-reset','--no-capture-video','--no-track','--no-save-model']
        record['command']=command;record['trainer_started_at']=time.time();tick=time.monotonic()
        with (root/'train.log').open('x') as log:
            process=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Hard300s trainer process limit')
        record['trainer_process_seconds']=time.monotonic()-tick
        if code:raise RuntimeError('Trainer failed; preserve logs and actual-work journal')
        record.update(state='complete',run_name=run_name)
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        journal=list(root.glob('work/runs/*/work.jsonl'))
        if len(journal)==1:
            events=[]
            for line in journal[0].read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:record['partial_final_journal_line']=True
            record['actual_work']={kind:max((e['count'] for e in events if e['event']==kind),default=0)
                for kind in ['environment_reserved','environment_completed','optimizer_reserved','optimizer_completed']}
            record['checkpoints']=[e for e in events if e['event']=='actor_export']
            record['training_finished']=any(e['event']=='training_complete' for e in events)
            if record['state']=='complete' and (not record['training_finished'] or record['actual_work']['environment_completed']>4096000):
                record.update(state='failed',error='Incomplete or over-ceiling journal')
        record['wrapper_seconds']=time.monotonic()-started
        (root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=20)
    return {k:v for k,v in record.items() if k not in ['traceback','command','checkpoints']}


@app.function(image=scores_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=180,
    startup_timeout=60,retries=0,max_containers=3,single_use_containers=True,scaledown_window=2,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/candidates':training_outputs.with_mount_options(read_only=True),'/scores':score_outputs,'/root/.sapien':cache})
def score_trial(index,score_plan_sha):
    from dataclasses import asdict
    import subprocess
    import time
    sys.path.insert(0,'/opt/automanip/src')
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.stackcube import STACK_TASK,check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter

    if type(index) is not int or not 0<=index<3:raise ValueError('Closed trial index')
    root=Path('/scores')/f'trial-{index}';root.mkdir(exist_ok=True)
    with (root/'attempt.json').open('x') as f:json.dump(dict(index=index,started_at=time.time()),f)
    subprocess.run(['sync',str(root)],check=True,timeout=10)
    raw=Path('/opt/score-plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=score_plan_sha:raise ValueError('Exact score admission missing')
    plan=json.loads(raw);receipt=json.loads((Path('/candidates')/f'trial-{index}/receipt.json').read_text())
    if receipt['state']!='complete':raise ValueError('Training incomplete; do not convert execution failure to success0')
    seed=41;method=['full','masked','hold'][index]
    if receipt['seed']!=seed or receipt['method']!=method:raise ValueError('Trial identity mismatch')
    record=dict(state='started',index=index,seed=seed,method=method,points=[],environment_reserved=0,environment_completed=0,
                final_queries=0,confirmation_queries=0,partition='public-development',protocol_sha256=check_frozen_protocol())
    started=time.monotonic();env=None
    journal=(root/'work.jsonl').open('x',buffering=1)
    def persist():
        record['wall_seconds']=time.monotonic()-started
        temporary=root/'report.tmp';temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(root/'report.json')
    try:
        env=StackCubeAdapter(128,'physx_cuda');schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        class CountedPhysics:
            def __init__(self,base):self.base=base
            def __getattr__(self,name):return getattr(self.base,name)
            def step(self,actions):
                if record['environment_reserved']+128>32000:raise ValueError('Scoring transition ceiling')
                record['environment_reserved']+=128
                journal.write(json.dumps(dict(event='environment_reserved',count=record['environment_reserved']))+'\n')
                result=self.base.step(actions)
                record['environment_completed']+=128
                journal.write(json.dumps(dict(event='environment_completed',count=record['environment_completed']))+'\n')
                return result
        env.env=CountedPhysics(env.env)
        for checkpoint in receipt['checkpoints']:
            steps=checkpoint['steps']
            if type(steps) is not int or steps<0 or steps>receipt['actual_work']['environment_completed']:raise ValueError('Uncompleted checkpoint')
            path=Path('/candidates')/f'trial-{index}/work/runs/gripper-hold-{method}-seed-{seed}/actors'/str(steps)
            actor=load_policy(path,schema)
            if actor.weights_sha256!=checkpoint['actor']['weights_sha256']:raise ValueError('Candidate checkpoint changed')
            base=plan['development_seed_base']
            persist()
            rollouts=evaluate_batch(env,NumpyPolicy(actor,schema),list(range(base,base+128)),horizon=50,action_dim=8)
            point=dict(steps=steps,trainer_seconds=checkpoint['trainer_seconds'],weights_sha256=actor.weights_sha256,
                       episodes=128,successes=sum(r.outcome.success_at_end for r in rollouts),
                       failures=sum(r.outcome.failed for r in rollouts),outcomes=[asdict(r) for r in rollouts])
            record['points'].append(point);persist()
            if point['failures']:raise ValueError('Scorer execution failure; retain complete outcomes')
        record['state']='complete'
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if env is not None:env.close()
        journal.close();persist();subprocess.run(['sync',str(root)],check=True,timeout=10)
    return dict(index=index,state=record['state'],points=len(record['points']),environment_completed=record['environment_completed'])


@app.local_entrypoint()
def launch():
    import time
    raw=(INPUTS/'training-plan.json').read_bytes();plan=json.loads(raw)
    if not plan['execution_authorized']:raise PermissionError('Not admitted')
    for filename,digest in plan['files_sha256'].items():
        if hashlib.sha256((INPUTS/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Local source drift')
    path=RUNTIME/'training-launch.json'
    with path.open('x') as f:json.dump(dict(state='submitting',submitted_at=time.time()),f)
    calls=[]
    for index in range(3):
        call=train_trial.spawn(index,hashlib.sha256(raw).hexdigest(),time.time()+900)
        calls.append(dict(index=index,function_call_id=call.object_id,submitted_at=time.time()))
        path.write_text(json.dumps(dict(state='submitted',calls=calls,volume=NAME),indent=2)+'\n')
    print(json.dumps(dict(state='submitted',calls=calls,volume=NAME)))


@app.local_entrypoint()
def score_completed():
    import time
    path=RUNTIME/'score-launch.json'
    with path.open('x') as f:json.dump(dict(state='submitting',submitted_at=time.time()),f)
    digest=hashlib.sha256((RUNTIME/'score-plan.json').read_bytes()).hexdigest();calls=[]
    for index in range(3):
        call=score_trial.spawn(index,digest)
        calls.append(dict(index=index,function_call_id=call.object_id,submitted_at=time.time()))
        path.write_text(json.dumps(dict(state='submitted',calls=calls,volume=NAME+'-scores'),indent=2)+'\n')
    print(json.dumps(dict(state='submitted',calls=calls,volume=NAME+'-scores')))
