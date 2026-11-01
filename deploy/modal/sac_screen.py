"""One root-reviewed adapted SAC performance screen; admission disabled by default."""
import hashlib
import json
from pathlib import Path
import sys

import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,validation_image,results
from smoke_capg import runtime_mount
from evaluate import frozen_evaluate
from sac_screen_contract import SOURCE_SHA,EXPERIMENT

output=modal.Volume.from_name('automanip-modal-sac-screen-seed1-20261027',create_if_missing=True,version=2)
helpers=[ROOT/'deploy/modal/screen_contract.py',ROOT/'deploy/modal/sac_screen_contract.py']
candidate_image=trainer_image
trusted_image=validation_image
for helper in helpers:
    candidate_image=candidate_image.add_local_file(helper,'/opt/screen/'+helper.name,copy=True)
    trusted_image=trusted_image.add_local_file(helper,'/opt/screen/'+helper.name,copy=True)
trusted_image=trusted_image.add_local_file(ROOT/'deploy/modal/sac-screen-plan.json','/opt/screen/plan.json',copy=True)


@app.function(image=candidate_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),
              timeout=1380,startup_timeout=30,retries=0,max_containers=1,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True,
              volumes={**runtime_mount,'/trainer-output':output.with_mount_options(sub_path='/trial')})
def train_sac_screen(program:bytes,expires_at:float,dispatched_at:float):
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import time
    sys.path.insert(0,'/opt/screen')
    from sac_screen_contract import SOURCE_SHA,EXPERIMENT,CONFIG,command
    from screen_contract import read_regular,remaining_lease
    admitted=time.time();mark=time.monotonic()
    if hashlib.sha256(program).hexdigest()!=SOURCE_SHA:raise ValueError('unreviewed SAC source')
    if not 1319.9<=expires_at-dispatched_at<=1320.1:raise ValueError('original lease required')
    root=Path('/trainer-output')
    with (root/'attempt.json').open('x') as stream:
        json.dump({'experiment_id':EXPERIMENT,'admitted_at':admitted,'expires_at':expires_at},stream)
    subprocess.run(['sync',str(root)],check=True,timeout=20)
    record={'state':'started','source_sha256':SOURCE_SHA,'experiment_id':EXPERIMENT,
            'dispatched_at':dispatched_at,'admitted_at':admitted,'expires_at':expires_at,
            'queue_seconds':max(0.,admitted-dispatched_at),'gpu_requested':'L40S','confirmation_queries':0}
    try:
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('trainer isolation failed')
        import torch
        record['gpu_actual_name']=torch.cuda.get_device_name(0)
        remaining,deadline=remaining_lease(expires_at,time.time(),time.monotonic())
        source=root/'trainer.py';source.write_bytes(program)
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(deadline)
        record['subprocess_started_at']=time.time();process_started=time.monotonic()
        with (root/'train.log').open('w') as log:
            process=subprocess.Popen(command(sys.executable,source,root/'run'),cwd=root,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('original1320-second hard deadline exceeded')
        record.update(trainer_process_seconds=time.monotonic()-process_started,return_code=code)
        if code:raise RuntimeError('SAC subprocess failed')
        info=json.loads(read_regular(root,'run/run.json',1024*1024))
        report=json.loads(read_regular(root,'run/report.json',2*1024*1024))
        if info['source_sha256']!=SOURCE_SHA or info['config']!=CONFIG:raise ValueError('source/configuration provenance mismatch')
        actual=report['transitions']
        if actual<=0 or actual>500000 or actual%32 or report['vector_step_calls']*32!=actual:raise ValueError('invalid actual transition count')
        counts=[report[k] for k in ('critic_optimizer_steps','actor_optimizer_steps','alpha_optimizer_steps')]
        if len(set(counts))!=1 or report['minibatch_presentations']!=counts[0]*1024:raise ValueError('inconsistent learner accounting')
        if report['resets']!=32+report['completed_episodes'] or report['evaluation_transitions'] or report['extra_simulator_queries']:
            raise ValueError('inconsistent simulator accounting')
        record.update(state='complete',metadata=info,report=report,actual_steps=actual)
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        record.update(wrapper_seconds=time.monotonic()-mark,completed_at=time.time())
        inventory=[];total=0
        for parent,dirs,files in os.walk(root,followlinks=False):
            if any((Path(parent)/name).is_symlink() for name in dirs):raise ValueError('symlink output directory')
            for name in sorted(files):
                path=Path(parent)/name;relative=str(path.relative_to(root))
                if relative=='record.json':continue
                raw=read_regular(root,relative,64*1024*1024);total+=len(raw)
                if total>256*1024*1024:raise ValueError('aggregate output bound')
                inventory.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                  'trainer_seconds':max(0.,path.stat().st_mtime-record.get('subprocess_started_at',admitted))})
        record.update(files=inventory,output_bytes=total)
        (root/'record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=30)
    return {'state':record['state'],'record_path':'record.json'}


@app.function(image=trusted_image,cpu=1,memory=2048,timeout=5400,retries=0,max_containers=1,
              single_use_containers=True,include_source=False,serialized=True,
              volumes={'/candidate':output.with_mount_options(read_only=True),'/results':results},
              _experimental_restrict_output=True)
def coordinate_sac_screen(program:bytes,trainer:modal.Function,evaluator:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/screen');sys.path.insert(0,'/opt/automanip/src');sys.path.insert(0,'/opt/automanip-modal')
    from sac_screen_contract import SOURCE_SHA,EXPERIMENT,CONFIG,SELECTED,select_exports
    from screen_contract import read_regular
    from transport import pack_weights
    from automanip.policy_io import load_policy,PolicySchema
    plan=json.loads(Path('/opt/screen/plan.json').read_text())
    if not plan['launch_enabled'] or plan['source_sha256']!=SOURCE_SHA or plan['config']!=CONFIG:
        raise PermissionError('SAC screen not admitted')
    if hashlib.sha256(program).hexdigest()!=SOURCE_SHA:raise ValueError('unreviewed source')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    dispatched=time.time();expires=dispatched+1320
    if not claims.put(EXPERIMENT+'-'+SOURCE_SHA,{'dispatched_at':dispatched,'expires_at':expires,'plan':plan},skip_if_exists=True):
        raise RuntimeError('screen already claimed; no fresh allowance')
    out=Path('/results')/EXPERIMENT;out.mkdir(exist_ok=False)
    summary={'state':'started','experiment_id':EXPERIMENT,'source_sha256':SOURCE_SHA,'dispatched_at':dispatched,
             'expires_at':expires,'confirmation_queries':0,'development_curve':[],'cost_plan':plan['cost']}
    def persist():
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    persist()
    try:
        try:trainer.remote(program,expires,dispatched)
        except Exception as exc:summary['receipt_transport_error']=type(exc).__name__+': '+str(exc)
        modal.Volume.from_name('automanip-modal-sac-screen-seed1-20261027').reload()
        native=json.loads(read_regular('/candidate','trial/record.json',2*1024*1024));summary['training']=native
        if native['state']!='complete':raise RuntimeError(native.get('error','training failed'))
        points=select_exports(native['report']);inventory={entry['path']:entry for entry in native['files']}
        summary['prespecified_censored_steps']=[steps for steps in SELECTED if steps>native['actual_steps']]
        schema=PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f',48,8)
        memo={}
        for point in points:
            policy=out/('step-'+str(point['steps']));policy.mkdir()
            raw_files={}
            for filename,limit in [('manifest.json',8192),('weights.json',32*1024*1024)]:
                relative='run/'+point['label']+'/'+filename
                raw=read_regular('/candidate','trial/'+relative,limit)
                if hashlib.sha256(raw).hexdigest()!=inventory[relative]['sha256']:raise ValueError('actor artifact changed')
                (policy/filename).write_bytes(raw);raw_files[filename]=raw
            artifact=load_policy(policy,schema)
            if artifact.weights_sha256!=point['weights_sha256']:raise ValueError('declared actor hash mismatch')
            trainer_seconds=inventory['run/'+point['label']+'/weights.json']['trainer_seconds']
            if artifact.weights_sha256 not in memo:
                result=evaluator.remote(raw_files['manifest.json'],pack_weights(raw_files['weights.json']),'development','L40S',time.time(),100)
                expected='frozen-evaluation-attempts/'+plan['protocol_sha256']+'/'+artifact.weights_sha256+'/development/result.json'
                if result['gpu_durable_result_path']!=expected:raise ValueError('unexpected evaluator path')
                modal.Volume.from_name('automanip-modal-validation').reload()
                durable=json.loads(read_regular('/results',expected,4*1024*1024));report=durable['report']
                if durable['state']!='complete' or report['weights_sha256']!=artifact.weights_sha256 or report['protocol_sha256']!=plan['protocol_sha256']:
                    raise ValueError('frozen report provenance mismatch')
                if report['split']!='development' or report['episode_count']!=128 or report['failed_episodes'] or report['smoke']:
                    raise ValueError('invalid protected development report')
                memo[artifact.weights_sha256]=(durable,expected)
            durable,path=memo[artifact.weights_sha256];report=durable['report']
            summary['development_curve'].append(dict(point,successes=report['successes_at_end'],episodes=128,
                trainer_seconds=trainer_seconds,evaluation_seconds=durable['evaluation_seconds'],
                evaluation_queue_seconds=durable['queue_seconds'],gpu_actual_name=durable['gpu_actual_name'],durable_report_path=path))
            persist()
        summary.update(state='complete',completed_at=time.time())
    except Exception as exc:summary.update(state='failed',completed_at=time.time(),error=type(exc).__name__+': '+str(exc))
    finally:persist()
    return {'state':summary['state'],'summary_path':EXPERIMENT+'/summary.json'}


@app.local_entrypoint()
def sac_performance():
    plan_path=ROOT/'deploy/modal/sac-screen-plan.json';plan=json.loads(plan_path.read_text())
    proof=json.loads((ROOT/'.runtime/modal-sac-smoke-v2-result.json').read_text())
    if not plan['launch_enabled'] or proof['state']!='complete' or proof['gpu_training_transitions']!=800:
        raise PermissionError('root admission and passed SAC smoke required')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:
        raise ValueError('scientific plan changed after admission')
    program=(ROOT/'research/discovery/sac_baseline.py').read_bytes()
    if hashlib.sha256(program).hexdigest()!=SOURCE_SHA:raise ValueError('reviewed source changed')
    target=ROOT/'.runtime/modal-sac-screen-launch.json'
    if target.exists():raise RuntimeError('already submitted; inspect durable outputs')
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=1);evaluator.hydrate()
    call=coordinate_sac_screen.spawn(program,train_sac_screen,evaluator)
    receipt={'experiment_id':EXPERIMENT,'function_call_id':call.object_id,'source_sha256':SOURCE_SHA,
             'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'summary_path':EXPERIMENT+'/summary.json',
             'budget_cap_usd':1.5,'training_seeds':[1],'gpu_requested':'L40S','confirmation_queries':0}
    target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
