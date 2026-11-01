"""Two reviewed from-scratch environment-count jobs; root admission required."""
import hashlib
import json
from pathlib import Path
import sys

import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,results
from smoke_capg import coordinator_image,convert_checkpoint,runtime_mount
from evaluate import frozen_evaluate
from env_count_screen_contract import REVIEWED,ARMS,EXPERIMENT,job

HELPER_FILES=('precision_screen_contract.py','env_count_screen_contract.py')
train_image=trainer_image
coordinate_image=coordinator_image
for name in HELPER_FILES:
    helper=ROOT/'deploy/modal'/name
    train_image=train_image.add_local_file(helper,'/opt/screen/'+name,copy=True)
    coordinate_image=coordinate_image.add_local_file(helper,'/opt/screen/'+name,copy=True)
coordinate_image=coordinate_image.add_local_file(ROOT/'deploy/modal/env-count-screen-plan.json','/opt/screen/plan.json',copy=True)
coordinate_image=coordinate_image.add_local_file(ROOT/'deploy/modal/env-count-reference.json','/opt/screen/reference.json',copy=True)
outputs=[modal.Volume.from_name('automanip-modal-envcount-'+arm+'-20261028',create_if_missing=True,version=2) for arm in ARMS]


@app.function(image=train_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),
              timeout=660,startup_timeout=30,retries=0,max_containers=2,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def env_count_train_trial(program:bytes,item:dict,expires_at:float,dispatched_at:float):
    import hashlib
    import importlib.metadata
    import math
    import json
    import os
    from pathlib import Path
    import re
    import signal
    import subprocess
    import sys
    import time
    sys.path.insert(0,'/opt/screen')
    from env_count_screen_contract import job,command,read_regular,remaining_lease,MAX_STEPS,audit_accounting,batch_steps
    admitted=time.time();started=time.monotonic()
    if item!=job(item['method']) or hashlib.sha256(program).hexdigest()!=item['trainer_sha256']:
        raise ValueError('unreviewed job or trainer')
    if not 599.9<=expires_at-dispatched_at<=600.1:raise ValueError('fixed original lease required')
    out=Path('/trainer-output')
    # A kernel-level Volume-v2 sync persists this marker before any training.
    # Restarts of the same input refuse a new process, even with time remaining.
    try:
        with (out/'attempt.json').open('x') as stream:
            json.dump({'job_id':item['job_id'],'admitted_at':admitted,'expires_at':expires_at},stream)
    except FileExistsError:raise RuntimeError('prior trainer attempt exists; no automatic rerun')
    subprocess.run(['sync',str(out)],check=True,timeout=20)
    record={'job':item,'state':'started','dispatched_at':dispatched_at,'admitted_at':admitted,
            'queue_seconds':max(0.,admitted-dispatched_at),'expires_at':expires_at,
            'trainer_sha256':hashlib.sha256(program).hexdigest(),'gpu_requested':'L40S',
            'confirmation_queries':0,'new_training_processes':0}
    work=out/'work';work.mkdir(exist_ok=False);source=out/'trainer.py';source.write_bytes(program)
    try:
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('trainer isolation boundary failed')
        import torch
        record['gpu_actual_name']=torch.cuda.get_device_name(0)
        record['versions']={k:importlib.metadata.version(k) for k in ('torch','mani_skill','sapien','numpy','gymnasium')}
        remaining,deadline=remaining_lease(expires_at,time.time(),time.monotonic())
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(deadline)
        record['subprocess_started_at']=time.time();process_started=time.monotonic()
        with (work/'train.log').open('w') as log:
            process=subprocess.Popen(command(sys.executable,source,item),cwd=work,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['new_training_processes']=1
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('original600-second hard deadline exceeded')
        record['trainer_process_seconds']=time.monotonic()-process_started
        record['return_code']=code
        if code:raise RuntimeError('trainer subprocess failed')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        run=work/'runs'/item['job_id'];events=EventAccumulator(str(run),size_guidance={'scalars':0});events.Reload()
        scalars={tag:[{'steps':int(e.step),'value':float(e.value) if math.isfinite(float(e.value)) else None,'wall_time':e.wall_time,
                      'trainer_seconds':e.wall_time-record['subprocess_started_at']} for e in events.Scalars(tag)]
                 for tag in events.Tags()['scalars']}
        record['nonfinite_diagnostic_scalars']={tag:sum(e['value'] is None for e in rows) for tag,rows in scalars.items() if any(e['value'] is None for e in rows)}
        (out/'scalars.json').write_text(json.dumps(scalars,allow_nan=False)+'\n')
        transitions=int(scalars['time/step'][-1]['value']);rollouts=len(scalars['time/step'])
        if transitions<=0 or transitions>MAX_STEPS or transitions!=rollouts*batch_steps(item['method']):raise RuntimeError('invalid actual step count')
        log=read_regular(work,'train.log',16*1024*1024).decode(errors='replace')
        builtins=re.findall(r'Evaluated (\d+) steps resulting in (\d+) episodes',log)
        record.update(actual_steps=transitions,completed_rollouts=rollouts,
                      optimizer_step_upper_bound=rollouts*8*32,
                      built_in_evaluation_calls=len(builtins),built_in_evaluation_transitions=sum(int(x) for x,_ in builtins),
                      built_in_evaluation_episodes=sum(int(y) for _,y in builtins),
                      state='complete',stopped_early=transitions<MAX_STEPS)
        record.update(audit_accounting(scalars,item['method'],rollouts,log))
        if not (run/'final_ckpt.pt').is_file():raise RuntimeError('final checkpoint missing')
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        record['wrapper_seconds']=time.monotonic()-started;record['completed_at']=time.time()
        inventory=[];total=0
        for parent,dirs,files in os.walk(out,followlinks=False):
            if any((Path(parent)/name).is_symlink() for name in dirs):raise ValueError('symlink output directory')
            for name in sorted(files):
                path=Path(parent)/name;relative=str(path.relative_to(out))
                if relative=='record.json':continue
                raw=read_regular(out,relative,64*1024*1024);total+=len(raw)
                if total>256*1024*1024:raise ValueError('trial output aggregate limit')
                inventory.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                  'trainer_seconds':max(0.,path.stat().st_mtime-record.get('subprocess_started_at',admitted))})
        record['files']=inventory;record['output_bytes']=total
        (out/'record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(out)],check=True,timeout=30)
    return {'job_id':item['job_id'],'state':record['state'],'record_path':'record.json'}


@app.function(image=coordinate_image,cpu=.125,memory=512,timeout=3600,retries=0,max_containers=1,
              single_use_containers=True,include_source=False,serialized=True,
              volumes={**{'/'+arm+'-output':volume.with_mount_options(read_only=True) for arm,volume in zip(ARMS,outputs)},'/results':results},
              _experimental_restrict_output=True)
def coordinate_env_count(programs:list,train_functions:list,converter_function:modal.Function,evaluator_function:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/screen');sys.path.insert(0,'/opt/automanip-modal')
    from env_count_screen_contract import REVIEWED,ARMS,EXPERIMENT,job,read_regular,select_checkpoints,selected,validate_admission,existing_report,PROTOCOL
    from transport import unpack_weights
    plan=json.loads(Path('/opt/screen/plan.json').read_text())
    validate_admission(plan)
    reference_raw=Path('/opt/screen/reference.json').read_bytes()
    if hashlib.sha256(reference_raw).hexdigest()!=plan['reference_provenance_sha256']:raise ValueError('reference provenance changed')
    reference=json.loads(reference_raw)
    if reference['binding']['reference_job']['trainer_sha256']!=next(iter(REVIEWED.values())):raise ValueError('reference source mismatch')
    if tuple(REVIEWED)!=ARMS or len(programs)!=2 or len(train_functions)!=2:raise ValueError('exactly two reviewed jobs required')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    if not claims.put(EXPERIMENT,{'started_at':time.time(),'plan':plan},skip_if_exists=True):
        raise RuntimeError('batch already claimed; no automatic campaign replay')
    out=Path('/results')/EXPERIMENT;out.mkdir(exist_ok=False)
    summary={'experiment_id':EXPERIMENT,'state':'started','confirmation_queries':0,'training':[],
             'development_curves':{},'evaluations':[],'cost_plan':plan['cost'],'started_at':time.time()}
    def persist():
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    calls=[]
    try:
        summary['historical_reference']=dict(reference,development_curves=[])
        for point in reference['points']:
            durable=json.loads(read_regular('/results',point['durable_report_path'],4*1024*1024))
            reused=existing_report(durable,point['weights_sha256'])
            summary['historical_reference']['development_curves'].append(dict(point,successes=reused['report']['successes_at_end'],episodes=128,reused_global_report=True))
        persist()
        for index,(method,digest) in enumerate(REVIEWED.items()):
            item=job(method)
            if hashlib.sha256(programs[index]).hexdigest()!=digest:raise ValueError('reviewed payload changed')
            key=item['job_id']+'-'+digest
            dispatched=time.time();expires=dispatched+600
            if not claims.put(key,{'job':item,'dispatched_at':dispatched,'expires_at':expires},skip_if_exists=True):
                raise RuntimeError('training job already claimed; no fresh lease')
            record={'job':item,'state':'submitted','dispatched_at':dispatched,'expires_at':expires}
            summary['training'].append(record);persist()
            call=train_functions[index].spawn(programs[index],item,expires,dispatched)
            record['function_call_id']=call.object_id;calls.append(call);persist()
        roots=[Path('/'+arm+'-output/trial') for arm in ARMS]
        volumes=[modal.Volume.from_name('automanip-modal-envcount-'+arm+'-20261028') for arm in ARMS]
        selected_points=[]
        for index,(call,root,volume) in enumerate(zip(calls,roots,volumes)):
            receipt_error=None
            try:call.get()
            except Exception as exc:receipt_error=type(exc).__name__+': '+str(exc)
            volume.reload()
            native=json.loads(read_regular(root,'record.json',2*1024*1024))
            record=summary['training'][index];record.update(native)
            if receipt_error:record['receipt_transport_error']=receipt_error
            if native['state']!='complete':raise RuntimeError('training failed; preserve artifacts without rerun')
            item=native['job'];prefix='work/runs/'+item['job_id']+'/'
            inventory={r['path']:r for r in native['files']}
            files={Path(p).name for p in inventory if p.startswith(prefix) and p.endswith('.pt')}
            points=select_checkpoints(files,native['actual_steps'],item['method'])
            method=item['method'];summary['development_curves'][method]=[]
            record['prespecified_censored_steps']=[steps for steps in selected(item['method']).values() if steps>native['actual_steps']]
            initial=inventory[prefix+'ckpt_1.pt']
            record['initial_checkpoint_sha256']=initial['sha256']
            record['initial_checkpoint_bytes_identical_to_reference']=initial['sha256']==reference['initial_checkpoint_sha256']
            record['initial_identity_interpretation']='Equal digests prove bytes identical; unequal digests alone do not establish different weights.'
            method_points=[]
            for point in points:
                relative=prefix+point['filename'];entry=inventory[relative]
                raw=read_regular(root,relative,64*1024*1024)
                digest=hashlib.sha256(raw).hexdigest()
                if digest!=entry['sha256']:raise ValueError('native checkpoint changed after receipt')
                # Sole native deserialization happens in the existing restricted,
                # weights-only converter; this trusted coordinator reads bytes.
                converted=converter_function.remote(raw)
                if converted['export']['checkpoint_sha256']!=digest:raise ValueError('conversion provenance mismatch')
                weights=unpack_weights(converted['weights_gzip']);metadata=json.loads(converted['manifest'])
                if hashlib.sha256(weights).hexdigest()!=metadata['weights_sha256']:raise ValueError('safe artifact digest mismatch')
                policy_dir=out/method/('step-'+str(point['steps']));policy_dir.mkdir(parents=True,exist_ok=False)
                (policy_dir/'manifest.json').write_bytes(converted['manifest']);(policy_dir/'weights.json').write_bytes(weights)
                (policy_dir/'export.json').write_text(json.dumps(converted['export'],indent=2)+'\n')
                method_points.append(dict(point,manifest=converted['manifest'],weights_gzip=converted['weights_gzip'],
                                          weights_sha256=metadata['weights_sha256'],native_checkpoint_sha256=digest,
                                          trainer_seconds=entry['trainer_seconds']))
            selected_points.append(method_points);persist()
        # Evaluate preselected points in rounds across methods. Identical weights
        # are scored once and reused; built-in training metrics never enter here.
        memo={}
        for point_index in range(max(map(len,selected_points))):
            pending=[]
            for method_index,points in enumerate(selected_points):
                if point_index>=len(points):continue
                point=points[point_index];digest=point['weights_sha256']
                if digest not in memo:
                    expected='frozen-evaluation-attempts/'+PROTOCOL+'/'+digest+'/development/result.json'
                    modal.Volume.from_name('automanip-modal-validation').reload()
                    try:durable=json.loads(read_regular('/results',expected,4*1024*1024))
                    except FileNotFoundError:durable=None
                    reused=existing_report(durable,digest)
                    if reused is not None:memo[digest]=reused
                    else:
                        gpu_claims=modal.Dict.from_name('automanip-modal-frozen-evaluation-attempts')
                        if gpu_claims.get(PROTOCOL+'-'+digest+'-development',None) is not None:raise RuntimeError('existing GPU claim without reusable report; wait/reconcile, never replay')
                        call=evaluator_function.spawn(point['manifest'],point['weights_gzip'],'development','L40S',time.time(),100)
                        memo[digest]={'call':call,'function_call_id':call.object_id}
                pending.append((method_index,point,memo[digest]))
            for method_index,point,result in pending:
                if 'report' not in result:
                    metadata=result['call'].get()
                    expected='frozen-evaluation-attempts/'+plan['protocol_sha256']+'/'+point['weights_sha256']+'/development/result.json'
                    if metadata['gpu_durable_result_path']!=expected:raise ValueError('unexpected evaluation result path')
                    modal.Volume.from_name('automanip-modal-validation').reload()
                    durable=json.loads(read_regular('/results',expected,4*1024*1024));report=durable['report']
                    if durable['state']!='complete' or report['split']!='development' or report['smoke'] or report['episode_count']!=128 or report['failed_episodes']:
                        raise ValueError('invalid protected development report')
                    if report['weights_sha256']!=point['weights_sha256'] or report['protocol_sha256']!=plan['protocol_sha256']:
                        raise ValueError('development report provenance mismatch')
                    result.update(report=report,durable_path=expected,evaluation_seconds=durable['evaluation_seconds'],
                                  gpu_actual_name=durable['gpu_actual_name'],queue_seconds=durable['queue_seconds'])
                report=result['report'];method=summary['training'][method_index]['job']['method']
                row={'steps':point['steps'],'successes':report['successes_at_end'],'episodes':128,
                     'trainer_seconds':point['trainer_seconds'],'point':point['point'],
                     'weights_sha256':point['weights_sha256'],'native_checkpoint_sha256':point['native_checkpoint_sha256'],
                     'evaluation_seconds':result['evaluation_seconds'],'evaluation_queue_seconds':result['queue_seconds'],
                     'gpu_actual_name':result['gpu_actual_name'],'durable_report_path':result['durable_path'],'reused_global_report':result.get('reused_global_report',False)}
                summary['development_curves'][method].append(row);summary['evaluations'].append(row);persist()
        summary.update(state='complete',completed_at=time.time())
    except Exception as exc:
        summary.update(state='failed',completed_at=time.time(),error=type(exc).__name__+': '+str(exc))
    finally:persist()
    return {'experiment_id':EXPERIMENT,'state':summary['state'],'summary_path':EXPERIMENT+'/summary.json'}


@app.local_entrypoint()
def run_env_count_pair():
    from env_count_screen_contract import REVIEWED,ARMS,EXPERIMENT,job,validate_admission,validate_budget
    plan_path=ROOT/'deploy/modal/env-count-screen-plan.json';plan=json.loads(plan_path.read_text())
    validate_admission(plan)
    ledger_path=ROOT/'.runtime/modal-tonight-budget-ledger.json';validate_budget(json.loads(ledger_path.read_bytes()))
    if hashlib.sha256((ROOT/'.runtime/modal-env-count-reference-binding.json').read_bytes()).hexdigest()!=plan['reference_binding_sha256']:raise ValueError('historical reference binding changed')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:
        raise ValueError('scientific plan changed after admission')
    launch_path=ROOT/'.runtime/modal-env-count-screen-launch.json'
    if launch_path.exists():raise RuntimeError('batch already submitted; inspect durable outputs')
    programs=[];functions=[]
    for method,volume in zip(REVIEWED,outputs):
        raw=(ROOT/'research/discovery/from-scratch-env-count-v2'/('scratch-kl-stop-control.py')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=REVIEWED[method]:raise ValueError('reviewed trainer changed')
        programs.append(raw)
        function=env_count_train_trial.with_options(volumes={**runtime_mount,'/trainer-output':volume.with_mount_options(sub_path='/trial')})
        function.hydrate();functions.append(function)
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=2);evaluator.hydrate()
    call=coordinate_env_count.spawn(programs,functions,convert_checkpoint,evaluator)
    launch={'experiment_id':EXPERIMENT,'function_call_id':call.object_id,'gpu_requested':'L40S','training_seeds':[1],
            'methods':list(REVIEWED),'budget_cap_usd':3.,'ledger_snapshot_sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'summary_path':EXPERIMENT+'/summary.json','confirmation_queries':0}
    launch_path.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch,indent=2))
