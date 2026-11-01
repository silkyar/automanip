"""Three-arm continuation pilot; sequential trainers, root admission pending."""
import hashlib
import json
from pathlib import Path
import sys

import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,results
from smoke_capg import coordinator_image,convert_checkpoint,runtime_mount
from evaluate import frozen_evaluate
from risk_screen_contract import REVIEWED,ARMS,EXPERIMENT,job

helpers=[ROOT/'deploy/modal/precision_screen_contract.py',ROOT/'deploy/modal/risk_screen_contract.py']
train_image=trainer_image
coordinate_image=coordinator_image
for helper in helpers:
    train_image=train_image.add_local_file(helper,'/opt/screen/'+helper.name,copy=True)
    coordinate_image=coordinate_image.add_local_file(helper,'/opt/screen/'+helper.name,copy=True)
coordinate_image=coordinate_image.add_local_file(ROOT/'deploy/modal/risk-screen-plan.json','/opt/screen/plan.json',copy=True)
outputs=[modal.Volume.from_name('automanip-modal-'+arm+'-seed1-20261028',create_if_missing=True,version=2) for arm in ARMS]


@app.function(image=train_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),
              timeout=330,startup_timeout=30,retries=0,max_containers=1,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def risk_train_trial(program:bytes,reference:bytes,item:dict,expires_at:float,dispatched_at:float):
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
    from risk_screen_contract import job,command,read_regular,remaining_lease,MAX_STEPS,validate_inputs,PREFIX,audit_accounting
    admitted=time.time();started=time.monotonic()
    validate_inputs(program,reference,item)
    if not 299.9<=expires_at-dispatched_at<=300.1:raise ValueError('fixed original lease required')
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
            'confirmation_queries':0,'new_training_processes':0,'training_prefix':PREFIX,'reference_verified':True}
    work=out/'work';work.mkdir(exist_ok=False);source=out/'trainer.py';source.write_bytes(program)
    (out/'reference.pt').write_bytes(reference)
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
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('original300-second hard deadline exceeded')
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
        if transitions<=0 or transitions>MAX_STEPS or transitions!=rollouts*51200:raise RuntimeError('invalid actual step count')
        log=read_regular(work,'train.log',16*1024*1024).decode(errors='replace')
        builtins=re.findall(r'Evaluated (\d+) steps resulting in (\d+) episodes',log)
        record.update(actual_steps=transitions,completed_rollouts=rollouts,
                      optimizer_step_count=None,optimizer_step_upper_bound=rollouts*8*32,
                      total_training_steps_including_prefix=PREFIX['training_steps']+transitions,
                      built_in_evaluation_calls=len(builtins),built_in_evaluation_transitions=sum(int(x) for x,_ in builtins),
                      built_in_evaluation_episodes=sum(int(y) for _,y in builtins),
                      state='complete',stopped_early=transitions<MAX_STEPS)
        record.update(audit_accounting(scalars,item['method'],rollouts))
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


@app.function(image=coordinate_image,cpu=.125,memory=512,timeout=1800,retries=0,max_containers=1,
              single_use_containers=True,include_source=False,serialized=True,
              volumes={**{'/'+arm+'-output':volume.with_mount_options(read_only=True) for arm,volume in zip(ARMS,outputs)},'/results':results},
              _experimental_restrict_output=True)
def coordinate_risk(programs:list,reference:bytes,train_functions:list,converter_function:modal.Function,evaluator_function:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/screen');sys.path.insert(0,'/opt/automanip-modal')
    from risk_screen_contract import REVIEWED,ARMS,EXPERIMENT,job,read_regular,select_checkpoints,PREFIX,validate_inputs
    from transport import unpack_weights
    plan=json.loads(Path('/opt/screen/plan.json').read_text())
    if not plan['launch_enabled'] or plan['jobs']!=[job(m) for m in REVIEWED]:raise PermissionError('screen not admitted')
    if tuple(REVIEWED)!=ARMS or len(programs)!=3 or len(train_functions)!=3:raise ValueError('exactly three reviewed jobs required')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    if not claims.put(EXPERIMENT,{'started_at':time.time(),'plan':plan},skip_if_exists=True):
        raise RuntimeError('batch already claimed; no automatic campaign replay')
    out=Path('/results')/EXPERIMENT;out.mkdir(exist_ok=False)
    summary={'experiment_id':EXPERIMENT,'state':'started','confirmation_queries':0,'training':[],
             'development_curves':{},'evaluations':[],'cost_plan':plan['cost'],'started_at':time.time(),'training_prefix':PREFIX,'training_dispatch':'sequential; no overlapping trainers'}
    def persist():
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    calls=[]
    try:
        for index,(method,digest) in enumerate(REVIEWED.items()):
            item=job(method)
            validate_inputs(programs[index],reference,item)
            key=EXPERIMENT+'-'+digest+'-seed-1'
            dispatched=time.time();expires=dispatched+300
            if not claims.put(key,{'job':item,'dispatched_at':dispatched,'expires_at':expires},skip_if_exists=True):
                raise RuntimeError('training job already claimed; no fresh lease')
            record={'job':item,'state':'submitted','dispatched_at':dispatched,'expires_at':expires}
            summary['training'].append(record);persist()
            call=train_functions[index].spawn(programs[index],reference,item,expires,dispatched)
            record['function_call_id']=call.object_id;calls.append(call);persist()
            # Wait for this trainer before the next arm receives its own lease.
            receipt=call.get()
            record['trainer_receipt']=receipt;persist()
        roots=[Path('/'+arm+'-output/trial') for arm in ARMS]
        volumes=[modal.Volume.from_name('automanip-modal-'+arm+'-seed1-20261028') for arm in ARMS]
        selected=[]
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
            points=select_checkpoints(files,native['actual_steps'])
            method=item['method'];summary['development_curves'][method]=[]
            record['prespecified_censored_steps']=[]
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
            selected.append(method_points);persist()
        # Evaluate preselected points in rounds across methods. Identical weights
        # are scored once and reused; built-in training metrics never enter here.
        memo={}
        for point_index in range(max(map(len,selected))):
            pending=[]
            for method_index,points in enumerate(selected):
                if point_index>=len(points):continue
                point=points[point_index];digest=point['weights_sha256']
                if digest not in memo:
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
                     'prefix_training_steps':PREFIX['training_steps'],'total_training_steps_including_prefix':PREFIX['training_steps']+point['steps'],
                     'prefix_trainer_seconds':PREFIX['trainer_seconds'],'total_trainer_seconds_prefix_plus_continuation':PREFIX['trainer_seconds']+point['trainer_seconds'],
                     'weights_sha256':point['weights_sha256'],'native_checkpoint_sha256':point['native_checkpoint_sha256'],
                     'evaluation_seconds':result['evaluation_seconds'],'evaluation_queue_seconds':result['queue_seconds'],
                     'gpu_actual_name':result['gpu_actual_name'],'durable_report_path':result['durable_path']}
                summary['development_curves'][method].append(row);summary['evaluations'].append(row);persist()
        summary.update(state='complete',completed_at=time.time())
    except Exception as exc:
        summary.update(state='failed',completed_at=time.time(),error=type(exc).__name__+': '+str(exc))
    finally:persist()
    return {'experiment_id':EXPERIMENT,'state':summary['state'],'summary_path':EXPERIMENT+'/summary.json'}


@app.local_entrypoint()
def run_risk_pilot():
    from risk_screen_contract import REVIEWED,ARMS,EXPERIMENT,job,validate_inputs,PREFIX
    plan_path=ROOT/'deploy/modal/risk-screen-plan.json';plan=json.loads(plan_path.read_text())
    if not plan['launch_enabled'] or tuple(REVIEWED)!=ARMS or plan['jobs']!=[job(arm) for arm in ARMS]:raise PermissionError('exact root admission required')
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:
        raise ValueError('scientific plan changed after admission')
    launch_path=ROOT/'.runtime/modal-risk-screen-launch.json'
    if launch_path.exists():raise RuntimeError('batch already submitted; inspect durable outputs')
    reference=(ROOT/'.runtime/continuation-reference-seed1/final_ckpt.pt').read_bytes()
    programs=[];functions=[]
    for method,volume in zip(REVIEWED,outputs):
        raw=(ROOT/'research/discovery/risk-pilot-v1'/(method+'.py')).read_bytes()
        validate_inputs(raw,reference,job(method))
        programs.append(raw)
        function=risk_train_trial.with_options(volumes={**runtime_mount,'/trainer-output':volume.with_mount_options(sub_path='/trial')})
        function.hydrate();functions.append(function)
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=3);evaluator.hydrate()
    call=coordinate_risk.spawn(programs,reference,functions,convert_checkpoint,evaluator)
    launch={'experiment_id':EXPERIMENT,'function_call_id':call.object_id,'gpu_requested':'L40S','training_seeds':[1],
            'methods':list(REVIEWED),'budget_cap_usd':1.,'training_prefix':PREFIX,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'summary_path':EXPERIMENT+'/summary.json','confirmation_queries':0}
    launch_path.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch,indent=2))
