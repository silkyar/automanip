"""One fixed BC exploration pilot. No upload/compute before explicit root admission."""
import hashlib
import json
from pathlib import Path
import sys

import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,ROOT,trainer_image,results
from smoke_capg import coordinator_image,convert_checkpoint,runtime_mount
from evaluate import frozen_evaluate
from bc_exploration_contract import STAGES,STUDENTS,EXPERIMENT,job,volume_name

candidate_image=trainer_image
trusted_image=coordinator_image
for name in ('precision_screen_contract.py','bc_exploration_contract.py'):
    helper=ROOT/'deploy/modal'/name
    candidate_image=candidate_image.add_local_file(helper,'/opt/screen/'+name,copy=True)
    trusted_image=trusted_image.add_local_file(helper,'/opt/screen/'+name,copy=True)
# Exact candidate helper allowlist, not the repository or any evaluator source.
for name in ('amp_v1.py','amp_collect_v1.py','amp_bc_v1.py'):
    candidate_image=candidate_image.add_local_file(ROOT/'research/discovery'/name,'/opt/assist/research/discovery/'+name,copy=True)
for name in ('__init__.py','policy_io.py'):
    trusted_image=trusted_image.add_local_file(ROOT/'src/automanip'/name,'/opt/trusted/automanip/'+name,copy=True)
trusted_image=trusted_image.add_local_file(ROOT/'deploy/modal/bc-exploration-plan.json','/opt/screen/plan.json',copy=True)
outputs={stage:modal.Volume.from_name(volume_name(stage),create_if_missing=True,version=2) for stage in STAGES}


@app.function(image=candidate_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=660,startup_timeout=30,
              retries=0,max_containers=2,restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def run_bc_exploration_stage(stage:str,program:bytes,bindings:dict,lineage:dict,teacher:dict|None,expires_at:float,dispatched_at:float):
    import hashlib
    import importlib.metadata
    import json
    import math
    import os
    from pathlib import Path
    import re
    import signal
    import subprocess
    import sys
    import time
    sys.path.insert(0,'/opt/screen');sys.path.insert(0,'/opt/automanip-modal')
    from bc_exploration_contract import (STUDENTS,PINS,PREFIX,CORE_SHA,TEACHER_SHA,INPUT_LIMITS,job,command,remaining_lease,
                                    read_regular,verify_sources,validate_collection,validate_bc,student_accounting,readonly_inputs)
    from transport import unpack_weights
    item=job(stage);root=Path('/trainer-output');admitted=time.time();mark=time.monotonic()
    core=Path('/opt/assist/research/discovery/amp_v1.py').read_bytes();verify_sources(program,stage,core)
    remaining_lease(stage,expires_at,dispatched_at,time.time(),time.monotonic())
    # This marker is durable before any candidate process. Replays fail closed.
    with (root/'attempt.json').open('x') as stream:json.dump({'stage':stage,'admitted_at':admitted,'expires_at':expires_at},stream)
    subprocess.run(['sync',str(root)],check=True,timeout=20)
    record={'stage':stage,'job':item,'state':'started','source_sha256':hashlib.sha256(program).hexdigest(),
            'core_source_sha256':CORE_SHA,'dispatched_at':dispatched_at,'admitted_at':admitted,'expires_at':expires_at,
            'queue_seconds':max(0.,admitted-dispatched_at),'gpu_requested':'L40S','teacher_prefix':PREFIX,
            'input_bindings':bindings,'producer_receipts':lineage,'new_training_processes':0,'confirmation_queries':0}
    (root/'source.py').write_bytes(program)
    try:
        if Path('/results').exists() or Path('/opt/automanip').exists():raise RuntimeError('candidate isolation boundary failed')
        import torch
        record['gpu_actual_name']=torch.cuda.get_device_name(0)
        record['versions']={key:importlib.metadata.version(key) for key in PINS}
        if record['versions']!=PINS or record['gpu_actual_name']!='NVIDIA L40S':raise ValueError('exact production GPU pins required')
        required={mount.lstrip('/') for mount in readonly_inputs(stage)}
        if set(bindings)!=required:raise ValueError('exact readonly stage input bindings required')
        for mount,files in bindings.items():
            allowed={'dataset.json','trajectories.npz'} if mount=='demo' else {'bc-report.json','final-actor/manifest.json','final-actor/weights.json'}
            if set(files)!=allowed:raise ValueError('unexpected mounted input file')
            for relative,digest in files.items():
                raw=read_regular('/'+mount,relative,INPUT_LIMITS[relative])
                if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('readonly producer artifact changed')
        metadata=report=None
        if stage=='collect':
            if lineage or teacher is None:raise ValueError('collector requires only reviewed teacher actor')
            weights=unpack_weights(teacher['weights_gzip']);manifest=json.loads(teacher['manifest'])
            if manifest['weights_sha256']!=TEACHER_SHA or hashlib.sha256(weights).hexdigest()!=TEACHER_SHA:raise ValueError('teacher digest mismatch')
            target=root/'teacher';target.mkdir();(target/'manifest.json').write_bytes(teacher['manifest']);(target/'weights.json').write_bytes(weights)
        else:
            if teacher is not None:raise ValueError('teacher weights forbidden in BC/student stages')
            metadata=json.loads(read_regular('/demo','dataset.json',INPUT_LIMITS['dataset.json']))
            validate_collection(metadata,lineage['collection'])
            if stage in STUDENTS:
                report=json.loads(read_regular('/bc','bc-report.json',INPUT_LIMITS['bc-report.json']))
                validate_bc(report,metadata,lineage['bc'])
        if stage in STUDENTS:
            source=root/'source.py'
        else:
            source=Path('/opt/assist')/item['source_path']
            if source.read_bytes()!=program:raise ValueError('helper image differs from reviewed program')
        work=root/'work';work.mkdir()
        remaining,deadline=remaining_lease(stage,expires_at,dispatched_at,time.time(),time.monotonic())
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env.update(AUTOMANIP_TRAINING_DEADLINE=str(deadline),PYTHONPATH='/opt/assist')
        record['subprocess_started_at']=time.time();started=time.monotonic()
        with (root/'train.log').open('w') as stream:
            process=subprocess.Popen(command(stage,sys.executable,str(source)),cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record['new_training_processes']=1
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('original stage hard deadline exceeded')
        record.update(return_code=code,trainer_process_seconds=time.monotonic()-started)
        if code:raise RuntimeError('candidate process failed')
        record['state']='complete'
        if stage=='collect':
            metadata=json.loads(read_regular(root,'run/dataset.json',INPUT_LIMITS['dataset.json']));validate_collection(metadata,record)
            raw=read_regular(root,'run/trajectories.npz',INPUT_LIMITS['trajectories.npz'])
            if hashlib.sha256(raw).hexdigest()!=metadata['sha256']:raise ValueError('dataset digest mismatch')
            # Publish only these two files. Teacher files/source are outside this mount.
            publish=root/'dataset';publish.mkdir()
            (publish/'trajectories.npz').write_bytes(raw);(publish/'dataset.json').write_bytes(read_regular(root,'run/dataset.json',INPUT_LIMITS['dataset.json']))
            record['collection']=metadata;record['student_environment_transitions']=0
        elif stage=='bc':
            report=json.loads(read_regular(root,'run/bc-report.json',INPUT_LIMITS['bc-report.json']));validate_bc(report,metadata,record)
            publish=root/'published';(publish/'final-actor').mkdir(parents=True)
            for relative in ('bc-report.json','final-actor/manifest.json','final-actor/weights.json'):
                (publish/relative).write_bytes(read_regular(root,'run/'+relative,INPUT_LIMITS[relative]))
            record['bc']=report;record['student_environment_transitions']=0
        else:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            run=work/'runs'/item['job_id'];events=EventAccumulator(str(run),size_guidance={'scalars':0});events.Reload()
            scalars={tag:[{'steps':int(e.step),'value':float(e.value) if math.isfinite(float(e.value)) else None,
                          'wall_time':e.wall_time,'trainer_seconds':e.wall_time-record['subprocess_started_at']} for e in events.Scalars(tag)] for tag in events.Tags()['scalars']}
            (root/'scalars.json').write_text(json.dumps(scalars,allow_nan=False)+'\n')
            log=read_regular(root,'train.log',16*1024*1024).decode(errors='replace')
            record.update(student_accounting(stage,scalars,log,bindings,metadata,report))
            if record['exploration_initialization']['calibration_wall_seconds']>record['trainer_process_seconds']+.1:raise ValueError('calibration time exceeds full child runtime')
            builtins=re.findall(r'Evaluated (\d+) steps resulting in (\d+) episodes',log)
            record.update(built_in_evaluation_calls=len(builtins),built_in_evaluation_transitions=sum(int(a) for a,b in builtins),
                          built_in_evaluation_episodes=sum(int(b) for a,b in builtins))
            if not (run/'final_ckpt.pt').is_file():raise ValueError('final student checkpoint missing')
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        record.update(wrapper_seconds=time.monotonic()-mark,completed_at=time.time())
        inventory=[];total=0
        for parent,dirs,files in os.walk(root,followlinks=False):
            if any((Path(parent)/name).is_symlink() for name in dirs):raise ValueError('symlink output directory')
            for name in sorted(files):
                relative=str((Path(parent)/name).relative_to(root))
                if relative=='record.json':continue
                raw=read_regular(root,relative,64*1024*1024);total+=len(raw)
                if total>256*1024*1024:raise ValueError('stage aggregate output bound')
                inventory.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        record.update(files=inventory,output_bytes=total)
        (root/'record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=True,timeout=30)
    return {'stage':stage,'state':record['state'],'record_path':'trial/record.json'}


@app.function(image=trusted_image,cpu=.125,memory=512,timeout=3600,retries=0,max_containers=1,
              single_use_containers=True,include_source=False,serialized=True,_experimental_restrict_output=True,
              volumes={**{'/stage-'+stage:volume.with_mount_options(read_only=True) for stage,volume in outputs.items()},'/results':results})
def coordinate_bc_exploration(programs:dict,teacher:dict,workers:dict,converter:modal.Function,evaluator:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/screen');sys.path.insert(0,'/opt/trusted');sys.path.insert(0,'/opt/automanip-modal')
    from bc_exploration_contract import (STAGES,STUDENTS,EXPERIMENT,PROTOCOL,TASK,PREFIX,INPUT_LIMITS,TEACHER_SHA,
                                    job,volume_name,read_regular,validate_admission,validate_collection,validate_bc)
    from automanip.policy_io import load_policy,PolicySchema
    from transport import pack_weights,unpack_weights
    plan=json.loads(Path('/opt/screen/plan.json').read_bytes());validate_admission(plan)
    if set(programs)!=set(STAGES) or set(workers)!=set(STAGES):raise ValueError('exact stage inputs required')
    if any(hashlib.sha256(programs[s]).hexdigest()!=job(s)['source_sha256'] for s in STAGES):raise ValueError('reviewed source changed')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    if not claims.put(EXPERIMENT,{'started_at':time.time(),'plan':plan},skip_if_exists=True):raise RuntimeError('pilot already claimed; no replay')
    out=Path('/results')/EXPERIMENT;out.mkdir(exist_ok=False)
    summary={'experiment_id':EXPERIMENT,'state':'started','teacher_prefix':PREFIX,'cost_plan':plan['cost'],'stages':{},
             'evaluations':{},'confirmation_queries':0,'started_at':time.time(),'accounting':'Shared teacher/demo/BC prefixes once per campaign; separately disclosed for each assisted policy.'}
    def persist():
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    def dispatch(stage,bindings,lineage,teacher_input=None):
        item=job(stage);now=time.time();expires=now+item['lease_seconds']
        if not claims.put(item['job_id']+'-'+item['source_sha256'],{'dispatched_at':now,'expires_at':expires},skip_if_exists=True):raise RuntimeError('stage already claimed')
        row={'state':'submitted','dispatched_at':now,'expires_at':expires};summary['stages'][stage]=row;persist()
        call=workers[stage].spawn(stage,programs[stage],bindings,lineage,teacher_input,expires,now)
        row['function_call_id']=call.object_id;persist();return call
    def native(stage,call):
        try:call.get()
        except Exception as exc:summary['stages'][stage]['receipt_error']=type(exc).__name__+': '+str(exc)
        modal.Volume.from_name(volume_name(stage)).reload()
        record=json.loads(read_regular('/stage-'+stage,'trial/record.json',2*1024*1024))
        summary['stages'][stage].update(record);persist()
        if record['state']!='complete':raise RuntimeError('stage failed; preserve outputs without rerun')
        return record
    def read(stage,relative,limit):
        record=summary['stages'][stage];entry=next((r for r in record['files'] if r['path']==relative),None)
        if entry is None:raise ValueError('file absent from durable producer inventory')
        raw=read_regular('/stage-'+stage,'trial/'+relative,limit)
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('producer output changed')
        return raw
    def receipt(record):
        return {k:record[k] for k in ('state','versions','gpu_actual_name','source_sha256','core_source_sha256','trainer_process_seconds')}
    def safe(stage,manifest,weights):
        root=out/stage;root.mkdir(exist_ok=False);(root/'manifest.json').write_bytes(manifest);(root/'weights.json').write_bytes(weights)
        actor=load_policy(root,PolicySchema(TASK,48,8));return actor.weights_sha256
    memo={}
    def score(stage,manifest,weights,expected_digest):
        digest=safe(stage,manifest,weights)
        if digest!=expected_digest:raise ValueError('declared actor identity mismatch before evaluation')
        if digest not in memo:
            result=evaluator.remote(manifest,pack_weights(weights),'development','L40S',time.time(),100)
            path='frozen-evaluation-attempts/'+PROTOCOL+'/'+digest+'/development/result.json'
            if result['gpu_durable_result_path']!=path:raise ValueError('unexpected frozen result path')
            modal.Volume.from_name('automanip-modal-validation').reload()
            durable=json.loads(read_regular('/results',path,4*1024*1024));report=durable['report']
            if durable['state']!='complete' or report['split']!='development' or report['smoke'] or report['episode_count']!=128 or report['failed_episodes'] or report['protocol_sha256']!=PROTOCOL or report['weights_sha256']!=digest:raise ValueError('invalid frozen development report')
            memo[digest]={'successes':report['successes_at_end'],'episodes':128,'weights_sha256':digest,'durable_report_path':path,
                          'evaluation_seconds':durable['evaluation_seconds'],'queue_seconds':durable['queue_seconds'],'gpu_actual_name':durable['gpu_actual_name']}
        summary['evaluations'][stage]=dict(memo[digest]);persist()
    try:
        if safe('teacher-input',teacher['manifest'],unpack_weights(teacher['weights_gzip']))!=TEACHER_SHA:raise ValueError('reviewed teacher identity mismatch')
        collect=native('collect',dispatch('collect',{}, {},teacher))
        metadata=json.loads(read('collect','dataset/dataset.json',INPUT_LIMITS['dataset.json']));validate_collection(metadata,collect)
        demo={name:hashlib.sha256(read('collect','dataset/'+name,INPUT_LIMITS[name])).hexdigest() for name in ('dataset.json','trajectories.npz')}
        if demo['trajectories.npz']!=metadata['sha256']:raise ValueError('published dataset differs from collection')
        lineage={'collection':receipt(collect)};bindings={'demo':demo}
        bc=native('bc',dispatch('bc',bindings,lineage))
        bc_files={name:read('bc','published/'+name,INPUT_LIMITS[name]) for name in ('bc-report.json','final-actor/manifest.json','final-actor/weights.json')}
        bc_report=json.loads(bc_files['bc-report.json']);validate_bc(bc_report,metadata,bc)
        # Validate and score BC only before students; its protected score never enters their arguments.
        score('bc-only',bc_files['final-actor/manifest.json'],bc_files['final-actor/weights.json'],bc_report['final']['weights_sha256'])
        if summary['evaluations']['bc-only']['weights_sha256']!=bc_report['final']['weights_sha256']:raise ValueError('BC report actor hash mismatch')
        lineage['bc']=receipt(bc);bindings['bc']={name:hashlib.sha256(raw).hexdigest() for name,raw in bc_files.items()}
        calls={stage:dispatch(stage,bindings,lineage) for stage in STUDENTS}
        for stage in STUDENTS:
            record=native(stage,calls[stage]);relative='work/runs/'+job(stage)['job_id']+'/final_ckpt.pt'
            raw=read(stage,relative,64*1024*1024);digest=hashlib.sha256(raw).hexdigest();converted=converter.remote(raw)
            if converted['export']['checkpoint_sha256']!=digest:raise ValueError('native conversion provenance mismatch')
            score(stage,converted['manifest'],unpack_weights(converted['weights_gzip']),converted['export']['weights_sha256'])
            (out/stage/'export.json').write_text(json.dumps(converted['export'],indent=2)+'\n')
        summary.update(state='complete',completed_at=time.time(),unique_development_queries=len(memo));persist()
    except Exception as exc:
        summary.update(state='failed',error=type(exc).__name__+': '+str(exc),completed_at=time.time());persist()
    return {'state':summary['state'],'summary_path':EXPERIMENT+'/summary.json'}


@app.local_entrypoint()
def run_bc_exploration_pilot():
    from bc_exploration_contract import validate_admission,validate_budget,CORE_SHA,TEACHER_SHA,TASK,readonly_inputs
    from transport import pack_weights
    sys.path.insert(0,str(ROOT/'src'))
    from automanip.policy_io import load_policy,PolicySchema
    plan_path=ROOT/'deploy/modal/bc-exploration-plan.json';plan=json.loads(plan_path.read_bytes());validate_admission(plan)
    if hashlib.sha256((ROOT/plan['scientific_plan']).read_bytes()).hexdigest()!=plan['scientific_plan_sha256']:raise ValueError('scientific plan changed')
    ledger_path=ROOT/'.runtime/modal-tonight-budget-ledger.json';validate_budget(json.loads(ledger_path.read_bytes()))
    launch_path=ROOT/'.runtime/modal-bc-exploration-pilot-launch.json'
    if launch_path.exists():raise RuntimeError('pilot already submitted; inspect durable outputs')
    if hashlib.sha256((ROOT/'research/discovery/amp_v1.py').read_bytes()).hexdigest()!=CORE_SHA:raise ValueError('core helper changed')
    teacher_path=ROOT/'runs/stackcube-precision-screen-20261027/precision-constant/step-49971200'
    actor=load_policy(teacher_path,PolicySchema(TASK,48,8))
    if actor.weights_sha256!=TEACHER_SHA:raise ValueError('reviewed teacher changed')
    teacher={'manifest':(teacher_path/'manifest.json').read_bytes(),'weights_gzip':pack_weights((teacher_path/'weights.json').read_bytes())}
    programs={};workers={}
    for stage in STAGES:
        item=job(stage);programs[stage]=(ROOT/item['source_path']).read_bytes()
        if hashlib.sha256(programs[stage]).hexdigest()!=item['source_sha256']:raise ValueError('source payload changed')
        mounts={**runtime_mount,'/trainer-output':outputs[stage].with_mount_options(sub_path='/trial')}
        for mount,(producer,sub_path) in readonly_inputs(stage).items():
            mounts[mount]=outputs[producer].with_mount_options(sub_path=sub_path,read_only=True)
        workers[stage]=run_bc_exploration_stage.with_options(timeout=item['wrapper_seconds'],max_containers=2 if stage in STUDENTS else 1,volumes=mounts)
        workers[stage].hydrate()
    evaluator=frozen_evaluate.with_options(gpu='L40S',timeout=120,max_containers=2);evaluator.hydrate()
    call=coordinate_bc_exploration.spawn(programs,teacher,workers,convert_checkpoint,evaluator)
    launch={'experiment_id':EXPERIMENT,'function_call_id':call.object_id,'summary_path':EXPERIMENT+'/summary.json',
            'budget_cap_usd':3.,'teacher_weights_sha256':TEACHER_SHA,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'ledger_snapshot_sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest(),'confirmation_queries':0}
    with launch_path.open('x') as stream:json.dump(launch,stream,indent=2);stream.write('\n')
    print(json.dumps(launch,indent=2))
