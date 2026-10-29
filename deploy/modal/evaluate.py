"""Trusted once-only frozen evaluation adapter. No confirmation is admitted by default."""
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app,validation_image,results,ROOT
from smoke_capg import runtime_mount,coordinator_image

admission_image=coordinator_image.add_local_file(ROOT/'deploy/modal/promotion-allowlist.json',
                                                '/opt/adapter/promotion-allowlist.json',copy=True)
recovery_image=admission_image.add_local_file(ROOT/'deploy/modal/full-horizon-recovery-20261027.json',
                                              '/opt/adapter/recovery-admission.json',copy=True)
evaluation_image=validation_image.add_local_file(ROOT/'deploy/modal/transport.py','/opt/automanip-modal/transport.py',copy=True).add_local_file(ROOT/'deploy/modal/promotion-allowlist.json',
                                                '/opt/adapter/promotion-allowlist.json',copy=True)


def evaluation_key(protocol,weights,split):
    import re
    if split not in ('development','confirmation'):raise ValueError('unsupported split')
    if not re.fullmatch('[a-f0-9]{64}',protocol) or not re.fullmatch('[a-f0-9]{64}',weights):raise ValueError('invalid digest')
    return 'frozen-evaluation-'+protocol+'-'+weights+'-'+split


@app.function(image=evaluation_image,cpu=.125,memory=512,timeout=60,retries=0,
              block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def adapter_preflight():
    import json
    from pathlib import Path
    import sys
    import modal
    sys.path.insert(0,'/opt/automanip-modal');sys.path.insert(0,'/opt/automanip/src')
    from transport import pack_weights,unpack_weights
    from automanip.stackcube import check_frozen_protocol
    from automanip.policy_io import load_policy
    assert unpack_weights(pack_weights(b'roundtrip'))==b'roundtrip'
    admission=json.loads(Path('/opt/adapter/promotion-allowlist.json').read_text())
    protocol=check_frozen_protocol()
    assert admission['protocol_sha256']==protocol
    # Synthetic service readiness key, never a benchmark/policy claim.
    canary=modal.Dict.from_name('automanip-modal-adapter-readiness',create_if_missing=True)
    canary.put('trusted-rpc-canary',True)
    assert canary.pop('trusted-rpc-canary') is True
    return {'passed':True,'transport_import':True,'frozen_loader_import':True,
            'trusted_claim_rpc_with_network_blocked':True,
            'protocol_sha256':protocol,'confirmation_admitted_count':len(admission['confirmation_weights_sha256']),
            'episodes_evaluated':0,'claims_created':0}


@app.function(image=evaluation_image,gpu='L4',cpu=(3,4),memory=(16384,16384),timeout=1200,startup_timeout=30,retries=0,
              block_network=True,single_use_containers=True,max_containers=3,
              include_source=False,serialized=True,volumes={'/gpu-results':results},
              _experimental_restrict_output=True)
def frozen_evaluate(manifest:bytes,weights_gzip:bytes,split:str,gpu_requested:str='L4',submitted_at:float|None=None,evaluation_timeout_seconds:int=1100):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    import time
    import hashlib
    import modal
    if split not in ('development','confirmation'):raise ValueError('unsupported split')
    if type(evaluation_timeout_seconds) is not int or not 1<=evaluation_timeout_seconds<=1100:raise ValueError('invalid evaluation timeout')
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    if len(manifest)>8192:raise ValueError('manifest byte limit')
    raw=unpack_weights(weights_gzip)
    digest=hashlib.sha256(raw).hexdigest()
    if json.loads(manifest).get('weights_sha256')!=digest:raise ValueError('artifact digest mismatch')
    admission=json.loads(Path('/opt/adapter/promotion-allowlist.json').read_text())
    if split=='confirmation' and digest not in admission['confirmation_weights_sha256']:
        raise PermissionError('confirmation digest not approved')
    # This fixed trusted function needs only its own attempt-claim RPC. Trainer
    # functions remain restricted and cannot call it. Claim inside the GPU
    # function prevents platform preemption from silently repeating episodes.
    attempts=modal.Dict.from_name('automanip-modal-frozen-evaluation-attempts',create_if_missing=True)
    attempt_key=admission['protocol_sha256']+'-'+digest+'-'+split
    started_at=time.time()
    if gpu_requested not in ('L4','L40S'):raise ValueError('unsupported evaluator GPU')
    record={'state':'started','protocol_sha256':admission['protocol_sha256'],
            'weights_sha256':digest,'split':split,'started_at':started_at,
            'submitted_at':submitted_at,'gpu_requested':gpu_requested,
            'queue_seconds':None if submitted_at is None else max(0.,started_at-submitted_at),
            'expected_episodes':128 if split=='development' else 512}
    if not attempts.put(attempt_key,record,skip_if_exists=True):
        raise RuntimeError('GPU evaluation attempt already claimed; no automatic replay')
    directory=Path('/gpu-results/frozen-evaluation-attempts')/admission['protocol_sha256']/digest/split
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'result.json'
    volume=modal.Volume.from_name('automanip-modal-validation')
    def persist():
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        temporary.replace(target)
        volume.commit()
    # Persist the attempt before executing any benchmark episode. A hard kill
    # leaves this started record and the global claim, never a replay allowance.
    persist()
    try:
        import torch
        import shutil
        cache=Path('/gpu-results/runtime/sapien-cache')
        physx=cache/'physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so'
        if hashlib.sha256(physx.read_bytes()).hexdigest()!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':
            raise RuntimeError('pinned PhysX cache digest mismatch')
        shutil.copytree(cache,Path.home()/'.sapien',dirs_exist_ok=True)
        record['gpu_actual_name']=torch.cuda.get_device_name(0)
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);policy=out/'policy';policy.mkdir()
            (policy/'manifest.json').write_bytes(manifest);(policy/'weights.json').write_bytes(raw)
            env=dict(os.environ,PYTHONPATH='/opt/automanip/src')
            started=time.monotonic()
            result=subprocess.run([sys.executable,'/opt/automanip/scripts/evaluate_stackcube.py',str(policy),
                                   '--split',split,'--output',str(out/'evaluation.json')],env=env,
                                  capture_output=True,text=True,timeout=evaluation_timeout_seconds)
            record['evaluation_seconds']=time.monotonic()-started
            if result.returncode:raise RuntimeError('frozen evaluation failed: '+result.stderr[-2000:])
            report=json.loads((out/'evaluation.json').read_text())
            expected=128 if split=='development' else 512
            if report['split']!=split or report['smoke'] or report['episode_count']!=expected or report['failed_episodes']:
                raise RuntimeError('invalid frozen evaluation report')
            if report['weights_sha256']!=digest or report['protocol_sha256']!=admission['protocol_sha256']:
                raise RuntimeError('frozen report provenance mismatch')
            record.update(state='complete',report=report,completed_at=time.time())
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),completed_at=time.time())
    finally:
        # Durable GPU output precedes transport/CPU coordination. The GPU claim
        # key never changes and is never deleted, including on failure.
        persist()
        attempts.put(attempt_key,{k:v for k,v in record.items() if k!='report'})
    if record['state']!='complete':raise RuntimeError(record['error'])
    return {'state':'complete','weights_sha256':digest,'protocol_sha256':admission['protocol_sha256'],
            'evaluation_seconds':record['evaluation_seconds'],
            'gpu_requested':gpu_requested,'gpu_actual_name':record['gpu_actual_name'],
            'queue_seconds':record['queue_seconds'],
            'gpu_durable_result_path':str(target.relative_to('/gpu-results'))}



@app.function(image=admission_image,cpu=.125,memory=512,timeout=5400,retries=0,
              single_use_containers=True,include_source=False,serialized=True,max_containers=3,
              volumes={'/results':results},_experimental_restrict_output=True)
def evaluate_once(manifest:bytes,weights_gzip:bytes,split:str,evaluator_function:modal.Function,gpu_requested:str='L4'):
    import hashlib
    import json
    from pathlib import Path
    import re
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    if split not in ('development','confirmation'):raise ValueError('unsupported split')
    if len(manifest)>8192:raise ValueError('manifest byte limit')
    metadata=json.loads(manifest);digest=hashlib.sha256(unpack_weights(weights_gzip)).hexdigest()
    if metadata.get('weights_sha256')!=digest:raise ValueError('artifact digest mismatch')
    approval=json.loads(Path('/opt/adapter/promotion-allowlist.json').read_text())
    protocol=approval['protocol_sha256']
    if split=='confirmation' and digest not in approval['confirmation_weights_sha256']:
        raise PermissionError('confirmation policy digest has not been admitted by main review')
    if not re.fullmatch('[a-f0-9]{64}',digest) or not re.fullmatch('[a-f0-9]{64}',protocol):raise ValueError('invalid digest')
    key='frozen-evaluation-'+protocol+'-'+digest+'-'+split
    claims=modal.Dict.from_name('automanip-modal-evaluation-claims',create_if_missing=True)
    if not claims.put(key,{'weights_sha256':digest,'split':split,'started_at':time.time()},skip_if_exists=True):
        raise RuntimeError('this policy/split is already claimed; no silent repeat evaluation')
    record={'weights_sha256':digest,'split':split,'protocol_sha256':protocol,'state':'started',
            'claimed_episode_count':128 if split=='development' else 512,'confirmation_claimed':split=='confirmation'}
    directory=Path('/results/frozen-evaluations')/digest;directory.mkdir(parents=True,exist_ok=True)
    target=directory/(split+'.json')
    try:
        result=evaluator_function.remote(manifest,weights_gzip,split,gpu_requested,time.time())
        expected_path='frozen-evaluation-attempts/'+protocol+'/'+digest+'/'+split+'/result.json'
        if result['gpu_durable_result_path']!=expected_path:raise RuntimeError('unexpected durable result path')
        modal.Volume.from_name('automanip-modal-validation').reload()
        durable=json.loads((Path('/results')/expected_path).read_text())
        if durable['state']!='complete':raise RuntimeError('GPU durable result incomplete')
        report=durable['report']
        if report['weights_sha256']!=digest or report['protocol_sha256']!=protocol:raise RuntimeError('result provenance mismatch')
        record.update(**result,report=report)
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    return {'state':record['state'],'weights_sha256':digest,'split':split,'result_path':str(target.relative_to('/results'))}


@app.local_entrypoint()
def evaluate_artifact(policy_dir:str,split:str='development'):
    from transport import pack_weights
    policy=Path(policy_dir)
    call=evaluate_once.spawn((policy/'manifest.json').read_bytes(),pack_weights((policy/'weights.json').read_bytes()),split,frozen_evaluate)
    print(json.dumps({'function_call_id':call.object_id,'split':split,'confirmation_admission':'explicit static digest allowlist'},indent=2))


@app.local_entrypoint()
def check_adapter():
    result=adapter_preflight.remote()
    (ROOT/'.runtime/modal-eval-adapter-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


@app.local_entrypoint()
def confirm_full_horizon():
    import hashlib
    from transport import pack_weights
    prereg_path=ROOT/'deploy/modal/full-horizon-promotion-20261027.json'
    prereg=json.loads(prereg_path.read_text())
    if prereg['split']!='confirmation' or prereg['total_planned_episodes']!=1536:
        raise ValueError('unexpected promotion preregistration')
    expected=json.loads((ROOT/'deploy/modal/promotion-allowlist.json').read_text())
    if expected['confirmation_weights_sha256']!=[p['weights_sha256'] for p in prereg['policies']]:
        raise ValueError('admission differs from preregistration')
    directory=ROOT/'.runtime/modal-full-horizon-promotion';directory.mkdir(exist_ok=True)
    launch_path=directory/'launch.json'
    if launch_path.exists():raise RuntimeError('promotion already submitted; inspect existing calls instead of re-querying')
    launch={'candidate_id':prereg['candidate_id'],'preregistration_sha256':hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
            'split':'confirmation','calls':[],'new_training_runs':0}
    # Persist each submitted call immediately so partial submission never loses its IDs.
    launch_path.write_text(json.dumps(launch,indent=2)+'\n')
    for item in prereg['policies']:
        policy=ROOT/item['policy_dir'];raw=(policy/'weights.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['weights_sha256']:raise ValueError('immutable policy digest changed')
        call=evaluate_once.spawn((policy/'manifest.json').read_bytes(),pack_weights(raw),'confirmation',frozen_evaluate)
        launch['calls'].append(dict(seed=item['seed'],weights_sha256=item['weights_sha256'],function_call_id=call.object_id,
                                    result_path='frozen-evaluations/'+item['weights_sha256']+'/confirmation.json'))
        launch_path.write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps({'submitted':len(launch['calls']),'launch_record':str(launch_path),'new_training_runs':0},indent=2))


@app.function(image=recovery_image,cpu=.125,memory=512,timeout=5400,retries=0,
              single_use_containers=True,include_source=False,serialized=True,max_containers=1,
              volumes={'/results':results},_experimental_restrict_output=True)
def coordinate_recovery(payloads:list,evaluator_function:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/automanip-modal')
    from transport import unpack_weights
    approval=json.loads(Path('/opt/adapter/recovery-admission.json').read_text())
    if not approval['root_authorized'] or approval['gpu_requested']!='L40S':raise PermissionError('recovery not admitted')
    expected=approval['policies'];protocol=approval['protocol_sha256']
    if len(payloads)!=2 or [p['seed'] for p in payloads]!=[p['seed'] for p in expected]:
        raise ValueError('recovery requires exactly the two never-started policies')
    attempts=modal.Dict.from_name('automanip-modal-frozen-evaluation-attempts')
    original_claims=modal.Dict.from_name('automanip-modal-evaluation-claims')
    recoveries=modal.Dict.from_name('automanip-modal-evaluation-recovery-claims',create_if_missing=True)
    directory=Path('/results/evaluation-recoveries')/approval['recovery_id'];directory.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('automanip-modal-validation')
    records=[];calls=[]
    def persist():
        (directory/'summary.json').write_text(json.dumps({'recovery_id':approval['recovery_id'],'gpu_requested':'L40S',
            'previous_app_id':approval['previous_app_id'],'records':records},indent=2,allow_nan=False)+'\n')
        volume.commit()
    # Complete all admissions before submitting GPU work. No original claim is
    # deleted or refreshed. The unchanged GPU claim is the final race arbiter.
    for payload,item in zip(payloads,expected):
        digest=hashlib.sha256(unpack_weights(payload['weights_gzip'])).hexdigest()
        if digest!=item['weights_sha256'] or json.loads(payload['manifest']).get('weights_sha256')!=digest:
            raise ValueError('recovery artifact mismatch')
        if digest==approval['excluded_completed_weights_sha256']:raise PermissionError('completed seed forbidden')
        gpu_key=protocol+'-'+digest+'-confirmation'
        if attempts.get(gpu_key) is not None:raise RuntimeError('GPU attempt exists; recovery forbidden')
        old=original_claims.get('frozen-evaluation-'+protocol+'-'+digest+'-confirmation')
        if not old or old['weights_sha256']!=digest or old['split']!='confirmation':raise RuntimeError('original admission missing')
        record={'seed':item['seed'],'weights_sha256':digest,'split':'confirmation','protocol_sha256':protocol,
                'recovery_id':approval['recovery_id'],'state':'admitted','admitted_at':time.time(),
                'original_admission_claim':old,'original_gpu_claim_absent':True,'gpu_requested':'L40S'}
        if not recoveries.put(approval['recovery_id']+'-'+gpu_key,record,skip_if_exists=True):
            raise RuntimeError('recovery admission already claimed')
        records.append(record)
    persist()
    for payload,record in zip(payloads,records):
        key=protocol+'-'+record['weights_sha256']+'-confirmation'
        if attempts.get(key) is not None:raise RuntimeError('late original GPU claim; no recovery replay')
        record.update(state='submitted',submitted_at=time.time())
        call=evaluator_function.spawn(payload['manifest'],payload['weights_gzip'],'confirmation','L40S',record['submitted_at'])
        record['function_call_id']=call.object_id;calls.append(call);persist()
    for call,record in zip(calls,records):
        try:
            result=call.get()
            expected_path='frozen-evaluation-attempts/'+protocol+'/'+record['weights_sha256']+'/confirmation/result.json'
            if result['gpu_durable_result_path']!=expected_path:raise RuntimeError('unexpected durable recovery path')
            volume.reload()
            durable=json.loads((Path('/results')/expected_path).read_text())
            if durable['state']!='complete':raise RuntimeError('GPU durable recovery incomplete')
            report=durable['report']
            if report['weights_sha256']!=record['weights_sha256'] or report['protocol_sha256']!=protocol:
                raise RuntimeError('recovery result provenance mismatch')
            if report['episode_count']!=512 or report['failed_episodes']:raise RuntimeError('invalid recovery report')
            record.update(**result,completed_at=time.time(),report=report)
        except Exception as exc:
            record.update(state='failed',completed_at=time.time(),error=type(exc).__name__+': '+str(exc))
        persist()
    return {'recovery_id':approval['recovery_id'],'states':[r['state'] for r in records],
            'summary_path':str((directory/'summary.json').relative_to('/results'))}


@app.local_entrypoint()
def recover_remaining_full_horizon():
    import hashlib
    from transport import pack_weights
    plan_path=ROOT/'deploy/modal/full-horizon-recovery-20261027.json';plan=json.loads(plan_path.read_text())
    launch_path=ROOT/'.runtime/modal-full-horizon-recovery-launch.json'
    if launch_path.exists():raise RuntimeError('recovery already submitted; inspect existing work')
    payloads=[]
    for item in plan['policies']:
        policy=ROOT/item['policy_dir'];raw=(policy/'weights.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['weights_sha256']:raise RuntimeError('immutable policy changed')
        payloads.append({'seed':item['seed'],'manifest':(policy/'manifest.json').read_bytes(),'weights_gzip':pack_weights(raw)})
    gpu_function=frozen_evaluate.with_options(gpu='L40S');gpu_function.hydrate()
    call=coordinate_recovery.spawn(payloads,gpu_function)
    launch={'recovery_id':plan['recovery_id'],'function_call_id':call.object_id,'gpu_requested':'L40S',
            'admission_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'seeds':[p['seed'] for p in plan['policies']],'new_training_runs':0,
            'summary_path':'evaluation-recoveries/'+plan['recovery_id']+'/summary.json'}
    launch_path.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch,indent=2))


@app.function(image=recovery_image,cpu=.125,memory=512,timeout=60,retries=0,
              single_use_containers=True,include_source=False,serialized=True,
              volumes={'/results':results},_experimental_restrict_output=True)
def reconcile_durable_recovery():
    """Recover transport only: no GPU handle or policy-execution path exists."""
    import hashlib
    import json
    from pathlib import Path
    import time
    import modal
    plan=json.loads(Path('/opt/adapter/recovery-admission.json').read_text())
    directory=Path('/results/evaluation-recoveries')/plan['recovery_id']
    original=json.loads((directory/'summary.json').read_text())
    records=[]
    for item in plan['policies']:
        relative='frozen-evaluation-attempts/'+plan['protocol_sha256']+'/'+item['weights_sha256']+'/confirmation/result.json'
        raw=(Path('/results')/relative).read_bytes();gpu=json.loads(raw);report=gpu['report']
        if gpu['state']!='complete' or report['weights_sha256']!=item['weights_sha256'] or report['protocol_sha256']!=plan['protocol_sha256']:
            raise RuntimeError('durable report provenance mismatch')
        if report['split']!='confirmation' or report['episode_count']!=512 or report['failed_episodes'] or report['smoke']:
            raise RuntimeError('invalid durable confirmation report')
        old=next(r for r in original['records'] if r['seed']==item['seed'])
        records.append({'seed':item['seed'],'weights_sha256':item['weights_sha256'],'state':'complete',
                        'successes':report['successes_at_end'],'episodes':512,'failed_episodes':0,
                        'gpu_durable_result_path':relative,'gpu_record_sha256':hashlib.sha256(raw).hexdigest(),
                        'original_coordinator_state':old['state'],'original_transport_error':old.get('error'),
                        'gpu_requested':gpu['gpu_requested'],'gpu_actual_name':gpu['gpu_actual_name'],
                        'evaluation_seconds':gpu['evaluation_seconds'],'queue_seconds':gpu['queue_seconds']})
    result={'recovery_id':plan['recovery_id'],'state':'reconciled_from_committed_gpu_reports',
            'new_confirmation_episodes_executed':0,'new_training_runs':0,'recorded_at':time.time(),'records':records}
    target=directory/'reconciled-summary.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    modal.Volume.from_name('automanip-modal-validation').commit()
    return {'state':result['state'],'new_confirmation_episodes_executed':0,
            'summary_path':str(target.relative_to('/results'))}


@app.local_entrypoint()
def reconcile_transport():
    result=reconcile_durable_recovery.remote()
    (ROOT/'.runtime/modal-full-horizon-transport-reconciliation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
