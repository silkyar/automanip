"""Fixed 50M policies, unchanged frozen scorer, one confirmation each."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import app, ROOT, results
from evaluate import frozen_evaluate, evaluation_image
from smoke_capg import coordinator_image
from precision_confirmation_contract import EXPERIMENT, DIGESTS, validate

PLAN = ROOT/'deploy/modal/precision-confirmation-plan.json'
image = coordinator_image.add_local_file(PLAN,'/opt/precision-confirmation/plan.json',copy=True).add_local_file(
    ROOT/'deploy/modal/precision_confirmation_contract.py','/opt/precision-confirmation/contract.py',copy=True)
# Override only the static digest admission in this function's image. The old
# promotion allowlist, evaluator source, benchmark and protocol remain unchanged.
scoring_image = evaluation_image.add_local_file(PLAN,'/opt/adapter/promotion-allowlist.json',copy=True)


def frozen_confirmation_body(source):
    """Extract the exact trusted function body, with its decorator removed.

    SDK 1.6.2 cannot replace a function's image through with_options. Register
    the identical trusted function against the narrowly admitted image instead.
    This never reads or executes candidate Python.
    """
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='frozen_evaluate')
    node.decorator_list=[]
    namespace={'__name__':__name__}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged-trusted-frozen-evaluator','exec'),namespace)
    return namespace['frozen_evaluate']


frozen_source=(ROOT/'deploy/modal/evaluate.py').read_bytes()
if hashlib.sha256(frozen_source).hexdigest()!=json.loads(PLAN.read_bytes())['frozen_adapter_source_sha256']:
    raise ValueError('Frozen evaluator adapter source changed')
precision_frozen_evaluate=app.function(
    image=scoring_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=240,
    startup_timeout=30,retries=0,block_network=True,single_use_containers=True,
    max_containers=3,include_source=False,serialized=True,volumes={'/gpu-results':results},
    name='precision_frozen_evaluate',_experimental_restrict_output=True)(frozen_confirmation_body(frozen_source))


@app.function(image=image,cpu=.125,memory=512,timeout=1800,retries=0,
              max_containers=1,single_use_containers=True,include_source=False,
              serialized=True,volumes={'/results':results},_experimental_restrict_output=True)
def coordinate_precision_confirmation(payloads:list,evaluator:modal.Function):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time
    import modal
    sys.path.insert(0,'/opt/precision-confirmation')
    sys.path.insert(0,'/opt/automanip-modal')
    from contract import EXPERIMENT, DIGESTS, validate
    from transport import unpack_weights
    plan=json.loads(Path('/opt/precision-confirmation/plan.json').read_bytes())
    validate(plan)
    if len(payloads)!=3:raise ValueError('Exactly three policies required')
    for payload,digest in zip(payloads,DIGESTS):
        if len(payload['manifest'])>8192 or json.loads(payload['manifest']).get('weights_sha256')!=digest:
            raise ValueError('Policy manifest differs from admission')
        if hashlib.sha256(unpack_weights(payload['weights_gzip'])).hexdigest()!=digest:
            raise ValueError('Policy weights differ from admission')
    claims=modal.Dict.from_name('automanip-modal-scientific-job-claims',create_if_missing=True)
    if not claims.put(EXPERIMENT,{'started_at':time.time()},skip_if_exists=True):
        raise RuntimeError('Confirmation campaign already claimed; no replay')
    out=Path('/results')/EXPERIMENT;out.mkdir(exist_ok=False)
    volume=modal.Volume.from_name('automanip-modal-validation')
    summary={'experiment_id':EXPERIMENT,'state':'started','protocol_sha256':plan['protocol_sha256'],
             'split':'confirmation','episodes_per_policy':512,'new_training_runs':0,
             'private_final_holdout':False,'acceptance_rule':plan['acceptance_rule'],
             'records':[],'started_at':time.time()}
    def persist():
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        volume.commit()
    calls=[]
    try:
        for seed,(payload,digest) in enumerate(zip(payloads,DIGESTS),1):
            submitted=time.time()
            row={'seed':seed,'weights_sha256':digest,'state':'submitted','submitted_at':submitted}
            summary['records'].append(row);persist()
            call=evaluator.spawn(payload['manifest'],payload['weights_gzip'],'confirmation','L40S',submitted,220)
            calls.append((call,row));row['function_call_id']=call.object_id;persist()
        for call,row in calls:
            try:
                receipt=call.get();volume.reload()
                expected='frozen-evaluation-attempts/'+plan['protocol_sha256']+'/'+row['weights_sha256']+'/confirmation/result.json'
                if receipt['gpu_durable_result_path']!=expected:raise ValueError('Unexpected durable path')
                target=Path('/results')/expected
                if target.stat().st_size>4*1024*1024:raise ValueError('Result byte limit')
                durable=json.loads(target.read_bytes());report=durable['report']
                if (durable['state']!='complete' or report['protocol_sha256']!=plan['protocol_sha256']
                    or report['weights_sha256']!=row['weights_sha256'] or report['split']!='confirmation'
                    or report['episode_count']!=512 or report['smoke'] or report['failed_episodes']):
                    raise ValueError('Invalid frozen confirmation report')
                row.update(state='complete',report=report,receipt=receipt)
            except Exception as exc:
                row.update(state='failed',error=type(exc).__name__+': '+str(exc))
            persist()
        if not all(row['state']=='complete' for row in summary['records']):
            raise RuntimeError('Incomplete confirmation; preserve results, do not repeat')
        rates=[row['report']['success_at_end_rate'] for row in summary['records']]
        mean=sum(rates)/3
        summary.update(state='complete',mean_success=mean,minimum_seed_success=min(rates),
                       target_met=mean>=.99 and min(rates)>=.98,completed_at=time.time())
    except Exception as exc:
        summary.update(state='failed',error=type(exc).__name__+': '+str(exc),completed_at=time.time())
    finally:persist()
    return {'state':summary['state'],'summary_path':EXPERIMENT+'/summary.json'}


@app.local_entrypoint()
def confirm_precision():
    from transport import pack_weights
    plan=json.loads(PLAN.read_bytes());validate(plan)
    launch_path=ROOT/'.runtime/modal-precision-confirmation-launch.json'
    if launch_path.exists():raise RuntimeError('Already submitted; inspect existing calls')
    if hashlib.sha256((ROOT/'deploy/modal/evaluate.py').read_bytes()).hexdigest()!=plan['frozen_adapter_source_sha256']:
        raise ValueError('Frozen evaluator adapter changed')
    payloads=[]
    for item,digest in zip(plan['policies'],DIGESTS):
        policy=ROOT/item['policy_dir'];raw=(policy/'weights.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Fixed policy changed')
        payloads.append({'manifest':(policy/'manifest.json').read_bytes(),'weights_gzip':pack_weights(raw)})
    evaluator=precision_frozen_evaluate
    evaluator.hydrate()
    call=coordinate_precision_confirmation.spawn(payloads,evaluator)
    launch={'experiment_id':EXPERIMENT,'function_call_id':call.object_id,
            'summary_path':EXPERIMENT+'/summary.json','plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            'budget_cap_usd':1,'new_training_runs':0,'confirmation_episodes':1536}
    launch_path.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch))
