"""Export fixed CPU-probe directions to safe actors, then preregister assessment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main(output):
    import numpy as np
    import torch
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.stackcube import STACK_TASK
    from automanip.numpy_inference import NumpyPolicy
    from research.discovery.amp_v1 import read_actor,export_actor,PRODUCTION_PINS

    torch.set_num_threads(2)
    report=json.loads((ROOT/'artifacts/cumulative/old-state-metric-cpu-v1/report.json').read_text())
    if report['state']!='complete' or report['gate_passed'] is not True:raise ValueError('No passed mechanism gate')
    output=Path(output);output.mkdir(exist_ok=False)
    source=ROOT/'runs/stackcube-bc-exploration-pilot-20261028/bc-only'
    schema=PolicySchema(STACK_TASK.fingerprint,48,8)
    base_artifact=load_policy(source,schema)
    if base_artifact.weights_sha256!='4bab8b71061341ea985a6bb0384170c66a1b9a963d283942b84a99c2ad42c0a2':raise ValueError('Historical assisted actor differs')
    base,_=read_actor(source);source_policy=output/'policies/no_update';source_policy.mkdir(parents=True)
    for name in ('manifest.json','weights.json'):(source_policy/name).write_bytes((source/name).read_bytes())
    with np.load(ROOT/'runs/stackcube-bc-exploration-pilot-20261028/collect/native/run/trajectories.npz',allow_pickle=False) as data:
        examples=torch.from_numpy(data['observations'][1,:16].copy())
    policies={'no_update':'policies/no_update'};parity=[]
    for seed in (90282001,90282002,90282003):
        with np.load(ROOT/('runs/stackcube-first-update-probe-20261028/run/'+str(seed)+'/update-state.npz'),allow_pickle=False) as z:
            for key,value in base.net.state_dict().items():
                if not np.array_equal(value.numpy(),z['initial_actor_mean.'+key]):raise ValueError('BC source/init parity differs')
        with np.load(ROOT/('artifacts/cumulative/old-state-metric-cpu-v1/'+str(seed)+'-branches.npz'),allow_pickle=False) as z:
            for method in ('current_only','old_state_metric'):
                for sign,name in ((1.,'forward'),(-1.,'reverse')):
                    actor=copy.deepcopy(base)
                    state={key:value+sign*(torch.from_numpy(z[method+'_actor_mean.'+key].copy())-value) for key,value in base.net.state_dict().items()}
                    actor.net.load_state_dict(state,strict=True)
                    key=str(seed)+'/'+method+'_'+name;path=output/'policies'/key
                    export_actor(actor,path);artifact=load_policy(path,schema)
                    expected=NumpyPolicy(artifact,schema).act(examples.tolist())
                    with torch.no_grad():actual=actor(examples).clamp(-1,1).numpy()
                    error=float(np.max(np.abs(actual-np.asarray(expected))))
                    if error>1e-5:raise ValueError('Safe export/inference parity failed')
                    policies[key]=str(path.relative_to(output));parity.append(dict(policy=key,max_action_error=error,weights_sha256=artifact.weights_sha256))
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'policies').rglob('*')) if p.is_file()}
    plan=dict(format='automanip-old-state-direction-public-probe-v1',created_at=time.time(),execution_authorized=True,
        user_authorization='2026-10-30 continue autoresearch until asked to stop; bounded diagnostic after passed CPU gate',
        production_pins=PRODUCTION_PINS,policies=policies,policy_files=files,safe_export_parity=parity,
        source_actor_weights_sha256=base_artifact.weights_sha256,
        collection_seeds=[90282001,90282002,90282003],diagnostic_seed_bases=[93500000,93501000,93502000],
        branch_order=['no_update','current_only_forward','current_only_reverse','old_state_metric_forward','old_state_metric_reverse'],
        vector_environments=64,batches_per_branch=8,episodes_per_branch=512,horizon=50,
        max_public_transitions=384000,optimizer_calls=0,protected_benchmark_queries=0,final_queries=0,
        primary='Equal-collection mean paired dense return: old-state forward minus reverse. Secondary: forward minus unchanged and comparison with current-only forward-minus-reverse.',
        intervals='10000 fixed-seed bootstrap draws over8 vector batches within each collection, holding3 fitted directions fixed; conditional reset uncertainty, not independent training-seed evidence',
        bootstrap_seed=93509999,
        gate='Positive old-state forward-minus-reverse in>=2/3 collections AND lower95% bootstrap bound>0 pooled; forward-minus-unchanged lower95%bound>0 pooled; mean success loss versus unchanged<=.05. Otherwise no full learning claim/admission.',
        interpretation='Mean-only geometry probe of an assisted BC policy. Same initial critic/noise, matched raw Gaussian KL .01; not complete alternative PPO procedures or from-scratch benchmark.',
        assistance=dict(teacher_transitions=49971200,teacher_trainer_seconds=2295.408300588,demo_transitions=25600,bc_optimizer_updates=5000,prior_state_rows_for_metric=1280),
        timeout_seconds=420,startup_timeout_seconds=60,internal_seconds=375,budget_reservation_usd=.5,
        cpu_gate_sha256=hashlib.sha256((ROOT/'artifacts/cumulative/old-state-metric-cpu-v1/report.json').read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256((ROOT/'research/old_state_direction_probe.py').read_bytes()).hexdigest(),
        followup='Inspect fixed diagnostic outcomes; no automatic large training or protected holdout. Preserve every branch/error and all costs.')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(policies=len(policies),safe_files=len(files),max_export_error=max(p['max_action_error'] for p in parity),max_public_transitions=384000,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
