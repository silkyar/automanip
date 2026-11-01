"""Prepare a separate proxy-calibration study; earlier failed screens remain failed."""
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
    report=json.loads((ROOT/'artifacts/cumulative/crossfit-value-cpu-v1/report.json').read_text())
    if report['state']!='complete':raise ValueError('No completed CPU evidence')
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
            initial={key:z[key].copy() for key in z.files}
            for key,value in base.net.state_dict().items():
                if not np.array_equal(value.numpy(),z['initial_actor_mean.'+key]):raise ValueError('BC source/init parity differs')
        with np.load(ROOT/('artifacts/cumulative/crossfit-value-cpu-v1/'+str(seed)+'-baseline-directions.npz'),allow_pickle=False) as z:
            for method in ('constant','state'):
                for sign,name in ((1.,'forward'),(-1.,'reverse')):
                    actor=copy.deepcopy(base)
                    direction=torch.from_numpy(z[method+'_direction'].copy())
                    std=torch.from_numpy(initial['initial_actor_logstd'].copy()).exp().flatten().double()
                    with torch.no_grad():
                        actor.net[-1].weight.add_((sign*std[:,None]*direction[:,:-1]).float())
                        actor.net[-1].bias.add_((sign*std*direction[:,-1]).float())
                    key=str(seed)+'/'+method+'_'+name;path=output/'policies'/key
                    export_actor(actor,path);artifact=load_policy(path,schema)
                    expected=NumpyPolicy(artifact,schema).act(examples.tolist())
                    with torch.no_grad():actual=actor(examples).clamp(-1,1).numpy()
                    error=float(np.max(np.abs(actual-np.asarray(expected))))
                    if error>1e-5:raise ValueError('Safe export/inference parity failed')
                    policies[key]=str(path.relative_to(output));parity.append(dict(policy=key,max_action_error=error,weights_sha256=artifact.weights_sha256))
        bad=copy.deepcopy(base)
        bad.net.load_state_dict({key:torch.from_numpy(initial['joint_update_actor_mean.'+key]) for key in bad.net.state_dict()},strict=True)
        path=output/'policies'/str(seed)/'known_harmful';export_actor(bad,path);load_policy(path,schema)
        policies[str(seed)+'/known_harmful']=str(path.relative_to(output))
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output/'policies').rglob('*')) if p.is_file()}
    plan=dict(format='automanip-proxy-calibration-public-probe-v1',created_at=time.time(),execution_authorized=True,
        user_authorization='2026-10-30 continue autoresearch until stopped; explicitly useGPUs without undue cost hesitation',
        purpose='Calibrate an unvalidated proxy screen against actual behavior, not promotion through a failed screen. Prior failed gate results remain unchanged.',
        previous_screen_state='Both MC methods failed the registered CPU screen; this distinct study tests that screen and actual behavior.',
        production_pins=PRODUCTION_PINS,policies=policies,policy_files=files,safe_export_parity=parity,
        source_actor_weights_sha256=base_artifact.weights_sha256,
        collection_seeds=[90282001,90282002,90282003],diagnostic_seed_bases=[93600000,93601000,93602000],
        noise_seed_bases=[93610000,93611000,93612000],gaussian_std=torch.from_numpy(initial['initial_actor_logstd']).exp().flatten().tolist(),
        modes=['deterministic','stochastic'],
        branch_order=['no_update','known_harmful','constant_forward','constant_reverse','state_forward','state_reverse'],
        vector_environments=64,batches_per_branch=8,episodes_per_branch=512,horizon=50,
        max_public_transitions=921600,max_public_transitions_per_collection=307200,
        optimizer_calls=0,protected_benchmark_queries=0,confirmation_queries=0,final_queries=0,
        reward='Enable unchanged official normalized_dense in this new diagnostic only; original frozen scoring adapter stays reward_mode=none.',
        primary='Stochastic-mode paired discounted return(gamma.9): forward-minus-reverse and forward-minus-unchanged for each MC method; deterministic mode and final success also recorded.',
        controls='Original Gaussian scale fixed. Noise added to RAW mean before clipping; same innovations per resetbatch/step/lane across branches. Deterministic negative-control damage assesses instrumentation.',
        intervals='10000 fixed-seed bootstrap draws over8 vector batches within each of3 fitted directions; conditional reset/noise uncertainty, not independent training-seed uncertainty',
        bootstrap_seed=93609999,
        calibration_rule='Expected harmful mean step must reduce deterministic pooled success by>=.05. A proxy false-negative requires positive98.75%bootstrap lower bounds for stochastic forward-minus-reverse AND forward-minus-unchanged, plus mean final-success loss<=.05. Otherwise calibration is inconclusive; do not call proxy universally validated.',
        comparison='Constant versus state baseline uses identicalMCtargets/data/features/curvature/KL; preserve earlier differing fitting ratios and do not select new step sizes from these outcomes.',
        interpretation='One-step actor-direction/proxy calibration of an assisted BC policy. No complete PPO procedures, training-curve comparison or sample-efficiency claim.',
        assistance=dict(teacher_transitions=49971200,teacher_trainer_seconds=2295.408300588,demo_transitions=25600,bc_optimizer_updates=5000),
        timeout_seconds=360,startup_timeout_seconds=60,internal_seconds=320,budget_reservation_usd=1,
        cpu_result_sha256=hashlib.sha256((ROOT/'artifacts/cumulative/crossfit-value-cpu-v1/report.json').read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256((ROOT/'research/proxy_calibration_probe.py').read_bytes()).hexdigest(),
        followup='Reassess screen and learning hypothesis; no automatic full training, retries or holdout query.')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(policies=len(policies),safe_files=len(files),max_export_error=max(p['max_action_error'] for p in parity),max_public_transitions=921600,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
