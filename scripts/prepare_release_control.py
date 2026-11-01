"""Stage two existing safe policies for a bounded manual intervention probe."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run=ROOT/'.runtime/release-control-20261030';inputs=run/'inputs';inputs.mkdir(parents=True,exist_ok=False)
    original=json.loads((ROOT/'cumulative/experiments/scratch-credit-v1.json').read_text())
    compute=json.loads((ROOT/'artifacts/cumulative/scratch-credit-20261030/learner-compute.json').read_text())['trials']
    policies={};files={};std={};prefixes=[]
    for index,seed in [(2,2),(4,3)]:
        receipt=json.loads((ROOT/f'artifacts/cumulative/scratch-credit-20261030/training/trial-{index}/receipt.json').read_text())
        point=next(c for c in receipt['checkpoints'] if c['steps']==8192000)
        source=ROOT/f'artifacts/cumulative/scratch-credit-20261030/training/trial-{index}/work'/point['actor']['path']
        target=inputs/f'policies/{seed}';target.mkdir(parents=True)
        for name in ['manifest.json','weights.json']:
            shutil.copyfile(source/name,target/name);files[str((target/name).relative_to(inputs))]=digest(target/name)
        if files[f'policies/{seed}/weights.json']!=point['actor']['weights_sha256']:raise ValueError('Source artifact identity differs')
        policies[str(seed)]=f'policies/{seed}'
        std[str(seed)]=[]
        assert compute[index]['method']=='ppo-gae' and compute[index]['seed']==seed
        for dim in range(8):
            value=[p['value'] for p in compute[index]['scalars'][f'audit/logstd_dim_{dim}'] if p['steps']==8192000]
            if len(value)!=1:raise ValueError('Gaussian state missing at source export')
            std[str(seed)].append(math.exp(value[0]))
        prefixes.append(dict(training_seed=seed,environment_transitions=8192000,checkpoint_trainer_seconds=point['trainer_seconds'],weights_sha256=point['actor']['weights_sha256'],source_acquisition='Already counted completed from-scratch training run; not new probe training'))
    plan=dict(format='automanip-release-control-public-probe-v1',created_at=time.time(),execution_authorized=True,user_authorization='Continue autonomous research until stopped; GPU cost explicitly not an issue',production_pins=original['training_plan']['production_pins'],collection_seeds=[2,3],diagnostic_seed_bases=[95200000,95201000],noise_seed_bases=[95210000,95211000],policies=policies,policy_files=files,gaussian_std=std,source_prefixes=prefixes,modes=['deterministic','stochastic'],branch_order=['no_update','scheduled_open','targeted_open','targeted_open_zero_arm'],max_public_transitions_per_collection=204800,max_public_transitions=409600,optimizer_calls=0,confirmation_queries=0,final_queries=0,partition='public-diagnostic-not-learner-input',hypothesis='Rare opening, rather than inability to reach stacking geometry, is a material bottleneck for two failed8.192MGAE policies.',intervention='Exactly five+1gripper commands: fixedsteps25..29; first aligned/narrow-gripper observation; same target plus seven zero joint-delta commands. Target conditions fall back tosteps45..49 so allassigned episodes receive5commands. No resampling or episode exclusion.',eligibility='Existing48observations only: cube XYdistance<=sqrt(2)*.02+.005, vertical difference.04+/-.005, qpos7+qpos8<=.042. This is a geometric/narrow-finger proxy, not a true grasp measurement.',primary='Stochastic final physical success, paired over all512assigned episodes per fixed source. Targeted-minus-unchanged and targeted-zero-arm-minus-unchanged require mean gain>=.20 and positive98.75%interval lower bound.',secondary='Targeted-minus-scheduled timing gain requires>=.10 and positive98.75%lower bound. Zero-arm-minus-targeted isolates added arm action control. Deterministic scores and stage completion are diagnostic.',inference='10000bootstrapdraws,seed95209999; resample8whole64episode vector batches within each fixed source. Four registered contrasts use98.75%intervals. No training-seed-population or learned-policy inference.',validity='Same initial reset and pre-intervention observations must match unchanged baseline within1e-5; common Gaussian draws consumed even when overridden; exact5command budget; all work counted. Stop on execution or pairing failure.',falsification='If neither target intervention meets the primary material screen, the specified opening repair is insufficient. Do not infer absence of every possible release bottleneck. Hand-specified success cannot establish sample-efficient learning.',followup='Reassess before any training. Any learned correction needs separate training data and explicit assistance accounting; public diagnostic trajectories never become learner input.',scope='Manual causal intervention only; preserve all frozen benchmark artifacts and policy scores.')
    raw=(json.dumps(plan,indent=2,allow_nan=False)+'\n').encode();(inputs/'plan.json').write_bytes(raw)
    (ROOT/'cumulative/experiments/release-control-probe-v1.json').write_bytes(raw)
    names=['research/proxy_calibration_probe.py','research/release_control_policy.py','research/release_control_probe.py','src/automanip/stackcube_adapter.py','src/automanip/evaluator.py','src/automanip/policy_io.py','src/automanip/numpy_inference.py','src/automanip/stackcube.py']
    admission=dict(format='automanip-release-control-admission-v1',execution_authorized=True,created_at=time.time(),input_plan_sha256=hashlib.sha256(raw).hexdigest(),sources={n:digest(ROOT/n) for n in names},adapter_sha256=digest(ROOT/'deploy/modal/release_control.py'),max_concurrent_gpu_workers=2,max_public_transitions=409600,training_updates=0,timeout_seconds_per_worker=300,retries=0,analysis_sha256=digest(ROOT/'scripts/analyze_release_control.py'))
    (run/'preflight-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    print(json.dumps(dict(state='prepared',policies=policies,max_public_transitions=409600,optimizer_calls=0)))


if __name__=='__main__':main()
