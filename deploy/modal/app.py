"""Trusted Modal L4 validation. No autonomous candidate training is exposed yet."""
import json
import os
from pathlib import Path
import sys
import uuid

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bundle import ROOT, build_context

BASE='pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee'
SDK_VERSION='1.6.2'
GPU='L4'
STAGE=ROOT/'.runtime/modal-image-context'
validation_context=build_context('validation',STAGE/'validation')
trainer_context=build_context('trainer',STAGE/'trainer')

common=(modal.Image.from_registry(BASE,add_python='3.11')
    .entrypoint([])
    .apt_install('libegl1','libgl1','libglib2.0-0','libxext6','libvulkan1','vulkan-tools','ca-certificates')
    .add_local_file(ROOT/'requirements/linux-cu124.txt','/opt/locks/linux-cu124.txt',copy=True)
    .add_local_file(ROOT/'requirements/stackcube-extra.txt','/opt/locks/stackcube-extra.txt',copy=True)
    .run_commands('python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /opt/locks/linux-cu124.txt -r /opt/locks/stackcube-extra.txt',
                  'python -m pip check')
    .add_local_file(ROOT/'deploy/modal/transport.py','/opt/automanip-modal/transport.py',copy=True)
    .env({'NVIDIA_DRIVER_CAPABILITIES':'compute,utility,graphics','OMP_NUM_THREADS':'4',
          'OPENBLAS_NUM_THREADS':'1','MS_SKIP_ASSET_DOWNLOAD_PROMPT':'1','PYTHONDONTWRITEBYTECODE':'1'}))
validation_image=common.add_local_dir(validation_context,'/opt/automanip',copy=True)
trainer_image=common.add_local_dir(trainer_context,'/opt/trainer',copy=True)
app=modal.App('automanip-modal-validation',include_source=False)
results=modal.Volume.from_name('automanip-modal-validation',create_if_missing=True,version=2)


@app.function(image=validation_image,gpu=GPU,cpu=(3,4),memory=(16384,16384),
              timeout=600,startup_timeout=600,retries=0,max_containers=1,
              single_use_containers=True,scaledown_window=2,include_source=False,
              volumes={'/results':results},serialized=True,_experimental_restrict_output=True)
def validate(run_id:str,manifest:bytes,weights:bytes,reference_json:bytes,gpu_requested:str='L4'):
    import ctypes
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import subprocess
    import shutil
    import sys
    import time
    import traceback
    import modal

    if not run_id.isalnum() or len(run_id)>40:raise ValueError('invalid run identifier')
    # Server-side claim survives preemption and refuses automatic repeat execution.
    claims=modal.Dict.from_name('automanip-modal-validation-claims',create_if_missing=True)
    if not claims.put(run_id,{'started_at':time.time()},skip_if_exists=True):
        raise RuntimeError('validation attempt already claimed; no automatic retry')
    started=time.time();out=Path('/results')/run_id;out.mkdir(exist_ok=False)
    if gpu_requested not in ('L4','L40S'):raise ValueError('unsupported compatibility GPU')
    report={'run_id':run_id,'started_at':started,'passed':False,'gpu_requested':gpu_requested,
            'confirmation_queries':0,'validation_timeout_seconds':600,'training_started':False}
    try:
        root=Path('/opt/automanip');sys.path.insert(0,str(root/'src'))
        from automanip.stackcube import STACK_TASK,check_frozen_protocol
        from automanip.policy_io import load_policy,PolicySchema,MAX_MANIFEST_BYTES,MAX_WEIGHTS_BYTES
        report['protocol_sha256']=check_frozen_protocol()
        if len(manifest)>MAX_MANIFEST_BYTES or len(weights)>MAX_WEIGHTS_BYTES:raise ValueError('policy size limit')
        artifact=out/'policy';artifact.mkdir();(artifact/'manifest.json').write_bytes(manifest);(artifact/'weights.json').write_bytes(weights)
        policy=load_policy(artifact,PolicySchema(STACK_TASK.fingerprint,48,8))
        report['weights_sha256']=policy.weights_sha256
        import torch
        report['versions']={name:importlib.metadata.version(name) for name in ['torch','mani_skill','sapien','numpy','gymnasium']}
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if not torch.__version__.startswith('2.6.0') or torch.version.cuda!='12.4':raise RuntimeError('PyTorch/CUDA pin mismatch')
        report['gpu_name']=torch.cuda.get_device_name(0)
        report['cuda_result']=float((torch.ones(64,device='cuda')*2).sum().item())
        if report['cuda_result']!=128.:raise RuntimeError('CUDA arithmetic failed')
        report['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
        env=dict(os.environ,PYTHONPATH=str(root/'src'))
        preflight=subprocess.run([sys.executable,str(root/'scripts/check_stackcube.py')],cwd=root,env=env,
                                 capture_output=True,text=True,timeout=180)
        (out/'preflight.log').write_text(preflight.stdout+preflight.stderr)
        report['preflight_log_tail']=(preflight.stdout+preflight.stderr)[-5000:]
        if preflight.returncode:raise RuntimeError('frozen simulator preflight failed')
        cache=Path.home()/'.sapien'
        versioned=cache/'physx/105.1-physx-5.3.1.patch0'
        if not versioned.is_dir():raise RuntimeError('expected pinned PhysX cache missing')
        runtime=Path('/results/runtime/sapien-cache')
        shutil.copytree(cache,runtime,dirs_exist_ok=True)
        report['physx_cache_sha256']={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in runtime.rglob('*') if p.is_file()}
        from automanip.stackcube_adapter import StackCubeAdapter
        scene=StackCubeAdapter(8,'physx_cuda')
        try:
            scene.reset(list(range(8)))
            pixels=scene.env.render_rgb_array('render_camera').detach().cpu().numpy()
            if pixels.shape!=(8,512,512,3) or float(pixels.std())<1:raise RuntimeError('invalid Vulkan render')
            import imageio.v2 as imageio
            imageio.imwrite(out/'render.png',pixels[0])
            report['render_shape']=list(pixels.shape);report['render_std']=float(pixels.std())
        finally:scene.close()
        evaluation=subprocess.run([sys.executable,str(root/'scripts/evaluate_stackcube.py'),str(artifact),
                                   '--split','development','--output',str(out/'development.json')],cwd=root,env=env,
                                  capture_output=True,text=True,timeout=300)
        (out/'evaluation.log').write_text(evaluation.stdout+evaluation.stderr)
        if evaluation.returncode:raise RuntimeError('frozen development evaluation failed')
        actual=json.loads((out/'development.json').read_text());expected=json.loads(reference_json)
        if actual['weights_sha256']!=expected['weights_sha256'] or actual['protocol_sha256']!=expected['protocol_sha256']:
            raise RuntimeError('reference provenance mismatch')
        if actual['split']!='development' or actual['episode_count']!=128 or actual['failed_episodes']:
            raise RuntimeError('invalid development report')
        mismatches=[i for i,(a,b) in enumerate(zip(actual['episodes'],expected['episodes']))
                    if a['outcome']!=b['outcome'] or a['diagnostics']!=b['diagnostics']]
        report.update(development_success=actual['success_at_end_rate'],reference_success=expected['success_at_end_rate'],
                      parity_mismatch_indices=mismatches,episode_parity_passed=not mismatches)
        if mismatches:raise RuntimeError('cross-provider episode parity mismatch; investigate before migration')
        # Probe Landlock availability only; Modal isolation does not depend on it.
        libc=ctypes.CDLL(None,use_errno=True);abi=libc.syscall(444,0,0,1)
        report['landlock_abi']=abi if abi>=0 else None
        report['passed']=True
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        report['elapsed_seconds']=time.time()-started
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        modal.Volume.from_name('automanip-modal-validation').commit()
    return report


@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=.125,memory=512,
              timeout=5400,retries=0,max_containers=1,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True)
def coordinate_compatibility(run_id:str,manifest:bytes,weights:bytes,reference_json:bytes,
                             gpu_function:modal.Function):
    # The GPU function persists its own evidence before returning. This longer
    # CPU lifetime accommodates capacity queueing without orphaning its result.
    result=gpu_function.remote(run_id,manifest,weights,reference_json,'L40S')
    return {'run_id':run_id,'passed':result['passed'],'summary_path':run_id+'/summary.json',
            'gpu_requested':'L40S','confirmation_queries':0,'training_started':False}


@app.local_entrypoint()
def l40s_compatibility():
    run_id='l40sparity'+uuid.uuid4().hex[:20]
    policy=ROOT/'.runtime/settling-20261027-inputs/anchor-policies/seed-1'
    reference=ROOT/'.runtime/settling-20261027-inputs/anchor-development/seed-1.json'
    gpu_function=validate.with_options(gpu='L40S')
    gpu_function.hydrate()
    call=coordinate_compatibility.spawn(run_id,(policy/'manifest.json').read_bytes(),
        (policy/'weights.json').read_bytes(),reference.read_bytes(),gpu_function)
    launch={'run_id':run_id,'function_call_id':call.object_id,'summary_path':run_id+'/summary.json',
            'gpu_requested':'L40S','confirmation_queries':0,'training_started':False}
    (ROOT/'.runtime/modal-l40s-compatibility-launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))


@app.function(image=trainer_image,gpu=GPU,cpu=(3,4),memory=(16384,16384),timeout=180,retries=0,
              restrict_modal_access=True,block_network=True,single_use_containers=True,
              include_source=False,serialized=True,_experimental_restrict_output=True,
              volumes={'/root/.sapien':results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')})
def boundary_probe(canary:modal.Dict):
    import json
    import os
    from pathlib import Path
    import socket
    import modal

    checks={}
    try:canary.get('trusted-only');checks['modal_rpc_blocked']=False
    except modal.exception.AuthError:checks['modal_rpc_blocked']=True
    try:
        with socket.create_connection(('1.1.1.1',443),timeout=3):pass
        checks['outbound_network_blocked']=False
    except OSError:checks['outbound_network_blocked']=True
    checks['evaluator_source_absent']=not Path('/opt/automanip/scripts/evaluate_stackcube.py').exists()
    checks['benchmark_absent']=not Path('/opt/automanip/benchmarks').exists()
    checks['results_mount_absent']=not Path('/results').exists()
    checks['trainer_present']=Path('/opt/trainer/third_party/maniskill/ppo.py').is_file()
    checks['local_credential_absent']=not Path('/root/.modal.toml').exists()
    checks['long_lived_credentials_absent']=not any(os.environ.get(name) for name in ('RUNPOD_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET'))
    cache=Path('/root/.sapien')
    try:
        (cache/'forbidden-write').write_text('must fail')
        checks['runtime_cache_read_only']=False
    except OSError:checks['runtime_cache_read_only']=True
    import torch
    checks['cuda_arithmetic']=bool(torch.cuda.is_available() and float(torch.ones(1,device='cuda').sum().item())==1.)
    import gymnasium as gym
    import mani_skill.envs
    import numpy as np
    env=gym.make('StackCube-v1',num_envs=8,obs_mode='state',control_mode='pd_joint_delta_pos',
                 sim_backend='physx_cuda',robot_uids='panda_wristcam')
    try:
        obs,_=env.reset(seed=list(range(8)))
        env.step(torch.zeros((8,8),device=env.unwrapped.device))
        checks['restricted_gpu_simulation']=tuple(obs.shape)==(8,48)
    finally:env.close()
    return {'checks':checks,'passed':all(checks.values()),'training_started':False}


@app.local_entrypoint()
def main():
    if modal.__version__!=SDK_VERSION:raise RuntimeError('use pinned isolated Modal SDK')
    run_id='validation'+uuid.uuid4().hex[:20]
    policy=ROOT/'.runtime/settling-20261027-inputs/anchor-policies/seed-1'
    reference=ROOT/'.runtime/settling-20261027-inputs/anchor-development/seed-1.json'
    call=coordinate.spawn(run_id,(policy/'manifest.json').read_bytes(),(policy/'weights.json').read_bytes(),
                          reference.read_bytes(),validate,boundary_probe)
    launch={'run_id':run_id,'function_call_id':call.object_id,'results_volume':'automanip-modal-validation',
            'summary_path':run_id+'/coordinator-summary.json','detached_cloud_orchestration':True}
    destination=ROOT/'.runtime'/('modal-'+run_id+'-launch.json')
    destination.write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))


@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=.125,memory=512,
              timeout=1500,retries=0,max_containers=1,single_use_containers=True,
              include_source=False,serialized=True,volumes={'/results':results},
              _experimental_restrict_output=True)
def coordinate(run_id:str,manifest:bytes,weights:bytes,reference_json:bytes,
               gpu_function:modal.Function,boundary_function:modal.Function):
    """Cloud coordinator keeps all validation stages alive after laptop disconnect."""
    import json
    from pathlib import Path
    import time
    import modal
    claims=modal.Dict.from_name('automanip-modal-validation-claims',create_if_missing=True)
    if not claims.put('coordinator-'+run_id,{'started_at':time.time()},skip_if_exists=True):
        raise RuntimeError('coordinator attempt already claimed; no silent retries')
    result=gpu_function.remote(run_id,manifest,weights,reference_json)
    boundary=(boundary_function.remote(claims) if result['passed'] else
              {'passed':False,'not_run':'GPU validation must pass first'})
    summary={'validation':result,'boundary':boundary,'passed':result['passed'] and boundary['passed'],
             'training_started':False,'confirmation_queries':0}
    volume=modal.Volume.from_name('automanip-modal-validation');volume.reload()
    out=Path('/results')/run_id;out.mkdir(exist_ok=True)
    (out/'coordinator-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    volume.commit()
    return {'run_id':run_id,'passed':summary['passed'],'summary_path':run_id+'/coordinator-summary.json'}
