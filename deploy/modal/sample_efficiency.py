"""Sample-efficiency screen: PPO update cadence and observation normalization; training and scoring isolated.

Each trial trains from random initialization and exports safe actors every 1,024,000 transitions.
A separate scorer image (read-only candidate mount, no network) scores every export on 128 fresh
public-development episodes. Protected development/confirmation seeds are never queried here.
"""
import hashlib
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0,str(Path(__file__).resolve().parent))
from app import ROOT, trainer_image, validation_image, results

import os
BATCH=os.environ.get('AUTOMANIP_SE_BATCH','b1')
NAME=f'automanip-sample-efficiency-{BATCH}-20261031'
SOURCE=ROOT/'research/sample-efficiency-20261031'
RUNTIME=ROOT/f'.runtime/sample-efficiency-{BATCH}-20261031'
PUBLIC_SEED_BASE=97100000  # disjoint from protected development (0-127) and confirmation (1,000,000+)
TOTAL=int(os.environ.get('AUTOMANIP_SE_TOTAL','15360000'))
TRAINER_SECONDS=int(os.environ.get('AUTOMANIP_SE_TRAINER_SECONDS','2700'))
CHECKPOINT=int(os.environ.get('AUTOMANIP_SE_CHECKPOINT','1024000'))
ARMS_BY_BATCH={
    'b1':{
        'control':dict(num_envs=1024,obs_norm=False),
        'env512':dict(num_envs=512,obs_norm=False),
        'env256':dict(num_envs=256,obs_norm=False),
        'obsnorm':dict(num_envs=1024,obs_norm=True),
        'obsnorm-env256':dict(num_envs=256,obs_norm=True),
    },
    # Heavy-tailed (Student-t, df 2) exploration; factorial with b1 control / obsnorm-env256 at the same seeds.
    'b2':{
        't-gripper':dict(num_envs=1024,obs_norm=False,heavy_tail_dims='gripper'),
        't-all':dict(num_envs=1024,obs_norm=False,heavy_tail_dims='all'),
        'obsnorm-env256-t-gripper':dict(num_envs=256,obs_norm=True,heavy_tail_dims='gripper'),
        'obsnorm-env256-t-all':dict(num_envs=256,obs_norm=True,heavy_tail_dims='all'),
    },
    # Bernoulli open/close gripper (no flat dead zone); factorial with b1 control / env256, plus a finer cadence probe.
    'b3':{
        'bin':dict(num_envs=1024,obs_norm=False,binary_gripper=True),
        'env256-bin':dict(num_envs=256,obs_norm=False,binary_gripper=True),
        'env128':dict(num_envs=128,obs_norm=False),
    },
    # Bounded gripper exploration mass with a live gradient: Gaussian std floor + mean barrier, or eps-floor Bernoulli + logit L2.
    'b4':{
        'gfloor':dict(num_envs=1024,obs_norm=False,gripper_std_floor=0.5,gripper_mean_barrier=1.0),
        'env256-gfloor':dict(num_envs=256,obs_norm=False,gripper_std_floor=0.5,gripper_mean_barrier=1.0),
        'bineps':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
        'env256-bineps':dict(num_envs=256,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
    },
    # Larger exploration floor, and temporally persistent (sticky) binary gripper exploration; all on the 1024-env base.
    'b5':{
        'bineps05':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.05,binary_logit_l2=1e-3),
        'bineps-sticky50':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,binary_sticky=0.5),
        'bineps-sticky80':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,binary_sticky=0.8),
    },
    # Exact base-yaw symmetry augmentation (probe-verified), on the control and bineps 1024-env bases.
    'b6':{
        'sym':dict(num_envs=1024,obs_norm=False,sym_theta=0.4),
        'bineps-sym':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,sym_theta=0.4),
        'bineps-sym08':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,sym_theta=0.8),
    },
    # Placement-precision clock: release starts once arm exploration std is small; start it smaller or stop resisting its decay.
    'b7':{
        'bineps-std37':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,init_logstd=-1.0),
        'bineps-std22':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,init_logstd=-1.5),
        'bineps-ent0':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,ent_coef=0.0),
    },
    'b8':{
        'env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
    },
    # Off-policy pilot (go/no-go: places and releases by about 3M transitions?).
    'o1':{
        'td3':dict(algo='td3',num_envs=1024,obs_norm=False),
        'td3-burst':dict(algo='td3',num_envs=1024,obs_norm=False,burst_prob=0.005),
        'td3-burst-sym':dict(algo='td3',num_envs=1024,obs_norm=False,burst_prob=0.005,sym_theta=0.4),
    },
    'r3':{
        'td3':dict(algo='td3',num_envs=1024,obs_norm=False),
    },
    'o2':{
        'td3-utd8':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=8),
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16),
    },
    # Transfer to PlaceSphere-v1: research-side deterministic evaluation (ManiSkill success at episode end, 128 fixed fresh seeds).
    't1':{
        'ppo-control':dict(num_envs=1024,obs_norm=False,total=15360000,env_id='PlaceSphere-v1',transfer=True),
        'ppo-env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,total=15360000,env_id='PlaceSphere-v1',transfer=True),
        'td3':dict(algo='td3',num_envs=1024,obs_norm=False,total=7680000,env_id='PlaceSphere-v1',transfer=True),
    },
    # Transfer to PickCube-v1: research-side deterministic evaluation (ManiSkill success at episode end, 128 fixed fresh seeds).
    't2':{
        'ppo-control':dict(num_envs=1024,obs_norm=False,total=15360000,env_id='PickCube-v1',transfer=True),
        'ppo-env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,total=15360000,env_id='PickCube-v1',transfer=True),
        'td3':dict(algo='td3',num_envs=1024,obs_norm=False,total=7680000,env_id='PickCube-v1',transfer=True),
    },
    'r4':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16),
    },
    't3':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PlaceSphere-v1',transfer=True),
    },
    't4':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PickCube-v1',transfer=True),
    },
    't5':{
        'ppo-control-anneal':dict(num_envs=1024,obs_norm=False,anneal_lr=True,env_id='PlaceSphere-v1',transfer=True,total=15360000),
        'ppo-env512-bineps-anneal':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,anneal_lr=True,env_id='PlaceSphere-v1',transfer=True,total=15360000),
    },
    't6':{
        'ppo-control':dict(num_envs=1024,obs_norm=False,total=15360000,env_id='LiftPegUpright-v1',transfer=True),
        'ppo-env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,total=15360000,env_id='LiftPegUpright-v1',transfer=True),
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='LiftPegUpright-v1',transfer=True),
    },
    # Diagnosis: td3-utd16 on PlaceSphere with per-stage counts in the research-side evaluation (seeds 11-13).
    'd1':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PlaceSphere-v1',transfer=True),
    },
    # Ablation on LiftPegUpright: 512-env cadence alone vs the bineps gripper alone (compare t6 control and env512-bineps).
    'a1':{
        'ppo-env512':dict(num_envs=512,obs_norm=False,total=15360000,env_id='LiftPegUpright-v1',transfer=True),
        'ppo-bineps':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,total=15360000,env_id='LiftPegUpright-v1',transfer=True),
    },
    # Prioritized replay (|TD error|^alpha with importance correction) vs uniform replay, StackCube 5.12M.
    'o4':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16),
        'td3-utd16-per':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,per_alpha=0.6),
    },
    # Does prioritized replay fix the PlaceSphere precision frontier found in d1? Stage diagnostics on.
    'd2':{
        'td3-utd16-per':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,per_alpha=0.6,total=7680000,env_id='PlaceSphere-v1',transfer=True),
    },
    # Lesson-transfer test on an unseen task (predictions registered in docs/lesson-transfer-predictions-20261101.md).
    'x1':{
        'ppo-control':dict(num_envs=1024,obs_norm=False,total=15360000,env_id='PokeCube-v1',transfer=True),
        'ppo-env512':dict(num_envs=512,obs_norm=False,total=15360000,env_id='PokeCube-v1',transfer=True),
        'ppo-env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,total=15360000,env_id='PokeCube-v1',transfer=True),
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PokeCube-v1',transfer=True),
    },
    # Wall-clock: compute-only speedups and a late update-ratio drop for td3-utd16 (StackCube, 5.12M).
    'f1':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16),
        'td3-utd16-fast':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,fast=True),
        'td3-utd16-fast-bf16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,fast=True,bf16=True),
        'td3-utd16-fast-late4':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,fast=True,utd_late=4),
    },
    # Reverse curriculum on PlaceSphere td3-utd16 (stalls at 5 mm placement, d1): 25% of episodes start along successful
    # trajectories, moving back as mastered. demo = 5 motion-planned demos (demo-assisted track); self = own first successes.
    'c1':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PlaceSphere-v1',transfer=True),
        'td3-utd16-rc-demo':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PlaceSphere-v1',transfer=True,
                                 rc_source='demo',rc_demos='PlaceSphere-v1-5.json'),
        'td3-utd16-rc-self':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PlaceSphere-v1',transfer=True,
                                 rc_source='self'),
    },
    # Headroom probe on harder tasks: our PPO (ManiSkill's tuned gamma, rollout = horizon) vs td3-utd16, 2 seeds each.
    'h1':{f'{name}-{arm}':cfg for name,env_id,horizon,gamma in [('peg','PegInsertionSide-v1',100,0.8),('pusht','PushT-v1',100,0.99),
                                                                ('tool','PullCubeTool-v1',100,0.8),('plug','PlugCharger-v1',200,0.8)]
          for arm,cfg in [('ppo',dict(num_envs=1024,obs_norm=False,total=30720000,env_id=env_id,transfer=True,num_steps=horizon,gamma=gamma)),
                          ('td3-utd16',dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=10240000,env_id=env_id,
                                            transfer=True,gamma=0.99 if gamma==0.99 else 0.9))]},
    # Does the self (demo-free) reverse curriculum shorten td3-utd16's StackCube discovery waits? Frozen-scorer exports.
    'c2':{
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16),
        'td3-utd16-rc-self':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,rc_source='self'),
        'td3-utd16-rc-demo':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,rc_source='demo',rc_demos='StackCube-v1-5.json'),
    },
    # Smoke only: exercise self-local resets (PlaceSphere finds early successes) and the Peg stage diagnostics.
    'p0':{
        'ps-rc-self-local':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,env_id='PlaceSphere-v1',transfer=True,rc_source='self-local'),
        'peg-td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,env_id='PegInsertionSide-v1',transfer=True),
    },
    # PegInsertionSide (h1: PPO ~0/128 at 30.7M; td3-utd16 75-82/128 at 10.24M, still rising): per-env self curriculum.
    'p1':{
        'peg-td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=15360000,env_id='PegInsertionSide-v1',transfer=True),
        'peg-td3-utd16-rc-self-local':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=15360000,
                                           env_id='PegInsertionSide-v1',transfer=True,rc_source='self-local',rc_frac=0.5),
        'peg-td3-utd16-g95':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=15360000,env_id='PegInsertionSide-v1',
                                 transfer=True,gamma=0.95),
    },
    # Fresh-seed replication of the p1 winner (PegInsertionSide, gamma 0.95).
    'r7':{
        'peg-td3-utd16-g95':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=15360000,env_id='PegInsertionSide-v1',
                                 transfer=True,gamma=0.95),
    },
    # Attribution and extension of the horizon-scaled discount: gamma 0.97 on Peg; PPO at gamma 0.95 on Peg (is it the learner or
    # the discount?); TD3 at gamma 0.97 on the 200-step PlugCharger (registered prediction: still 0, no stage signal).
    's1':{
        'peg-td3-utd16-g97':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=15360000,env_id='PegInsertionSide-v1',
                                 transfer=True,gamma=0.97),
        'peg-ppo-g95':dict(num_envs=1024,obs_norm=False,total=30720000,env_id='PegInsertionSide-v1',transfer=True,num_steps=100,gamma=0.95),
        'plug-td3-utd16-g97':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=10240000,env_id='PlugCharger-v1',
                                  transfer=True,gamma=0.97),
    },
    'r5':{
        'ppo-control-anneal':dict(num_envs=1024,obs_norm=False,anneal_lr=True,env_id='PlaceSphere-v1',transfer=True,total=15360000),
        'ppo-env512-bineps-anneal':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3,anneal_lr=True,env_id='PlaceSphere-v1',transfer=True,total=15360000),
    },
    'r6':{
        'ppo-control':dict(num_envs=1024,obs_norm=False,total=15360000,env_id='PickCube-v1',transfer=True),
        'td3-utd16':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=16,total=7680000,env_id='PickCube-v1',transfer=True),
    },
    'o3':{
        'td3-utd32':dict(algo='td3',num_envs=1024,obs_norm=False,updates_per_step=32),
    },
    'r2':{
        'env512-bineps':dict(num_envs=512,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
    },
    # Fresh-seed replication of the promising arms with a fresh control (registered before outcomes).
    'r1':{
        'control':dict(num_envs=1024,obs_norm=False),
        'bineps':dict(num_envs=1024,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
        'env256-bineps':dict(num_envs=256,obs_norm=False,binary_gripper=True,binary_eps=0.02,binary_logit_l2=1e-3),
    },
}
ARMS=ARMS_BY_BATCH[BATCH.split('-',1)[1] if BATCH.startswith('smoke-') else 'b1' if BATCH.startswith('smoke') else BATCH]
if BATCH.startswith('smoke-') and 'AUTOMANIP_SE_TOTAL' in os.environ:
    ARMS={k:{**v,'total':TOTAL} for k,v in ARMS.items()}
SEEDS={'r1':(21,22,23,24,25),'r2':(21,22,23,24,25),'o1':(11,12),'r3':(21,22,23,24,25),'o2':(11,12),'r4':(21,22,23,24,25),'o3':(11,12),'r5':(21,22,23,24,25),'r6':(21,22,23,24,25),'h1':(11,12),'r7':(21,22,23,24,25)}.get(BATCH,(11,12,13))
TRIALS=[(arm,seed) for arm in ARMS for seed in SEEDS]
FILES=('ppo.py','export_helper.py','symmetry.py','td3.py')+tuple('demos/'+p.name for p in sorted((SOURCE/'demos').glob('*.json')))
PINS={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'}
SOURCE_SHA={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}

app=modal.App(NAME,include_source=False)
training_image=trainer_image.add_local_file(SOURCE/'ppo.py','/opt/se/ppo.py',copy=True).add_local_file(
    SOURCE/'export_helper.py','/opt/se/export_helper.py',copy=True).add_local_file(
    SOURCE/'symmetry.py','/opt/se/symmetry.py',copy=True).add_local_file(
    SOURCE/'td3.py','/opt/se/td3.py',copy=True)
for _f in FILES:
    if _f.startswith('demos/'):training_image=training_image.add_local_file(SOURCE/_f,'/opt/se/'+_f,copy=True)
training_outputs=modal.Volume.from_name(NAME,create_if_missing=True,version=2)
score_outputs=modal.Volume.from_name(NAME+'-scores',create_if_missing=True,version=2)
cache=results.with_mount_options(read_only=True,sub_path='/runtime/sapien-cache')


@app.function(image=training_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=TRAINER_SECONDS+300,
    startup_timeout=120,retries=0,max_containers=len(TRIALS),single_use_containers=True,scaledown_window=2,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/output':training_outputs,'/root/.sapien':cache})
def train_trial(index,expected_sha):
    import importlib.metadata
    import os
    import signal
    import subprocess
    import time
    import traceback

    if type(index) is not int or not 0<=index<len(TRIALS):raise ValueError('Closed trial index')
    arm,seed=TRIALS[index];config=ARMS[arm]
    root=Path('/output')/f'trial-{index}';root.mkdir(exist_ok=False)
    started=time.monotonic();record=dict(state='started',index=index,arm=arm,seed=seed,config=config,
        training_initialization='from_scratch',protected_queries=0)
    try:
        actual={f:hashlib.sha256((Path('/opt/se')/f).read_bytes()).hexdigest() for f in FILES}
        if actual!=expected_sha:raise ValueError('Training source differs from launch')
        if Path('/opt/automanip').exists() or Path('/results').exists():raise RuntimeError('Trainer contains scorer filesystem')
        record['versions']={name:importlib.metadata.version(name) for name in PINS}
        if record['versions']!=PINS:raise ValueError('Pinned versions differ')
        import torch
        record['gpu']=torch.cuda.get_device_name(0)
        if record['gpu']!='NVIDIA L40S':raise ValueError('Hardware differs')
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!='ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440':raise ValueError('Task source differs')
        work=root/'work';work.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith('MODAL_') and k!='RUNPOD_API_KEY'}
        env['AUTOMANIP_TRAINING_DEADLINE']=str(time.monotonic()+TRAINER_SECONDS-30)
        env['PYTHONPATH']='/opt/se'
        command=[sys.executable,'/opt/se/ppo.py','--env-id',config.get('env_id','StackCube-v1'),'--seed',str(seed),'--num-envs',str(config['num_envs']),
                 '--num-eval-envs','8','--num-minibatches','32','--update-epochs','8','--total-timesteps',str(config.get('total',TOTAL)),
                 '--exp-name',f'se-{BATCH}-{arm}-seed-{seed}','--gamma',str(config.get('gamma',0.9)),'--num-steps',str(config.get('num_steps',50)),
                 '--num-eval-steps',str(config.get('num_steps',50)),'--ent-coef',str(config.get('ent_coef',0.003)),'--learning-rate','0.0003',
                 '--checkpoint-every','1024000','--obs-norm' if config['obs_norm'] else '--no-obs-norm',
                 '--heavy-tail-dims',config.get('heavy_tail_dims','none'),'--t-df',str(config.get('t_df',2.0)),
                 '--critic-width',str(config.get('critic_width',256)),
                 '--critic-layernorm' if config.get('critic_layernorm') else '--no-critic-layernorm',
                 '--binary-gripper' if config.get('binary_gripper') else '--no-binary-gripper',
                 '--binary-eps',str(config.get('binary_eps',0.0)),'--binary-logit-l2',str(config.get('binary_logit_l2',0.0)),
                 '--gripper-std-floor',str(config.get('gripper_std_floor',0.0)),
                 '--binary-sticky',str(config.get('binary_sticky',0.0)),
                 '--sym-theta',str(config.get('sym_theta',0.0)),
                 '--init-logstd',str(config.get('init_logstd',-0.5)),
                 '--gripper-mean-barrier',str(config.get('gripper_mean_barrier',0.0)),
                 '--transfer-eval' if config.get('transfer') else '--no-transfer-eval',
                 '--anneal-lr' if config.get('anneal_lr') else '--no-anneal-lr',
                 '--no-partial-reset','--no-capture-video','--no-track','--no-save-model']
        if config.get('algo')=='td3':
            command=[sys.executable,'/opt/se/td3.py','--seed',str(seed),'--total-timesteps',str(config.get('total',TOTAL)),
                     '--env-id',config.get('env_id','StackCube-v1'),'--transfer-eval' if config.get('transfer') else '--no-transfer-eval',
                     '--exp-name',f'se-{BATCH}-{arm}-seed-{seed}','--checkpoint-every',str(CHECKPOINT),
                     '--burst-prob',str(config.get('burst_prob',0.0)),'--sym-theta',str(config.get('sym_theta',0.0)),
                     '--updates-per-step',str(config.get('updates_per_step',4)),'--per-alpha',str(config.get('per_alpha',0.0)),
                     '--fast' if config.get('fast') else '--no-fast','--bf16' if config.get('bf16') else '--no-bf16',
                     '--utd-late',str(config.get('utd_late',0)),'--gamma',str(config.get('gamma',0.9)),'--rc-source',config.get('rc_source','none'),'--rc-frac',str(config.get('rc_frac',0.25)),
                     '--rc-demos',('/opt/se/demos/'+config['rc_demos']) if config.get('rc_demos') else '']
        record['command']=command;tick=time.monotonic()
        with (root/'train.log').open('x') as log:
            process=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=TRAINER_SECONDS+60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('Hard trainer process limit')
        record['trainer_process_seconds']=time.monotonic()-tick
        if code:raise RuntimeError(f'Trainer exited {code}')
        record['state']='complete'
    except Exception as exc:
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        journal=list(root.glob('work/runs/*/work.jsonl'))
        if len(journal)==1:
            events=[]
            for line in journal[0].read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:record['partial_final_journal_line']=True
            record['actual_work']={kind:max((e['count'] for e in events if e['event']==kind),default=0)
                for kind in ['environment_completed','optimizer_completed']}
            record['checkpoints']=[e for e in events if e['event']=='actor_export']
            record['transfer_evals']=[e for e in events if e['event']=='transfer_eval']
            record['training_finished']=any(e['event']=='training_complete' for e in events)
        record['wrapper_seconds']=time.monotonic()-started
        (root/'receipt.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        subprocess.run(['sync',str(root)],check=False,timeout=60)
    return {k:v for k,v in record.items() if k not in ['traceback','command','checkpoints']}  # includes transfer_evals


@app.function(image=validation_image,gpu='L40S',cpu=(3,4),memory=(16384,16384),timeout=1500,
    startup_timeout=120,retries=0,max_containers=len(TRIALS),single_use_containers=True,scaledown_window=2,
    restrict_modal_access=True,block_network=True,include_source=False,serialized=True,
    _experimental_restrict_output=True,volumes={'/candidates':training_outputs.with_mount_options(read_only=True),'/scores':score_outputs,'/root/.sapien':cache})
def score_trial(index):
    from dataclasses import asdict
    import time
    sys.path.insert(0,'/opt/automanip/src')
    from automanip.evaluator import evaluate_batch
    from automanip.numpy_inference import NumpyPolicy
    from automanip.policy_io import load_policy,PolicySchema
    from automanip.stackcube import STACK_TASK,check_frozen_protocol
    from automanip.stackcube_adapter import StackCubeAdapter

    if type(index) is not int or not 0<=index<len(TRIALS):raise ValueError('Closed trial index')
    arm,seed=TRIALS[index]
    root=Path('/scores')/f'trial-{index}';root.mkdir(exist_ok=True)
    if (root/'report.json').exists() and json.loads((root/'report.json').read_text())['state']=='complete':
        raise FileExistsError('Trial already scored; no repeat scoring')
    receipt=json.loads((Path('/candidates')/f'trial-{index}/receipt.json').read_text())
    if receipt['arm']!=arm or receipt['seed']!=seed:raise ValueError('Trial identity mismatch')
    record=dict(state='started',index=index,arm=arm,seed=seed,points=[],partition='public-development',
                public_seed_base=PUBLIC_SEED_BASE,confirmation_queries=0,protocol_sha256=check_frozen_protocol(),
                training_state=receipt['state'])
    started=time.monotonic();env=None
    def persist():
        record['wall_seconds']=time.monotonic()-started
        temporary=root/'report.tmp';temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(root/'report.json')
    try:
        env=StackCubeAdapter(128,'physx_cuda');schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        completed=receipt.get('actual_work',{}).get('environment_completed',0)
        for checkpoint in receipt.get('checkpoints',[]):
            steps=checkpoint['steps']
            if type(steps) is not int or steps<0 or steps>completed:raise ValueError('Uncompleted checkpoint')
            actor=load_policy(Path('/candidates')/f'trial-{index}/work/runs/se-{BATCH}-{arm}-seed-{seed}/actors'/str(steps),schema)
            if actor.weights_sha256!=checkpoint['actor']['weights_sha256']:raise ValueError('Candidate checkpoint changed')
            rollouts=evaluate_batch(env,NumpyPolicy(actor,schema),list(range(PUBLIC_SEED_BASE,PUBLIC_SEED_BASE+128)),horizon=50,action_dim=8)
            point=dict(steps=steps,trainer_seconds=checkpoint['trainer_seconds'],weights_sha256=actor.weights_sha256,
                       episodes=128,successes=sum(r.outcome.success_at_end for r in rollouts),
                       failures=sum(r.outcome.failed for r in rollouts),outcomes=[asdict(r) for r in rollouts])
            reports=[d.report(r.outcome) for d,r in zip(env.diagnostics,rollouts)]  # observational stages only
            point['stages_ever']={k:sum(x['ever'][k] for x in reports) for k in reports[0]['ever']}
            point['first_unmet_stage']={k:sum(x['first_unmet_stage']==k for x in reports) for k in {x['first_unmet_stage'] for x in reports}}
            record['points'].append(point);persist()
            if point['failures']:raise ValueError('Scorer execution failure')
        record['state']='complete'
    except Exception as exc:record.update(state='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if env is not None:env.close()
        persist()
    return dict(index=index,arm=arm,seed=seed,state=record['state'],
                curve=[(p['steps'],p['successes'],p['stages_ever']) for p in record['points']],error=record.get('error'))


@app.function(image=modal.Image.debian_slim(python_version='3.11').pip_install('modal==1.6.2'),cpu=1,memory=1024,
    timeout=TRAINER_SECONDS+3600,retries=0,max_containers=1,include_source=False,serialized=True,
    volumes={'/scores':score_outputs})
def coordinate(indices):
    """Remote coordinator: train the given trials, score each as it finishes, journal to the score volume."""
    import time
    log=Path('/scores')/'coordinator.jsonl'
    def note(**kw):
        kw['t']=time.time()
        with log.open('a') as f:f.write(json.dumps(kw)+'\n')
        score_outputs.commit()
    note(event='submit',indices=indices,trials=[TRIALS[i] for i in indices],source_sha256=SOURCE_SHA,total=TOTAL)
    calls={i:train_trial.spawn(i,SOURCE_SHA) for i in indices}
    for i,c in calls.items():note(event='spawned',index=i,call=c.object_id)
    scores={};pending=dict(calls)
    while pending:
        for i,c in list(pending.items()):
            try:result=c.get(timeout=5)
            except (TimeoutError,modal.exception.TimeoutError):continue
            except Exception as exc:result=dict(state='failed',error=repr(exc))
            note(event='trained',index=i,result=result);del pending[i]
            scores[i]=score_trial.spawn(i)
    for i,c in scores.items():
        try:result=c.get()
        except Exception as exc:result=dict(state='failed',error=repr(exc))
        note(event='scored',index=i,result=result)
    note(event='done')


@app.local_entrypoint()
def run(only:str=''):
    """Submit the remote coordinator for all trials (or a comma-separated subset of indices) and return."""
    import time
    RUNTIME.mkdir(parents=True,exist_ok=True)
    indices=[int(i) for i in only.split(',')] if only else list(range(len(TRIALS)))
    call=coordinate.spawn(indices)
    record=dict(batch=BATCH,volume=NAME,indices=indices,trials=[TRIALS[i] for i in indices],coordinator=call.object_id,
                source_sha256=SOURCE_SHA,total=TOTAL,submitted_at=time.time())
    with (RUNTIME/'launch.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record))


@app.local_entrypoint()
def score_only(only:str):
    """Score already-finished trials (comma-separated indices) without retraining."""
    indices=[int(i) for i in only.split(',')]
    for result in score_trial.map(indices):print(json.dumps(result))
