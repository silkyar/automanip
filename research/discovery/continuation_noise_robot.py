"""Fixed public nested rollouts; no learning, scorer, state restoration or search."""
import argparse,hashlib,importlib.metadata,json,os,time
from pathlib import Path

import numpy as np
import torch
from sac_baseline import Actor
from continuation_variance import components,cluster_interval

PINS={'mani_skill':'3.0.0b21','sapien':'3.0.0b1','torch':'2.6.0+cu124','numpy':'1.26.4','gymnasium':'0.29.1'}
TASK_SHA='ff76ae608f79ba3ea13334e61ccb72f291660f9461895c70e3c30f8390694440'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load_actor(path,expected):
    raw=path.read_bytes()
    if len(raw)>8*1024*1024 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Policy source differs')
    payload=json.loads(raw);actor=Actor().cuda().eval();state=actor.state_dict()
    if set(state)!=set(payload['state']):raise ValueError('Actor fields differ')
    for name,tensor in state.items():
        value=payload['state'][name]
        if value['shape']!=list(tensor.shape):raise ValueError('Actor shape differs')
        array=np.asarray(value['values'],dtype=np.float32).reshape(tensor.shape)
        if not np.isfinite(array).all():raise ValueError('Actor contains nonfinite values')
        state[name]=torch.from_numpy(array.copy()).cuda()
    actor.load_state_dict(state);actor.requires_grad_(False)
    return actor


def run(inputs,output):
    started=time.monotonic();deadline=min(started+480,float(os.environ['AUTOMANIP_DIAGNOSTIC_DEADLINE']))
    output.mkdir(exist_ok=False);plan=json.loads((inputs/'plan.json').read_bytes())
    report=dict(state='started',kind='Public first-decision continuation-noise probe, not training or benchmark',
        started_epoch=time.time(),actual_public_transitions=0,attempted_public_transitions=0,vector_step_calls=0,
        explicit_vector_reset_calls=0,optimizer_calls=0,new_training_runs=0,protected_evaluations=0,
        policies=plan['policies'],public_initial_seeds=plan['public_initial_seeds'],results=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={k:importlib.metadata.version(k) for k in PINS})
    env=None
    def bounded():
        if time.monotonic()>deadline:raise TimeoutError('Original diagnostic lease expired; no retry')
    try:
        if not plan['admitted']:raise PermissionError('Diagnostic not root-admitted')
        for name,expected in plan['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:raise ValueError('Reviewed source differs')
        if report['versions']!=PINS or torch.cuda.get_device_name(0)!='NVIDIA L40S':raise RuntimeError('Pins/GPU differ')
        if plan['public_initial_seeds']!=list(range(93274000,93274004)) or plan['max_public_transitions']!=409600:raise ValueError('Fixed diagnostic differs')
        import gymnasium as gym
        import mani_skill.envs
        from mani_skill.envs.tasks.tabletop import stack_cube
        if hashlib.sha256(Path(stack_cube.__file__).read_bytes()).hexdigest()!=TASK_SHA:raise RuntimeError('Task differs')
        cache=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        if hashlib.sha256(cache.read_bytes()).hexdigest()!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':raise RuntimeError('Physics cache differs')
        env=gym.make('StackCube-v1',num_envs=1024,robot_uids='panda_wristcam',obs_mode='state',
            reward_mode='normalized_dense',control_mode='pd_joint_delta_pos',sim_backend='physx_cuda',render_mode='rgb_array')
        if env.unwrapped.single_observation_space.shape!=(48,) or env.unwrapped.single_action_space.shape!=(8,):raise ValueError('Interface differs')
        for policy_index,policy in enumerate(plan['policies']):
            actor=load_actor(inputs/policy['filename'],policy['sha256'])
            for state_index,seed in enumerate(plan['public_initial_seeds']):
                bounded();obs,_=env.reset(seed=[seed]*1024);report['explicit_vector_reset_calls']+=1
                initial=obs[:1].clone();difference=float((obs-initial).abs().max())
                report['current_batch']=dict(policy=policy['name'],seed=seed,initial_observation_max_difference=difference)
                if difference>1e-6:raise RuntimeError('Replicated initial observations differ; no branching claim')
                generator=torch.Generator(device='cuda').manual_seed(93275000+100*policy_index+state_index)
                noise_rng=np.random.default_rng(93276000+100*policy_index+state_index)
                first_z=noise_rng.normal(size=(64,8)).astype(np.float32)
                first_repeated=torch.from_numpy(np.repeat(first_z,16,axis=0)).cuda()
                rewards=[];future_noises=[];first_action=None;successor_difference=None
                ever_grasp=torch.zeros(1024,dtype=torch.bool,device='cuda')
                with torch.no_grad():
                    mean0,logstd0=actor(initial)
                    for step in range(50):
                        bounded();mean,logstd=actor(obs)
                        noise=first_repeated if step==0 else torch.randn((1024,8),device='cuda',generator=generator)
                        if step>0:future_noises.append(noise.reshape(64,16,8).cpu().numpy())
                        action=torch.tanh(mean+logstd.exp()*noise)
                        if not torch.isfinite(action).all():raise ValueError('Nonfinite public action')
                        if step==0:first_action=action.reshape(64,16,8)[:,0].cpu().numpy()
                        if report['attempted_public_transitions']+1024>409600:raise RuntimeError('Next step would exceed query cap')
                        report['attempted_public_transitions']+=1024
                        obs,reward,term,trunc,info=env.step(action)
                        report['actual_public_transitions']+=1024;report['vector_step_calls']+=1
                        if step==0:
                            grouped=obs.reshape(64,16,48)
                            successor_difference=float((grouped-grouped[:,:1]).abs().max())
                            report['current_batch']['repeated_first_action_successor_max_difference']=successor_difference
                            if successor_difference>1e-5:raise RuntimeError('Repeated first-action successors differ; no nested grouping claim')
                        if report['actual_public_transitions']>409600:raise RuntimeError('Transition cap exceeded')
                        if not torch.isfinite(reward).all() or float(reward.max())>1.00001 or float(reward.min())<-.00001:raise ValueError('Reward convention differs')
                        if bool(trunc.any()) and step<49:raise RuntimeError('Unexpected early cutoff')
                        rewards.append(reward.reshape(64,16).cpu().numpy())
                        ever_grasp|=info['is_cubeA_grasped'].bool()
                    if not bool(trunc.all()):raise RuntimeError('Expected full50-step horizon')
                raw_rewards=np.stack(rewards)
                name=policy['name']+'-seed-'+str(seed)
                arrays=dict(rewards=raw_rewards,first_standardized_noise=first_z,
                    future_standardized_noise=np.stack(future_noises),first_action=first_action,
                    initial_observation=initial.cpu().numpy(),initial_mean=mean0.cpu().numpy(),
                    initial_logstd=logstd0.cpu().numpy(),terminal_success=info['success'].cpu().numpy(),
                    ever_grasp=ever_grasp.cpu().numpy())
                np.savez_compressed(output/(name+'.npz'),**arrays)
                entry=dict(policy=policy['name'],seed=seed,initial_observation_max_difference=difference,
                    repeated_first_action_successor_max_difference=successor_difference,
                    public_episodes=1024,ever_grasp_count=int(ever_grasp.sum()),terminal_successes=int(info['success'].sum()),
                    data=name+'.npz',objectives={})
                for gamma in (.99,.9):
                    returns=(raw_rewards.astype(np.float64)*gamma**np.arange(50)[:,None,None]).sum(0)
                    value=components(returns,first_z)
                    value['interval']=cluster_interval(returns,first_z)
                    entry['objectives'][str(gamma)]=value
                report['results'].append(entry);report['elapsed_seconds']=time.monotonic()-started
                report.pop('current_batch')
                write(output/'report.json',report)
        report['state']='complete'
    except Exception as exc:
        report.update(state='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if env is not None:env.close()
        report['elapsed_seconds']=time.monotonic()-started
        write(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('state','actual_public_transitions','elapsed_seconds')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs,args.output)
