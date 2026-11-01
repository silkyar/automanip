"""Bounded new-track mechanics and source-only development assessment.

Never trains or uses final seeds. Invoked only by the admitted Modal wrapper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def run(output):
    started=time.monotonic();output=Path(output);output.mkdir(exist_ok=False)
    record=dict(state='started',training_updates=0,final_episodes=0,
                mechanical_transitions=0,development_interactions_reserved=0,development_transitions=0)
    def persist():
        record['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    persist()
    try:
        from importlib.metadata import version
        import numpy as np
        import torch
        from automanip.cumulative import ROOT,SOURCE,NORTH
        from automanip.cumulative_env import make_training_environment,VariationAdapter,assess_controller
        from automanip.policy_io import load_policy
        from research.discovery.amp_v1 import PRODUCTION_PINS
        record['versions']={name:version(name) for name in PRODUCTION_PINS}
        if record['versions']!=PRODUCTION_PINS:raise ValueError('Production dependency pins differ')
        if not torch.cuda.is_available():raise ValueError('CUDA required for this registered preflight')
        record['gpu']=torch.cuda.get_device_name(0)
        if 'L40S' not in record['gpu']:raise ValueError('Registered hardware differs')
        physx=Path('/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so')
        record['physx_sha256']=hashlib.sha256(physx.read_bytes()).hexdigest()
        if record['physx_sha256']!='4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706':raise ValueError('PhysX pin differs')
        policy=ROOT/'runs/stackcube-bc-exploration-pilot-20261028/teacher-input'
        asset=load_policy(policy,SOURCE.schema)
        if asset.weights_sha256!='ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16':raise ValueError('Source differs')
        env=make_training_environment(NORTH,num_envs=8)
        try:
            obs,_=env.reset(seed=93439000);owner=env.base.unwrapped
            if owner.control_freq!=20 or owner.sim_freq!=100:raise ValueError('Effective simulator clocks differ')
            record['clocks']=dict(control_hz=owner.control_freq,simulation_hz=owner.sim_freq)
            a,b=owner.cubeA.pose.p.clone(),owner.cubeB.pose.p.clone()
            if obs.shape!=(8,48) or not torch.isfinite(obs).all():raise ValueError('Invalid transformed observations')
            if not torch.allclose(obs[:,25:28],a,atol=1e-6) or not torch.allclose(obs[:,32:35],b,atol=1e-6):raise ValueError('GPU pose/observation mismatch')
            if not bool(((b[:,1]>=.32-1e-6)&(b[:,1]<=.36+1e-6)).all()):raise ValueError('Goal support differs')
            if not bool((torch.linalg.vector_norm(a[:,:2]-b[:,:2],dim=1)>.15).all()):raise ValueError('Initial cube separation differs')
            record['initial_cube_a']=a.cpu().tolist();record['initial_cube_b']=b.cpu().tolist()
            record['initial_qpos']=owner.agent.robot.get_qpos()[:,:7].cpu().tolist()
            record['initial_tcp']=owner.agent.tcp.pose.p.cpu().tolist()
            # Analytic URDF kinematics is a feasibility diagnostic only. No
            # actions, labels, optimized joints or solver outputs enter PPO.
            from research.discovery.panda_kinematics import Kinematics,parse_chain
            from scipy.optimize import least_squares
            kin=Kinematics(parse_chain(owner.agent.urdf_path))
            expected=kin.fk(np.asarray(record['initial_qpos']))[:,:3,3]
            if np.max(np.abs(expected-np.asarray(record['initial_tcp'])))>1e-4:raise ValueError('Analytic FK frame differs')
            bounds=np.asarray([j['limits'] for j in kin.chain if j['type']=='revolute']).T
            residuals=[]
            for q,target in zip(record['initial_qpos'],record['initial_cube_b']):
                target=np.asarray(target)+np.array([0.,0.,.08])
                def residual(joints):return kin.fk(joints)[0,:3,3]-target
                solution=least_squares(residual,np.asarray(q),bounds=bounds,max_nfev=100)
                residuals.append(float(np.linalg.norm(residual(solution.x))))
            record['position_only_ik_error_m']=residuals
            record['analytic_fk_evaluations']=kin.fk_pose_evaluations
            if max(residuals)>.01:raise ValueError('Goal fails basic position reachability gate')
            for _ in range(50):
                record['mechanical_transitions']+=8;persist()
                obs,reward,term,trunc,info=env.step(torch.zeros((8,8),device=owner.device))
                if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():raise ValueError('Nonfinite physical step')
            if not bool((owner.cubeB.pose.p[:,2]>.005).all()):raise ValueError('Destination left table')
            record['mechanics_passed']=True
            record['reachability_scope']='Position-only analytic IK plus valid stationary-object physics; not collision-free grasp/stack proof'
        finally:env.close()
        persist()
        # These are the registered unchanged-source/actor-warmstart step-zero
        # development endpoints, not an additional feedback allocation.
        zero_started=time.monotonic();reports=[]
        for task in (SOURCE,NORTH):
            record['development_interactions_reserved']+=128*50;persist()
            report=assess_controller(policy,task,list(range(93450000,93450128)),
                partition='development',environment=VariationAdapter(task,8))
            reports.append(report)
            record['development_transitions']+=report['environment_transitions'];persist()
            (output/(task.goal+'-development.json')).write_text(json.dumps(report,indent=2)+'\n')
            if report['failed_episodes']:raise ValueError('Development assessment execution failed; not a policy score')
        record['source_development']=[dict(task_id=r['task_id'],successes=r['successes'],episodes=r['episodes'],failed_episodes=r['failed_episodes']) for r in reports]
        record['zero_shot_wall_seconds']=time.monotonic()-zero_started
        record['zero_shot_joint_gate']=all(r['successes']>=116 and not r['failed_episodes'] for r in reports)
        record['state']='complete'
    except Exception as exc:
        import traceback
        record.update(state='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:persist()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    record=run(parser.parse_args().output)
    print(json.dumps({k:v for k,v in record.items() if k not in ('traceback','initial_qpos','initial_tcp','initial_cube_a','initial_cube_b')}))
    raise SystemExit(0 if record['state']=='complete' else 1)
