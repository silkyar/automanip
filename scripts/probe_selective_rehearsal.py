"""Four fixed CPU solves, isolating stage choice from amount/time of deletion."""
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.prefix_search import raw_mean
from research.readout_distill import features


def solve_gripper(t, residual, r, keep):
    if keep.shape != (len(r),) or keep.dtype != np.bool_:
        raise ValueError('Boolean row mask required')
    gram=t.T@t/len(t)+4*(r.T@(r*keep[:,None]))/len(r)+.001*np.eye(t.shape[1])
    target=t.T@residual/len(t)
    delta=np.linalg.solve(gram,target)
    if np.max(np.abs(gram@delta-target))>1e-9:
        raise ValueError('Masked normal equation failed')
    return delta


def matched_random(keep, times, seed):
    generator=np.random.default_rng(seed)
    control=np.ones_like(keep)
    for step in np.unique(times):
        rows=np.flatnonzero(times==step)
        remove=int((~keep[rows]).sum())
        control[generator.choice(rows,remove,replace=False)]=False
    return control


def probe():
    started=time.process_time()
    parents=[]
    for index,parent in enumerate((2,3)):
        with np.load(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz',allow_pickle=False) as z:audit={k:z[k] for k in z.files}
        with np.load(f'artifacts/cumulative/prefix-search-20261030/remote/collection-{index}/candidate-0-r0/trace.npz',allow_pickle=False) as z:
            if not np.array_equal(z['observations'][:30].transpose(1,0,2),data['retain_observations']):
                raise ValueError('Native data alignment differs')
            flags=np.zeros((64,30,7),dtype=np.int8)
            flags[:,1:]=z['flags'][:29].transpose(1,0,2)
        ti,ri,ep=audit['teacher_row_indices'],audit['rehearsal_row_indices'],audit['fitting_episode_indices']
        obs=data['teacher_observations'][ti];target=data['teacher_actions'][ti]
        retain=data['retain_observations'][ep].reshape(-1,48)[ri]
        retained_flags=flags[ep].reshape(-1,7)[ri]
        times=np.tile(np.arange(30),len(ep))[ri]
        keep=~(retained_flags[:,0].astype(bool)&retained_flags[:,3].astype(bool))
        control=matched_random(keep,times,97010000+parent)
        validation=data['teacher_observations'][np.isin(data['teacher_episode'],audit['validation_episode_indices'])]
        vy=data['teacher_actions'][np.isin(data['teacher_episode'],audit['validation_episode_indices'])]
        vr=data['retain_observations'][audit['validation_episode_indices']].reshape(-1,48)
        vf=flags[audit['validation_episode_indices']].reshape(-1,7)
        # Exclude reset frames with unknown stages from conditional diagnostic.
        non_on_top=(~vf[:,3].astype(bool)) & (np.tile(np.arange(30),len(audit['validation_episode_indices']))>0)
        schema=PolicySchema(STACK_TASK.fingerprint,48,8)
        base=NumpyPolicy(load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}',schema),schema)
        t,r,v,m=[features(base,x) for x in (obs,retain,validation,vr)]
        old_t,old_v,old_m=[raw_mean(base,x) for x in (obs,validation,vr)]
        methods={}
        for name,mask in [('original',None),('stage_selective',keep),('time_matched_random',control)]:
            delta=audit['delta'].copy()
            if mask is not None:delta[:,7]=solve_gripper(t,target[:,7]-old_t[:,7],r,mask)
            pred=old_v+v@delta;memory=old_m+m@delta
            change=np.clip(memory,-1,1)-np.clip(old_m,-1,1)
            methods[name]=dict(finger_target_RMSE_metres=float(.025*np.sqrt(np.square(np.clip(pred[:,7],-1,1)-vy[:,7]).mean())),
                non_on_top_memory_executed_RMS_change=float(np.sqrt(np.square(change[non_on_top]).mean())),
                all_memory_executed_RMS_change=float(np.sqrt(np.square(change).mean())),
                arm_delta_max_change=float(np.abs(delta[:,:7]-audit['delta'][:,:7]).max()))
        old,new,random=[methods[k] for k in ('original','stage_selective','time_matched_random')]
        passed=(new['finger_target_RMSE_metres']<.5*old['finger_target_RMSE_metres'] and
                new['finger_target_RMSE_metres']<.75*random['finger_target_RMSE_metres'] and
                new['non_on_top_memory_executed_RMS_change']<=old['non_on_top_memory_executed_RMS_change'])
        parents.append(dict(parent=parent,methods=methods,passed=passed,removed_gripper_rows=int((~keep).sum()),
            per_time_counts_match=bool(all((~keep[times==step]).sum()==(~control[times==step]).sum() for step in range(30)))))
    return dict(state='complete',passed=all(p['passed'] for p in parents),parents=parents,
        linear_solves=4,CPU_seconds=time.process_time()-started,new_environment_transitions=0,
        optimizer_calls=0,policy_exports=0,robot_assessments=0,
        scope='Fixed assisted-data interference probe; heldoutcorpusadaptivelyreused, nolearningorpolicyperformanceclaim')


if __name__=='__main__':
    result=probe()
    with Path('cumulative/results/selective-rehearsal-CPU-20261031.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
