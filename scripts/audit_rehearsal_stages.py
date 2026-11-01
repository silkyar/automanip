"""Exploratory audit of saved pre-action stages; no fitting or policy selection."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def summary(observations, commands, flags):
    target_gap = .015+.025*np.clip(commands[:, 7], -1, 1)-observations[:, 7]
    grasp, placed, static = flags[:, 0].astype(bool), flags[:, 3].astype(bool), flags[:, 4].astype(bool)
    subsets = dict(all=np.ones(len(flags), dtype=bool), grasped_not_on_top=grasp & ~placed,
                   on_top_grasped=grasp & placed, on_top_grasped_static=grasp & placed & static)
    return {name:dict(rows=int(mask.sum()),fraction=float(mask.mean()),
        closing_fraction=float((target_gap[mask] < -1e-6).mean()) if mask.any() else None,
        opening_fraction=float((target_gap[mask] > 1e-6).mean()) if mask.any() else None,
        mean_target_gap_metres=float(target_gap[mask].mean()) if mask.any() else None)
        for name, mask in subsets.items()}


def audit():
    started=time.process_time()
    parents=[]
    for index,parent in enumerate((2,3)):
        path=Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        ap=Path(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz')
        with np.load(ap,allow_pickle=False) as z:fit={k:z[k] for k in z.files}
        bp=Path(f'artifacts/cumulative/prefix-search-20261030/remote/collection-{index}/candidate-0-r0/trace.npz')
        with np.load(bp,allow_pickle=False) as z:base={k:z[k] for k in ('observations','flags')}
        if not np.array_equal(data['retain_observations'],base['observations'][:30].transpose(1,0,2)):
            raise ValueError('Rehearsal source alignment differs')
        # Flag[t] was measured AFTER action[t]. Rehearsal observations are
        # BEFORE actions, so shift by one and explicitly exclude reset frames.
        flags=np.zeros((64,30,7),dtype=np.int8)
        flags[:,1:]=base['flags'][:29].transpose(1,0,2)
        episodes=fit['fitting_episode_indices']
        row_indices=fit['rehearsal_row_indices']
        r_obs=data['retain_observations'][episodes].reshape(-1,48)[row_indices]
        r_commands=data['retain_parent_means'][episodes].reshape(-1,8)[row_indices]
        r_flags=flags[episodes].reshape(-1,7)[row_indices]
        times=np.tile(np.arange(30),len(episodes))[row_indices]
        observed=times>0
        teacher_flags=[]
        teacher_obs=[]
        source_hashes={str(bp):hashlib.sha256(bp.read_bytes()).hexdigest()}
        for lane,candidate in enumerate(data['selected_candidate']):
            if candidate<0:continue
            cp=Path(f'artifacts/cumulative/relative-gripper-20261031/remote/collection-{index}/current-candidate-{candidate}-r0/trace.npz')
            with np.load(cp,allow_pickle=False) as z:
                teacher_flags.extend(z['flags'][29:49,lane])
                teacher_obs.extend(z['observations'][30:50,lane])
            source_hashes[str(cp)]=hashlib.sha256(cp.read_bytes()).hexdigest()
        if not np.array_equal(np.asarray(teacher_obs),data['teacher_observations']):
            raise ValueError('Teacher pre-action alignment differs')
        ti=fit['teacher_row_indices']
        teacher_summary=summary(data['teacher_observations'][ti],data['teacher_actions'][ti],np.asarray(teacher_flags)[ti])
        by_time={f'{lo}-{hi-1}':summary(r_obs[mask],r_commands[mask],r_flags[mask])
                 for lo,hi in [(1,10),(10,20),(20,30)]
                 for mask in [observed & (times>=lo) & (times<hi)]}
        parents.append(dict(parent=parent,reset_rows_excluded=int((~observed).sum()),
            rehearsal=summary(r_obs[observed],r_commands[observed],r_flags[observed]),
            rehearsal_by_time=by_time,teacher_fit=teacher_summary,
            source_sha256=source_hashes,data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            audit_sha256=hashlib.sha256(ap.read_bytes()).hexdigest()))
    return dict(state='complete',analysis='exploratory_descriptive',parents=parents,
        CPU_seconds=time.process_time()-started,new_environment_transitions=0,linear_solves=0,
        optimizer_calls=0,policy_exports=0,robot_assessments=0,
        limitation='Saved acquired-policy stages. Shared coarse stage is not identical physical state, does not establish optimal action or causal interference. No memory rows removed, objective refit, thresholds selected or policies scored.')


if __name__=='__main__':
    result=audit()
    with Path('cumulative/results/rehearsal-stage-audit-20261031.json').open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
    print(json.dumps([{k:v for k,v in p.items() if k not in ('source_sha256','data_sha256','audit_sha256')} for p in result['parents']],indent=2))
