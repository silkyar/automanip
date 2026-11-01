"""One fixed old/new data mixture; source policy is the only bank difference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema,load_policy
from automanip.stackcube import STACK_TASK
from research.readout_distill import export_readout,features,fit_delta


def fit(parent,kind,output):
    if parent not in (2,3) or kind not in ('original','student'):raise ValueError('Fixed source kind')
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    old_path=Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
    new_path=Path(f'artifacts/cumulative/onpolicy-support-20261031/student-inputs/{kind}-{parent}.npz')
    with np.load(old_path,allow_pickle=False) as z:old={k:z[k] for k in z.files}
    with np.load(new_path,allow_pickle=False) as z:new={k:z[k] for k in z.files}
    with np.load(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz',allow_pickle=False) as z:audit={k:z[k] for k in z.files}
    old_indices=audit['teacher_row_indices'][:192]
    available=np.flatnonzero(np.isin(new['teacher_episode'],new['fitting_episode_indices']))
    if len(available)<192:raise ValueError('Insufficient preregistered new labels')
    chosen=np.random.default_rng(96750000+parent).choice(available,192,replace=False)
    observations=np.vstack([old['teacher_observations'][old_indices],new['teacher_observations'][chosen]])
    targets=np.vstack([old['teacher_actions'][old_indices],new['teacher_actions'][chosen]])
    retain=old['retain_observations'][audit['fitting_episode_indices']].reshape(-1,48)[audit['rehearsal_row_indices']]
    schema=PolicySchema(STACK_TASK.fingerprint,48,8);artifact=load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}',schema)
    base=NumpyPolicy(artifact,schema);w,b=base.layers[-1];W=np.column_stack([w,b]).T
    t,r=features(base,observations),features(base,retain)
    delta,report=fit_delta(t,targets,t@W,r)
    manifest=export_readout(artifact,schema,delta,output/'policy');updated=NumpyPolicy(load_policy(output/'policy',schema),schema)
    expected=np.clip(t[:64]@(W+delta),-1,1);parity=float(np.abs(np.asarray(updated.act(observations[:64]))-expected).max())
    if parity>5e-5:raise ValueError('Export parity failed')
    report.update(state='complete',parent=parent,source_kind=kind,old_teacher_rows=192,new_teacher_rows=192,
                  retain_rows=1024,new_physics=0,linear_solves=1,optimizer_steps=0,
                  old_data_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),new_data_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weights_sha256=manifest['weights_sha256'],
                  export_parity_max_error=parity,CPU_seconds=time.monotonic()-started)
    np.savez_compressed(output/'fit-audit.npz',delta=delta,old_teacher_row_indices=old_indices,new_teacher_row_indices=chosen)
    (output/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True);p.add_argument('--source-kind',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(fit(a.parent,a.source_kind,a.output),indent=2))
