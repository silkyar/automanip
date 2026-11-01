"""One unregularized-rehearsal endpoint, for the registered causal probe only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.readout_distill import export_readout, features


def fit(parent, output):
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    path = Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
    with np.load(path, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    with np.load(f'.runtime/readout-students-20261031/search-{parent}/fit-audit.npz', allow_pickle=False) as z: indices = z['teacher_row_indices'].copy()
    schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
    artifact = load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}', schema)
    base = NumpyPolicy(artifact, schema); x = features(base, data['teacher_observations'][indices])
    w, b = base.layers[-1]; original = np.column_stack([w, b]).T
    gram = x.T@x/len(x)+.001*np.eye(x.shape[1]); target = x.T@(data['teacher_actions'][indices]-x@original)/len(x)
    delta = np.linalg.solve(gram, target)
    manifest = export_readout(artifact, schema, delta, output/'policy')
    load_policy(output/'policy', schema)
    report = dict(state='complete', parent=parent, method='teacher_only_endpoint', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), teacher_rows=len(x), new_physics=0,
                  linear_solves=1, ridge=.001, rehearsal_weight=0., optimizer_steps=0,
                  normal_equation_max_residual=float(np.abs(gram@delta-target).max()),
                  weights_sha256=manifest['weights_sha256'], CPU_seconds=time.monotonic()-started,
                  scope='Causal endpoint probe, not selected as an optimized learning method')
    (output/'fit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'fit-audit.npz', delta=delta, teacher_row_indices=indices)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--parent',type=int,choices=[2,3],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(fit(a.parent,a.output),indent=2))
