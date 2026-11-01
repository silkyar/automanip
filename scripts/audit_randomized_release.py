"""Reconstruct every recorded diagnostic action on CPU; never runs physics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy
from automanip.stackcube import STACK_TASK
from research.randomized_release_policy import RandomizedReleasePolicy,BRANCHES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();started=time.monotonic()
    plan=json.loads((a.archive/'provenance/inputs/plan.json').read_text());schema=PolicySchema(STACK_TASK.fingerprint,48,8)
    result=dict(state='started',physics_calls=0,policy_forward_rows=0,max_executed_action_error=0.,cells=[])
    for index,seed in enumerate([2,3]):
        policy_data=load_policy(a.archive/'provenance/inputs'/plan['policies'][str(seed)],schema)
        report=json.loads((a.archive/f'remote/collection-{index}/run/report.json').read_text())
        by_branch={r['branch']:r for r in report['records']};counts={b:0 for b in BRANCHES};stages={b:{} for b in BRANCHES}
        for batch in range(32):
            with np.load(a.archive/f'remote/collection-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                assignment=np.random.default_rng(plan['assignment_seed_bases'][index]+batch).permutation(np.tile(np.arange(4),16))
                np.testing.assert_array_equal(z['assignment'],assignment)
                policy=RandomizedReleasePolicy(NumpyPolicy(policy_data,schema),plan['gaussian_std'][str(seed)],assignment)
                policy.reset_noise(plan['noise_seed_bases'][index]+batch)
                for step in range(50):
                    predicted=np.asarray(policy.act(z['observations'][step]),dtype=np.float32)
                    error=float(np.abs(predicted-z['executed_actions'][step]).max())
                    result['max_executed_action_error']=max(result['max_executed_action_error'],error);result['policy_forward_rows']+=64
                    if error>1e-6:raise ValueError('Executed diagnostic action reconstruction differs')
                for name,actual in [('trigger',policy.trigger),('applied',policy.applied),('first_eligible',policy.first_eligible),('fallback',policy.fallback),('override_masks',policy.override_masks)]:np.testing.assert_array_equal(z[name],actual)
                np.testing.assert_array_equal(z['applied'],np.where(assignment==0,0,5))
                flags=z['stage_flags'].astype(bool)
                for group,branch in enumerate(BRANCHES):
                    lanes=np.flatnonzero(assignment==group)
                    n=int(flags[-1,lanes,6].sum());counts[branch]+=n
                    if by_branch[branch]['batch_success_rates'][batch]!=n/16:raise ValueError('Reported success differs from physical trace')
                    for j,name in enumerate(['grasp','lift','transport','on_top','static','contact_free','success']):stages[branch][name]=stages[branch].get(name,0)+int(flags[:,lanes,j].any(0).sum())
                    stages[branch]['last10_success']=stages[branch].get('last10_success',0)+int(flags[-10:,lanes,6].all(0).sum())
        if any(counts[b]!=by_branch[b]['successes'] for b in BRANCHES):raise ValueError('Aggregate score differs')
        result['cells'].append(dict(seed=seed,weights_sha256=policy_data.weights_sha256,successes=counts,ever_stages=stages))
    result.update(state='verified',elapsed_seconds=time.monotonic()-started,scope='CPU reconstruction of declared randomization, controller and raw-action execution; not another physical evaluation',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (a.archive/'action-audit-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
