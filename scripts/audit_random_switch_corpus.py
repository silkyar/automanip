"""Reconstruct actual behavior and describe separately recorded cube motion."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy
from research.random_switch_collection import RandomSwitchPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();root=a.archive;started=time.monotonic()
    plan=json.loads((root/'provenance/inputs/plan.json').read_text());rows=[];forward_rows=0;max_error=0.
    for index,source in enumerate(plan['sources']):
        schema=PolicySchema(plan['task_fingerprint'],48,8);actor=NumpyPolicy(load_policy(root/'provenance/inputs'/source['policy'],schema),schema)
        stats=dict(on_top_post_action_frames=0,on_top_nonstatic=0,linear_only_failure=0,angular_only_failure=0,both_velocity_failures=0,instantaneous_angular_failure_with_small_control_interval_rotation=0,static_info_velocity_mismatches=0)
        for batch in range(32):
            rng=np.random.default_rng(source['assignment_seed']+batch);treatment=rng.permutation(np.tile([0,1],32));decision=rng.integers(10,45,64)
            policy=RandomSwitchPolicy(actor,source['gaussian_std'],treatment,decision,source['noise_seed']+batch)
            with np.load(root/f'remote/source-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['treatment'],treatment);np.testing.assert_array_equal(z['decision'],decision)
                np.testing.assert_array_equal(z['state_at_decision'],z['observations'][decision,np.arange(64)])
                for t in range(50):
                    action=policy.act(z['observations'][t]);error=float(np.abs(action-z['executed_actions'][t]).max());max_error=max(error,max_error);forward_rows+=64
                    if error>1e-6:raise ValueError('Executedbehavior reconstruction differs')
                    np.testing.assert_allclose(z['raw_proposals'][t],policy.raw_proposal.astype(np.float32),atol=1e-6,rtol=0)
                np.testing.assert_array_equal(z['applied'],policy.applied)
                flags=z['task_flags'].astype(bool);np.testing.assert_array_equal(flags[-1,:,3],z['outcome'])
                np.testing.assert_array_equal(flags[:,:,3],flags[:,:,1]&flags[:,:,2]&~flags[:,:,0])
                lin=np.linalg.norm(z['diagnostic_cube_linear_velocity'],axis=-1);ang=np.linalg.norm(z['diagnostic_cube_angular_velocity'],axis=-1)
                static=(lin<=.01)&(ang<=.5);top=flags[:,:,1];badlin=lin>.01;badang=ang>.5
                stats['on_top_post_action_frames']+=int(top.sum());stats['on_top_nonstatic']+=int((top&~flags[:,:,2]).sum())
                stats['linear_only_failure']+=int((top&badlin&~badang).sum());stats['angular_only_failure']+=int((top&~badlin&badang).sum());stats['both_velocity_failures']+=int((top&badlin&badang).sum());stats['static_info_velocity_mismatches']+=int((static!=flags[:,:,2]).sum())
                # Endpoint rotation is a control-interval average. It is NOT the
                # instantaneous substep angular velocity used by task success.
                q=z['observations'][:,:,28:32].astype(float);q/=np.linalg.norm(q,axis=-1,keepdims=True)
                average=2*np.arccos(np.clip(np.abs((q[1:]*q[:-1]).sum(-1)),0,1))*20
                stats['instantaneous_angular_failure_with_small_control_interval_rotation']+=int((top&badang&(average<=.5)).sum())
        report=json.loads((root/f'remote/source-{index}/run/report.json').read_text())
        if report['environment_reserved']!=102400 or report['environment_completed']!=102400 or report['object_velocity_query_vectors_completed']!=204800:raise ValueError('Unexpectedworkaccounting')
        rows.append(dict(seed=source['training_seed'],source_actor_sha256=source['parent']['weights_sha256'],new_training_transitions=102400,episodes=2048,worker_seconds=report['wall_seconds'],base_successes=sum(b['base_successes'] for b in report['batches']),correction_successes=sum(b['correction_successes'] for b in report['batches']),motion_diagnostics=stats))
    result=dict(state='verified',rows=rows,CPU_actor_forward_rows=forward_rows,max_executed_action_error=max_error,physics_calls=0,elapsed_seconds=time.monotonic()-started,scope='Behaviorauditpluspost-outcomedescriptivemotionalanalysis. Instantaneous-vs-average differences do notproveasimulatorbug. Privilegedvelocitycolumnsneverusedforlearnerfeatures.')
    with (root/'behavior-audit-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
