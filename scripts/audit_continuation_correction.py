"""Reconstruct all frozen learned-selector prototype actions on CPU."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from automanip.policy_io import load_policy,PolicySchema
from automanip.numpy_inference import NumpyPolicy
from automanip.stackcube import STACK_TASK
from research.continuation_correction_policy import ContinuationCorrectionPolicy,BRANCHES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();root=a.archive;started=time.monotonic()
    plan=json.loads((root/'provenance/inputs/plan.json').read_text());rows=[];forwards=0;max_error=0.
    for index,seed in enumerate([2,3]):
        schema=PolicySchema(STACK_TASK.fingerprint,48,8);base=NumpyPolicy(load_policy(root/'provenance/inputs'/plan['policies'][str(seed)],schema),schema)
        head=json.loads((root/'provenance/inputs'/plan['heads'][str(seed)]).read_text())
        aggregate={b:dict(episodes=0,successes=0,triggered=0,initial_triggers=0,trigger_times=[],pretrigger_on_top=0) for b in BRANCHES}
        for batch in range(8):
            assignment=np.random.default_rng(plan['assignment_seed_bases'][index]+batch).permutation(np.tile(np.arange(4),16))
            policy=ContinuationCorrectionPolicy(base,plan['gaussian_std'][str(seed)],assignment,head,plan['timing_seed_bases'][index]+batch);policy.reset_noise(plan['noise_seed_bases'][index]+batch)
            with np.load(root/f'remote/collection-{index}/run/batch-{batch}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['assignment'],assignment)
                for t in range(50):
                    actions=np.asarray(policy.act(z['observations'][t]),np.float32);error=float(np.abs(actions-z['executed_actions'][t]).max());max_error=max(error,max_error);forwards+=64
                    if error>1e-6:raise ValueError('Executedprototypeactiondiffers')
                for key,value in [('trigger',policy.trigger),('applied',policy.applied),('override_masks',policy.override_masks),('scheduled_time',policy.scheduled_time)]:np.testing.assert_array_equal(z[key],value)
                np.testing.assert_allclose(z['value_scores'],policy.scores,atol=1e-6,rtol=0);np.testing.assert_allclose(z['support_leverages'],policy.leverages,atol=1e-5,rtol=0)
                for group,b in enumerate(BRANCHES):
                    selected=np.flatnonzero(assignment==group);r=aggregate[b];r['episodes']+=16;r['successes']+=int(z['stage_flags'][-1,selected,6].sum())
                    for lane in selected:
                        trigger=int(policy.trigger[lane])
                        if trigger>=0:
                            r['triggered']+=1;r['initial_triggers']+=trigger==0;r['trigger_times'].append(trigger)
                            if trigger>0:r['pretrigger_on_top']+=int(z['stage_flags'][trigger-1,lane,3])
        rows.append(dict(seed=seed,groups=aggregate))
    result=dict(state='verified',rows=rows,CPU_policy_forward_rows=forwards,max_executed_action_error=max_error,physics_calls=0,elapsed_seconds=time.monotonic()-started,scope='CPUbehaviorreconstructionanddescriptivetriggercoverage; conditionaltriggeredsuccessisnotarandomizedcausalsubgroupeffect')
    with (root/'action-audit-v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
