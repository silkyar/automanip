"""Analytically check waiting-value regression before using robot data."""
import json
from pathlib import Path
import time
import numpy as np
from research.least_squares_stopping import fit_continuation,evaluate_paths


def paths(rng,n):
    z=rng.integers(0,2,n);good=rng.random(n)<np.where(z==0,.2,.8)
    state=np.stack([np.column_stack([np.ones(n),z,np.zeros(n)]),np.column_stack([np.ones(n),z,good])])
    payoff=np.stack([np.full(n,.4),np.where(good,.9,.1)])
    return state,payoff,np.ones((2,n),bool),np.zeros(n)


def main():
    start=time.monotonic();plan_path=Path('cumulative/experiments/learned-waiting-toy-v1.json');plan=json.loads(plan_path.read_text());rows=[]
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed);train=paths(rng,512);test=paths(rng,32768)
        state_model=fit_continuation(*train);time_model=fit_continuation(train[0][:,:,:1],*train[1:])
        value,times=evaluate_paths(*test,state_model);constant,_=evaluate_paths(test[0][:,:,:1],*test[1:],time_model)
        threshold=np.where(test[1][1]>=.8,test[1][1],0);greedy=np.full(len(value),.4);clairvoyant=test[1].max(0)
        row=dict(seed=seed,state_value=float(value.mean()),time_only_value=float(constant.mean()),fixed_threshold_value=float(threshold.mean()),greedy_base_value=float(greedy.mean()),INVALID_clairvoyant_value=float(clairvoyant.mean()),low_signal_stops_at0=float((times[test[0][0,:,1]==0]==0).mean()),high_signal_waits=float((times[test[0][0,:,1]==1]==1).mean()))
        row['screen_passed']=bool(abs(row['state_value']-.57)<=.02 and row['state_value']-row['time_only_value']>=.05 and row['state_value']-row['fixed_threshold_value']>=.08 and row['INVALID_clairvoyant_value']>.62);rows.append(row)
    result=dict(state='complete',rows=rows,all_seeds_pass=all(r['screen_passed'] for r in rows),toy_generated_transitions=199680,robot_transitions=0,linear_solves=12,elapsed_seconds=time.monotonic()-start,scope=plan['scope'])
    output=Path('artifacts/cumulative/learned-waiting-toy-v1');output.mkdir(exist_ok=False)
    for source in [Path(__file__),Path('research/least_squares_stopping.py')]:
        (output/source.name).write_bytes(source.read_bytes())
    (output/'plan.json').write_bytes(plan_path.read_bytes());(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
