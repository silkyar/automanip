"""A limited invariant-set argument, with its nonconvex counterexample."""
import json
from pathlib import Path
import time
import numpy as np
from research.persistent_switch_toy import rollout,wilson_lower


def support_gate(state,treatment,outcome):
    positive=state[(treatment==1)&outcome,0]
    if len(positive)<2:raise ValueError('At least2successful corrective outcomes required')
    low,high=float(positive.min()),float(positive.max())
    return lambda obs:(obs[:,0]>=low)&(obs[:,0]<=high),dict(low=low,high=high,successful_support_points=len(positive))


def stateless_rollout(initial,speed,gate):
    x=initial.copy()
    for _ in range(50):x-=np.where(gate(np.column_stack([x,speed])),0,speed)
    return np.abs(x)<=.2


def main():
    started=time.monotonic();plan_path=Path('cumulative/experiments/invariant-support-toy-v1.json');plan=json.loads(plan_path.read_text())
    original=json.loads(Path('artifacts/cumulative/persistent-switch-toy-v1/report.json').read_text());output=Path('artifacts/cumulative/invariant-support-toy-v1');output.mkdir(exist_ok=False);rows=[]
    for seed,old in zip(plan['seeds'],original['rows']):
        rng=np.random.default_rng(seed);initial=rng.uniform(1,2,2048);speed=rng.uniform(.02,.08,2048);decision=rng.integers(0,50,2048);treatment=rng.permutation(np.tile([0,1],1024))
        x=initial.copy();state=np.zeros((2048,2))
        for t in range(50):
            selected=decision==t;state[selected]=np.column_stack([x[selected],speed[selected]])
            x-=np.where((treatment==1)&(t>=decision),0,speed)
        outcome=np.abs(x)<=.2
        if outcome.sum()!=old['training_successes']:raise ValueError('Toy reconstruction differs')
        np.savez_compressed(output/f'training-{seed}.npz',state=state,treatment=treatment,outcome=outcome,decision=decision,initial=initial,speed=speed)
        gate,region=support_gate(state,treatment,outcome);false_gate,false_region=support_gate(state,treatment,rng.permutation(outcome))
        fresh=np.random.default_rng(seed+200000);cx=fresh.uniform(1,2,512);cv=fresh.uniform(.02,.08,512)
        calibration=rollout(cx,cv,'learned',gate);baseline_calibration=rollout(cx,cv,'base');negative=rollout(cx,cv,'learned',false_gate)
        admitted=wilson_lower(int(calibration.sum()),512)>=.8;false_admitted=wilson_lower(int(negative.sum()),512)>=.8
        tx=np.concatenate([fresh.uniform(1,2,8192),fresh.uniform(.5,2.5,8192)]);tv=np.concatenate([fresh.uniform(.02,.08,8192),fresh.uniform(.01,.1,8192)])
        base=rollout(tx,tv,'base');learned=rollout(tx,tv,'learned',gate) if admitted else rollout(tx,tv,'base');stateless=stateless_rollout(tx,tv,gate) if admitted else rollout(tx,tv,'base');oracle=rollout(tx,tv,'oracle')
        splits={}
        for name,section in [('original',slice(0,8192)),('wider_initial_conditions',slice(8192,None))]:
            b,l,o=base[section],learned[section],oracle[section]
            splits[name]=dict(base_success=float(b.mean()),learned_success=float(l.mean()),oracle_success=float(o.mean()),base_success_retention=float(l[b].mean()))
        passed=admitted and not false_admitted and np.array_equal(learned,stateless) and splits['original']['learned_success']-splits['original']['base_success']>=.40 and all(s['oracle_success']-s['learned_success']<=.05 and s['base_success_retention']>=.99 for s in splits.values())
        rows.append(dict(seed=seed,region=region,negative_region=false_region,calibration_successes=int(calibration.sum()),calibration_baseline_successes=int(baseline_calibration.sum()),calibration_base_success_retention=float(calibration[baseline_calibration].mean()),admitted=bool(admitted),negative_admitted=bool(false_admitted),splits=splits,latched_stateless_equal=bool(np.array_equal(learned,stateless)),screen_passed=bool(passed)))
    # The same interpolation claim is false without the convex/invariant premise.
    x=np.zeros(128);speed=np.linspace(.0021,.0039,128);baseline=x.copy();wrong=x.copy()
    for _ in range(50):
        baseline-=speed;wrong-=np.where((wrong>=-.15)&(wrong<=.15),0,speed)
    target=lambda z:((z>=-.2)&(z<=-.1))|((z>=.1)&(z<=.2))
    counterexample=dict(base_successes=int(target(baseline).sum()),unsupported_interpolation_successes=int(target(wrong).sum()),episodes=128)
    result=dict(state='complete',rows=rows,all_seeds_pass=all(r['screen_passed'] for r in rows),nonconvex_assumption_counterexample=counterexample,new_toy_transitions=10073600,reconstructed_toy_transitions=307200,robot_transitions=0,optimizer_calls=0,learned_interval_fits=6,elapsed_seconds=time.monotonic()-started,scope=plan['scope'])
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'plan.json').write_bytes(plan_path.read_bytes());(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
