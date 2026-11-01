"""One-factor diagnostic of the failed stopping toy; no robot computation."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from research.persistent_switch_toy import features,fit_models,rollout,wilson_lower


def gate_from_record(fit):
    def gate(state):
        z=np.column_stack([np.ones(len(state)),(features(state)-fit['center'])/fit['scale']])
        logits=np.clip(z@np.asarray(fit['coefficients']).T,-700,700);p=1/(1+np.exp(-logits))
        return (p[:,1]>=.9)&(p[:,1]-p[:,0]>=.1)
    return gate


def main():
    started=time.monotonic();torch.set_num_threads(2);plan_path=Path('cumulative/experiments/persistent-switch-regularization-v1.json');plan=json.loads(plan_path.read_text())
    original=json.loads(Path('artifacts/cumulative/persistent-switch-toy-v1/report.json').read_text());rows=[]
    for seed,old in zip(plan['seeds'],original['rows']):
        if seed!=old['seed']:raise ValueError('Toy identity differs')
        rng=np.random.default_rng(seed);initial=rng.uniform(1,2,2048);speed=rng.uniform(.02,.08,2048);decision=rng.integers(0,50,2048);treatment=rng.permutation(np.tile([0,1],1024))
        x=initial.copy();state=np.zeros((2048,2))
        for t in range(50):
            selected=decision==t;state[selected]=np.column_stack([x[selected],speed[selected]])
            x-=np.where((treatment==1)&(t>=decision),0,speed)
        outcome=np.abs(x)<=.2
        if outcome.sum()!=old['training_successes']:raise ValueError('Reconstructed training data differ')
        gate,fit=fit_models(state,treatment,outcome,l2=0)
        false_gate,false_fit=fit_models(state,treatment,rng.permutation(outcome),l2=0)
        old_gate=gate_from_record(old['fit'])
        # These draws are fresh and were not used to choose the intervention.
        fresh=np.random.default_rng(seed+100000);cx=fresh.uniform(1,2,512);cv=fresh.uniform(.02,.08,512)
        counts={k:int(rollout(cx,cv,'learned',g).sum()) for k,g in [('old',old_gate),('new',gate),('shuffled',false_gate)]}
        admission={k:bool(wilson_lower(n,512)>=.8) for k,n in counts.items()}
        tx=fresh.uniform(1,2,16384);tv=fresh.uniform(.02,.08,16384)
        base=rollout(tx,tv,'base');learned=rollout(tx,tv,'learned',gate) if admission['new'] else rollout(tx,tv,'base');random=rollout(tx,tv,'random',rng=fresh);oracle=rollout(tx,tv,'oracle')
        retention=float(learned[base].mean());passed=admission['new'] and not admission['old'] and not admission['shuffled'] and learned.mean()-base.mean()>=.40 and oracle.mean()-learned.mean()<=.05 and retention>=.99
        rows.append(dict(seed=seed,calibration_successes=counts,admitted=admission,test_success={k:float(v.mean()) for k,v in [('base',base),('random',random),('learned',learned),('oracle',oracle)]},base_success_retention=retention,screen_passed=bool(passed),fit=fit,negative_fit=false_fit))
    result=dict(state='complete',rows=rows,all_seeds_pass=all(r['screen_passed'] for r in rows),new_toy_transitions=3*(3*512+4*16384)*50,reconstructed_toy_transitions=3*2048*50,robot_transitions=0,fit_closure_and_backward_calls=sum(r[k]['closure_and_backward_calls'] for r in rows for k in ['fit','negative_fit']),elapsed_seconds=time.monotonic()-started,scope=plan['scope'])
    out=Path('artifacts/cumulative/persistent-switch-regularization-v1');out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'helper.py').write_bytes(Path('research/persistent_switch_toy.py').read_bytes());(out/'plan.json').write_bytes(plan_path.read_bytes());(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ['fit','negative_fit']}))


if __name__=='__main__':main()
