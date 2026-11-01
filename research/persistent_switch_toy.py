"""Learn a persistent switch from randomized outcomes, in a scalar toy only."""
import json
from pathlib import Path
import time
import numpy as np
import torch


def features(state):
    x,v=np.asarray(state).T
    return np.column_stack([x,v,x*x,x*v,v*v])


def rollout(initial,speed,kind,gate=None,rng=None):
    x=initial.copy();stopped=np.zeros(len(x),bool)
    for step in range(50):
        state=np.column_stack([x,speed])
        if kind=='learned':stopped|=gate(state)
        elif kind=='random':stopped|=rng.random(len(x))<.02
        elif kind=='oracle':stopped|=np.abs(x)<=.2
        elif kind!='base':raise ValueError('Closed toy controller')
        x-=np.where(stopped,0,speed)
    return np.abs(x)<=.2


def fit_models(state,treatment,outcome,*,l2=.001):
    design=features(state);center=design.mean(0);scale=design.std(0);scale=np.maximum(scale,1e-8)
    design=torch.from_numpy(np.column_stack([np.ones(len(state)),(design-center)/scale])).double()
    models=[];closures=0
    for arm in [0,1]:
        selected=treatment==arm;x=design[selected];y=torch.from_numpy(outcome[selected].astype(float))
        coefficient=torch.zeros(6,dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.LBFGS([coefficient],max_iter=100,line_search_fn='strong_wolfe')
        def closure():
            nonlocal closures
            closures+=1;optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(x@coefficient,y)+l2*coefficient[1:].square().sum()/2
            loss.backward();return loss
        optimizer.step(closure);models.append(coefficient.detach().numpy().copy())
    coefficients=np.stack(models)
    def gate(observations):
        z=np.column_stack([np.ones(len(observations)),(features(observations)-center)/scale])
        logits=np.clip(z@coefficients.T,-700,700);p=1/(1+np.exp(-logits))
        return (p[:,1]>=.9)&(p[:,1]-p[:,0]>=.1)
    return gate,dict(coefficients=coefficients.tolist(),center=center.tolist(),scale=scale.tolist(),closure_and_backward_calls=closures)


def wilson_lower(successes,total):
    z=1.959963984540054;p=successes/total
    return (p+z*z/(2*total)-z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def main():
    start=time.monotonic();torch.set_num_threads(2);plan_path=Path('cumulative/experiments/persistent-switch-toy-v1.json');plan=json.loads(plan_path.read_text());rows=[]
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed);initial=rng.uniform(1,2,2048);speed=rng.uniform(.02,.08,2048)
        decision=rng.integers(0,50,2048);treatment=rng.permutation(np.tile([0,1],1024))
        x=initial.copy();states=np.zeros((2048,2))
        for step in range(50):
            selected=decision==step;states[selected]=np.column_stack([x[selected],speed[selected]])
            x-=np.where((treatment==1)&(step>=decision),0,speed)
        outcome=np.abs(x)<=.2
        gate,fits=fit_models(states,treatment,outcome)
        false_gate,false_fits=fit_models(states,treatment,rng.permutation(outcome))
        cal_x=rng.uniform(1,2,512);cal_v=rng.uniform(.02,.08,512)
        calibrated=rollout(cal_x,cal_v,'learned',gate);negative=rollout(cal_x,cal_v,'learned',false_gate)
        lower=wilson_lower(int(calibrated.sum()),512);false_lower=wilson_lower(int(negative.sum()),512)
        admitted=lower>=.8;negative_admitted=false_lower>=.8
        test_x=rng.uniform(1,2,16384);test_v=rng.uniform(.02,.08,16384)
        base=rollout(test_x,test_v,'base');learned=rollout(test_x,test_v,'learned',gate) if admitted else rollout(test_x,test_v,'base')
        random=rollout(test_x,test_v,'random',rng=rng);oracle=rollout(test_x,test_v,'oracle')
        retention=float(learned[base].mean()) if base.any() else None
        row=dict(seed=seed,training_successes=int(outcome.sum()),calibration_successes=int(calibrated.sum()),calibration_wilson_lower=lower,admitted=bool(admitted),negative_calibration_successes=int(negative.sum()),negative_admitted=bool(negative_admitted),test_success={k:float(v.mean()) for k,v in [('base',base),('random_switch',random),('learned_switch',learned),('oracle',oracle)]},base_success_retention=retention,fit=fits,negative_fit=false_fits)
        row['screen_passed']=bool(admitted and not negative_admitted and learned.mean()-base.mean()>=.40 and oracle.mean()-learned.mean()<=.05 and retention>=.99);rows.append(row)
    result=dict(state='complete',rows=rows,all_seeds_pass=all(r['screen_passed'] for r in rows),robot_transitions=0,toy_episodes_per_seed=2048+1024+4*16384,toy_transitions=3*(2048+1024+4*16384)*50,fit_closure_and_backward_calls=sum(r[k]['closure_and_backward_calls'] for r in rows for k in ['fit','negative_fit']),elapsed_seconds=time.monotonic()-start,scope='Assisted scalar stopping toy. Base controller and zero-actionprior given; no robot sample-efficiency, learned low-level correction or deployment-representation claim.')
    out=Path('artifacts/cumulative/persistent-switch-toy-v1');out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'plan.json').write_bytes(plan_path.read_bytes());(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ['fit','negative_fit']}))


if __name__=='__main__':main()
