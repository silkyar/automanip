"""Essential future-innovation control-variate check, without robot physics.

Coefficients are trained on separate trajectories. Subtracting a *future*
conditionally zero-mean control preserves the expectation of the earlier score.
Subtracting a current-action control needs its analytic score correction.
"""
import json
import math
from pathlib import Path
import time
import numpy as np


def data(epsilon,clipped):
    actions=np.clip(epsilon,-1,1) if clipped else epsilon
    state=np.cumsum(actions,axis=1)-actions
    return state,actions.sum(1)


def future_sum(control,gamma):
    result=np.zeros_like(control);tail=np.zeros(control.shape[0])
    for t in reversed(range(control.shape[1])):
        result[:,t]=gamma*tail;tail=control[:,t]+gamma*tail
    return result


def estimate(epsilon,advantages,correction=None):
    scores=np.stack([epsilon,epsilon**2-1],axis=-1)
    out=(scores*advantages[:,:,None]).sum(1)
    if correction is not None:out+=correction
    return out


def fit(epsilon,state,returns):
    coeff=[];beta=[]
    for t in range(epsilon.shape[1]):
        design=np.stack([state[:,t],np.ones(len(state))],axis=1)
        value=np.linalg.lstsq(design,returns[:,t],rcond=None)[0]
        residual=returns[:,t]-design@value
        coeff.append(value);beta.append(float(epsilon[:,t]@residual/(epsilon[:,t]@epsilon[:,t])))
    return np.asarray(coeff),np.asarray(beta)


def summarize(gradients,truth):
    mean=gradients.mean(0);cov=np.cov(gradients,rowvar=False,ddof=1)
    se=np.sqrt(np.diag(cov)/len(gradients))
    return dict(mean=mean.tolist(),standard_error=se.tolist(),truth=truth.tolist(),covariance_trace=float(np.trace(cov)),within4standarderrors=bool(np.all(np.abs(mean-truth)<=4*se+1e-10)))


def main():
    started=time.monotonic();plan_path=Path('cumulative/experiments/future-innovation-toy-v1.json');plan=json.loads(plan_path.read_text())
    rng=np.random.default_rng(plan['seed']);train=rng.standard_normal((4096,50));test=rng.standard_normal((32768,50));cells=[]
    for clipped in [False,True]:
        x,r=data(train,clipped);xt,rt=data(test,clipped)
        for gamma in plan['gammas']:
            weight=gamma**np.arange(49,-1,-1)
            targets=r[:,None]*weight;test_returns=rt[:,None]*weight
            value,beta=fit(train,x,targets)
            baseline=xt*value[:,0]+value[:,1]
            advantage=test_returns-baseline
            control=test*beta
            future=future_sum(control,gamma)
            correction=np.array([beta.sum(),0.])
            probability=math.erf(1/math.sqrt(2)) if clipped else 1.
            truth=np.array([probability*weight.sum(),0.])
            cases={'state_baseline_MC':estimate(test,advantage),
                   'future_only':estimate(test,advantage-future),
                   'current_corrected':estimate(test,advantage-control,correction),
                   'full_corrected':estimate(test,advantage-future-control,correction),
                   'NEGATIVE_current_uncorrected':estimate(test,advantage-future-control),
                   'NEGATIVE_leaky_future':estimate(test,advantage-future_sum(xt*test**2,gamma))}
            if not clipped:
                exact_adv=test_returns-xt*weight
                exact_control=test*weight
                cases['oracle_future']=estimate(test,exact_adv-future_sum(exact_control,gamma))
                cases['oracle_full']=estimate(test,exact_adv-future_sum(exact_control,gamma)-exact_control,np.array([weight.sum(),0.]))
            summaries={k:summarize(v,truth) for k,v in cases.items()}
            ratio=summaries['future_only']['covariance_trace']/summaries['state_baseline_MC']['covariance_trace']
            cells.append(dict(gamma=gamma,dynamics='clipped' if clipped else 'linear',coefficient_beta=beta.tolist(),estimates=summaries,future_variance_ratio=ratio,screen_passed=ratio<=.75 and summaries['future_only']['within4standarderrors'],negative_controls_detected=all(not summaries[k]['within4standarderrors'] for k in ['NEGATIVE_current_uncorrected','NEGATIVE_leaky_future'])))
    output=Path('artifacts/cumulative/future-innovation-toy-v1');output.mkdir(exist_ok=False)
    result=dict(state='complete',cells=cells,all_cells_pass=all(c['screen_passed'] for c in cells),negative_controls_detected=all(c['negative_controls_detected'] for c in cells),new_robot_transitions=0,optimizer_calls=0,toy_generated_trajectories=36864,toy_generated_transitions=1843200,time_specific_two_coefficient_value_fits=200,time_specific_scalar_control_fits=200,elapsed_seconds=time.monotonic()-started,scope=plan['scope'])
    (output/'source.py').write_bytes(Path(__file__).read_bytes());(output/'plan.json').write_bytes(plan_path.read_bytes())
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_cells_pass=result['all_cells_pass'],negative_controls_detected=result['negative_controls_detected'],cells=[{k:c[k] for k in ['gamma','dynamics','future_variance_ratio','screen_passed']} for c in cells],elapsed_seconds=result['elapsed_seconds']),indent=2))


if __name__=='__main__':main()
