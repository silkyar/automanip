"""Two-decision algebra: temporal noise requires its conditional score.

Not a PPO implementation, robot policy, or sample-efficiency result.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np

started=time.monotonic();rng=np.random.default_rng(20261028);n=262144
eps=rng.normal(size=(n,2));sigma=.5;coefficient=.5;target=1.
true_gradient=np.array([-1.,2.]);rows=[]
for rho in (0.,.5,.9):
    a0=sigma*eps[:,0]
    a1=sigma*(rho*eps[:,0]+math.sqrt(1-rho*rho)*eps[:,1])
    reward=-(a1-coefficient*a0-target)**2
    baseline=-(target**2+sigma**2*(1+coefficient**2-2*coefficient*rho))
    exact_score=np.stack([eps[:,0]/sigma-rho*eps[:,1]/(sigma*math.sqrt(1-rho*rho)),
                          eps[:,1]/(sigma*math.sqrt(1-rho*rho))],-1)
    naive_score=np.stack([a0,a1],-1)/sigma**2
    estimates=exact_score*(reward-baseline)[:,None]
    naive=naive_score*(reward-baseline)[:,None]
    naive_expectation=np.array([[1.,rho],[rho,1.]])@true_gradient
    for values,expected in ((estimates,true_gradient),(naive,naive_expectation)):
        assert np.all(np.abs(values.mean(0)-expected)<6*values.std(0,ddof=1)/math.sqrt(n))
    rows.append(dict(rho=rho,true_gradient=true_gradient.tolist(),correct_score_sample_gradient=estimates.mean(0).tolist(),
        naive_marginal_expected_gradient=naive_expectation.tolist(),naive_marginal_sample_gradient=naive.mean(0).tolist(),
        correct_estimator_coordinate_variance=estimates.var(0,ddof=1).tolist(),
        naive_expected_first_coordinate_wrong_sign=bool(naive_expectation[0]*true_gradient[0]<0)))
result=dict(state='passed',seconds=time.monotonic()-started,samples_per_condition=n,
    synthetic_normal_draws=2*n,synthetic_reward_evaluations=3*n,optimizer_calls=0,robot_transitions=0,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=rows,
    conclusion='Keeping stationary Gaussian marginals does not preserve an independent-action likelihood; at rho.9 the naive score reverses one true gradient component.',
    limits='Two-decision unbounded Gaussian reward toy; no robot/policy update, PPO clipping, shared-network dynamics, critic or learning comparison.',
    method_promotion=False)
path=Path(__file__).with_name('autoregressive-score-probe-20261028.json')
if path.exists():raise FileExistsError('Preserve receipt')
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
