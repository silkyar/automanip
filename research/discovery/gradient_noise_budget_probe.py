"""Cheap falsification of action-control-variate promises; synthetic only."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def main():
    started=time.monotonic()
    rng=np.random.default_rng(20261028);n=262144;sigma=.5;target=.5
    z=rng.normal(size=n);future=rng.normal(size=n);action=sigma*z
    score=z/sigma
    # Exact trace-variance-optimal state-only baseline, not fitted to these data.
    baseline=-(target**2+3*sigma**2)
    true_gradient=2*target
    records=[]
    for beta in (0.,.1,1.,3.):
        reward=-(action-target)**2+beta*future
        plain=score*(reward-baseline)
        # Linear Q-Prop around mean0: slope2*target, add back its exact mean term.
        corrected=score*(reward-baseline-2*target*action)+2*target
        within=beta**2/sigma**2
        exact_plain=6*sigma**2+8*target**2+within
        exact_corrected=6*sigma**2+within
        row=dict(future_noise_scale=beta,true_gradient=true_gradient,
            sample_plain_mean=float(plain.mean()),sample_linear_cv_mean=float(corrected.mean()),
            sample_plain_variance=float(plain.var(ddof=1)),
            sample_linear_cv_variance=float(corrected.var(ddof=1)),
            exact_plain_variance=exact_plain,exact_linear_cv_variance=exact_corrected,
            irreducible_future_variance=within,
            ideal_all_action_variance_removed_max_reduction=None if within==0 else exact_plain/within,
            exact_linear_cv_reduction=exact_plain/exact_corrected)
        for values,expected in ((plain,exact_plain),(corrected,exact_corrected)):
            assert abs(values.mean()-true_gradient)<6*math.sqrt(expected/n)
            assert abs(values.var(ddof=1)/expected-1)<.04
        records.append(row)
    # A zero-mean variate ceases to be zero-mean under same-sample gating.
    control=z*z-1
    gated=np.where(control>0,control,0)
    exact_gate_bias=-2*math.exp(-.5)/math.sqrt(2*math.pi)
    sample_gate_bias=-float(gated.mean())
    independent_gate=(rng.normal(size=n)**2-1)>0
    independent_residual=float((independent_gate*control).mean())
    assert abs(sample_gate_bias-exact_gate_bias)<.012
    assert abs(independent_residual)<.012
    result=dict(state='passed',kind='synthetic estimator algebra/Monte Carlo check, not robot learning',
        seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples_per_condition=n,unique_synthetic_normal_draws=3*n,
        synthetic_reward_evaluations=4*n,optimizer_calls=0,robot_transitions=0,
        records=records,same_sample_gate_exact_gradient_bias=exact_gate_bias,
        same_sample_gate_sample_gradient_bias=sample_gate_bias,
        independent_gate_sample_residual=independent_residual,
        conclusion='Even perfect removal of current-action variance cannot remove continuation noise; unbiasedness is not automatic under same-sample adaptive gating.',
        novelty_claim=False,robot_method_promotion=False)
    path=Path(__file__).with_name('gradient-noise-budget-probe-20261028.json')
    if path.exists():raise FileExistsError('Preserve the registered receipt')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
