"""Analytic Gaussian nested-estimator check before any robot probe."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from continuation_variance import components,cluster_interval

started=time.monotonic();rng=np.random.default_rng(20261028);records=[]
for coefficient,beta in ((.1,1.),(1.,.1)):
    measured=[];intervals=[]
    for repeat in range(1000):
        score=rng.normal(size=(64,8))
        returns=coefficient*score[:,:1]+beta*rng.normal(size=(64,16))
        measured.append(components(returns,score))
        if repeat==0:intervals.append(cluster_interval(returns,score))
    # With an EXACT zero state baseline: action variance9*c², future variance8*b².
    # Estimated independent baseline adds action noise, so do not demand the
    # exact total variance for a different baseline. Within component is invariant.
    expected_within=8*beta*beta
    actual_within=np.mean([r['continuation_noise_trace'] for r in measured])
    assert abs(actual_within/expected_within-1)<.03
    records.append(dict(action_coefficient=coefficient,future_noise=beta,
        exact_continuation_trace=expected_within,mean_estimated_continuation_trace=float(actual_within),
        exact_zero_baseline_action_trace=9*coefficient**2,
        exact_zero_baseline_ideal_max_reduction=(9*coefficient**2+expected_within)/expected_within,
        mean_estimated_action_trace=float(np.mean([r['corrected_action_variance_trace_unclipped'] for r in measured])),
        mean_estimated_first_gradient=float(np.mean([r['estimated_local_gradient'][0] for r in measured])),
        true_first_gradient=coefficient,first_replica_interval=intervals[0]))
receipt=dict(state='passed',seconds=time.monotonic()-started,synthetic_independent_experiments=2000,
    synthetic_continuation_returns=2000*64*16,optimizer_calls=0,robot_transitions=0,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    estimator_sha256=hashlib.sha256(Path(__file__).with_name('continuation_variance.py').read_bytes()).hexdigest(),
    records=records,interpretation='Identifies continuation variance independently of baseline; finite16-future groups require noise correction; intervals are approximate and not global-policy bounds.')
p=Path(__file__).with_name('continuation-variance-gate-20261028.json')
if p.exists():raise FileExistsError('Preserve receipt')
p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
