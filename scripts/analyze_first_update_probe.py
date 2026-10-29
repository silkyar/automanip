"""Read public diagnostic outputs only; no training, simulator or cloud imports."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_interval(values, rng, repetitions=5000):
    values=np.asarray(values,dtype=float)
    draws=rng.integers(0,len(values),(repetitions,len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(1),[.025,.975])]


def analyze(report):
    if report['state']!='complete' or len(report['collections'])!=3:
        raise ValueError('Incomplete probe; no completed scientific conclusion')
    rng=np.random.default_rng(1793203500)
    records=[];return_diffs=[]
    for row in report['collections']:
        if not row['parity']['passed']:raise ValueError('Production parity failed')
        d=row['diagnostics']
        success={name:np.asarray(v['episode_successes'],dtype=float) for name,v in d.items()}
        delta=np.asarray(d['tiny_forward']['episode_dense_returns'])-np.asarray(d['tiny_reverse']['episode_dense_returns'])
        return_diffs.append(delta)
        loss=float((success['no_update']-success['full']).mean())
        recovery=float((success['bounded']-success['full']).mean())
        records.append(dict(collection_seed=row['seed'],full_success_loss=loss,bounded_success_recovery=recovery,
            small_forward_minus_reverse_return=float(delta.mean()),return_difference_paired_95=paired_interval(delta,rng),
            full_loss_paired_95=paired_interval(success['no_update']-success['full'],rng),
            bounded_recovery_paired_95=paired_interval(success['bounded']-success['full'],rng),
            success_counts={name:int(values.sum()) for name,values in success.items()}))
    # Hierarchical resampling: collection seed clusters, then paired episodes.
    # Only three clusters: descriptive uncertainty, not a well-powered population guarantee.
    cluster_values=[]
    for _ in range(5000):
        clusters=rng.integers(0,3,3)
        cluster_values.append(np.mean([return_diffs[c][rng.integers(0,64,64)].mean() for c in clusters]))
    interval=[float(x) for x in np.quantile(cluster_values,[.025,.975])]
    loss=float(np.mean([r['full_success_loss'] for r in records]))
    recovery=float(np.mean([r['bounded_success_recovery'] for r in records]))
    agreement=sum(r['full_success_loss']>0 and r['bounded_success_recovery']>0 and r['small_forward_minus_reverse_return']>0 for r in records)
    support=loss>=.1 and recovery>=.5*loss and interval[0]>0 and agreement>=2
    if support:decision='H1 supported for this assisted first-update probe only'
    elif interval[1]<0:decision='Locally useful-direction prediction rejected: tiny forward return is worse than reverse'
    elif loss<.1:decision='Initial full-step loss below material-loss criterion; first-step explanation not supported'
    else:decision='Inconclusive; preregistered H1 gate not met'
    return dict(decision=decision,H1_gate_passed=support,per_collection=records,
        equal_collection_mean_full_success_loss=loss,equal_collection_mean_bounded_recovery=recovery,
        small_forward_minus_reverse_return_cluster_95=interval,agreeing_collections=agreement,
        uncertainty_limit='Three collection data draws from one assisted actor; not three training seeds. Bootstrap intervals are exploratory.',
        progress=report['progress'],automatic_followups=False,from_scratch_gain_established=False,
        scope='Actor movement response with fixed original critic/std, not complete alternative PPO procedures.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(json.loads(a.report.read_text()));a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
