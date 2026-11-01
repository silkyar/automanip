"""Nested-rollout moment estimator, with no trainer or simulator dependencies."""
import numpy as np


def components(returns,score,calibration_actions=16):
    returns=np.asarray(returns,dtype=np.float64);score=np.asarray(score,dtype=np.float64)
    if returns.ndim!=2 or score.ndim!=2 or returns.shape[0]!=score.shape[0]:
        raise ValueError('Expected[action,continuation] returns and[action,coordinate] scores')
    if not np.isfinite(returns).all() or not np.isfinite(score).all():raise ValueError('Finite data required')
    if returns.shape[1]<2 or not 1<=calibration_actions<returns.shape[0]-2:raise ValueError('Independent calibration and repeated continuations required')
    norms=(score*score).sum(-1)
    denominator=norms[:calibration_actions].sum()
    if denominator<=0:raise ValueError('Nonzero calibration score required')
    baseline=float((norms[:calibration_actions]*returns[:calibration_actions].mean(-1)).sum()/denominator)
    s=score[calibration_actions:];g=returns[calibration_actions:]
    group_mean=s*(g.mean(-1)-baseline)[:,None]
    within=float(((s*s).sum(-1)*g.var(-1,ddof=1)).mean())
    noisy_between=float(group_mean.var(0,ddof=1).sum())
    corrected_between=noisy_between-within/g.shape[1]
    total=corrected_between+within
    return dict(calibration_baseline=baseline,calibration_actions=calibration_actions,
        measurement_actions=len(s),continuations_per_action=g.shape[1],score_coordinates=s.shape[1],
        estimated_local_gradient=group_mean.mean(0).tolist(),
        continuation_noise_trace=within,noisy_group_mean_variance_trace=noisy_between,
        corrected_action_variance_trace_unclipped=corrected_between,
        estimated_total_single_rollout_variance_trace=total,
        ideal_all_action_removed_reduction_unclipped=None if within==0 else total/within)


def cluster_interval(returns,score,calibration_actions=16,repeats=1000,seed=20261028):
    """Percentile bootstrap of independent action clusters, not branch pseudo-N.

    Reuses the whole observed continuation group when resampling an action.
    Resampling individual futures again would add extra group-mean noise.
    This is an approximate small-sample interval, not a certified upper bound.
    """
    rng=np.random.default_rng(seed);returns=np.asarray(returns);score=np.asarray(score)
    n=len(returns);ratios=[]
    for _ in range(repeats):
        index=np.concatenate([rng.integers(0,calibration_actions,calibration_actions),
            rng.integers(calibration_actions,n,n-calibration_actions)])
        value=components(returns[index],score[index],calibration_actions)['ideal_all_action_removed_reduction_unclipped']
        if value is not None:ratios.append(value)
    return dict(kind='approximate percentile bootstrap across action clusters; no confidence guarantee',
        action_cluster_resamples=repeats,seed=seed,
        reduction_95_percentile_interval=None if not ratios else np.quantile(ratios,[.025,.975]).tolist())
