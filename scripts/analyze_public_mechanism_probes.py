"""Analyze the registered October 30 probes; no simulator or cloud imports.

Intervals condition on the fixed actors and fitted directions. Whole vector
batches are resampled within each collection; collection IDs are NOT training
seeds. Thresholds and random seeds come from the pre-outcome protocols.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def episode_values(record, field):
    values = np.asarray(record[field], dtype=float)
    if values.shape != (512,) or not np.isfinite(values).all():
        raise ValueError('Expected 512 finite episode values')
    return values


def paired_summary(differences, draws):
    """Equal collection means; paired 64-lane batches within each collection."""
    values = np.asarray(differences, dtype=float)
    if values.shape != (draws.shape[0], 512) or not np.isfinite(values).all():
        raise ValueError('Wrong collection/episode shape')
    batches = values.reshape(values.shape[0], 8, 64).mean(axis=2)
    resampled = np.stack([batches[c][draws[c]].mean(axis=1)
                          for c in range(len(batches))]).mean(axis=0)
    return dict(mean=float(values.mean()), per_collection=values.mean(axis=1).tolist(),
                paired_95=np.quantile(resampled, [.025, .975]).tolist(),
                paired_97_5=np.quantile(resampled, [.0125, .9875]).tolist(),
                paired_98_75=np.quantile(resampled, [.00625, .99375]).tolist())


def validate(report, expected_records, expected_transitions):
    if report['state'] != 'complete':
        raise ValueError('Incomplete execution; no scientific conclusion')
    if len(report['records']) != expected_records:
        raise ValueError('Incomplete record set')
    if report['new_simulator_transitions_completed'] != expected_transitions:
        raise ValueError('Incomplete transition count')
    for name in ['optimizer_calls', 'protected_benchmark_queries', 'confirmation_queries', 'final_queries']:
        if report.get(name, 0) != 0:
            raise ValueError('Unexpected protected query or update')
    if report['initial_observation_max_difference'] > 1e-5:
        raise ValueError('Physical reset pairing failed')
    for row in report['records']:
        if row['execution_failures'] or row['episodes'] != 512:
            raise ValueError('Execution failures or incomplete episodes')
        success = episode_values(row, 'episode_successes')
        if not np.isin(success, [0, 1]).all() or success.sum() != row['successes']:
            raise ValueError('Invalid success vector')
        for field in ['episode_dense_returns', 'episode_discounted_returns']:
            episode_values(row, field)


def calibration(reports):
    if len(reports) != 3:
        raise ValueError('Exactly three collection reports required')
    for report in reports:
        validate(report, 12, 307200)
    expected = [90282001, 90282002, 90282003]
    records = []
    branches = ['no_update', 'known_harmful', 'constant_forward', 'constant_reverse', 'state_forward', 'state_reverse']
    keys = {(m, b) for m in ['deterministic', 'stochastic'] for b in branches}
    for seed, report in zip(expected, reports):
        rows = {(r['mode'], r['branch']): r for r in report['records']}
        if set(rows) != keys or any(r['collection_seed'] != seed for r in rows.values()):
            raise ValueError('Wrong collection IDs or duplicate/missing branches')
        records.append(rows)
    draws = np.random.default_rng(93609999).integers(0, 8, (3, 10000, 8))

    def contrast(mode, left, right, field):
        return paired_summary([episode_values(r[mode, left], field) -
                               episode_values(r[mode, right], field) for r in records], draws)

    damage = contrast('deterministic', 'no_update', 'known_harmful', 'episode_successes')
    guard = damage['mean'] >= .05
    methods = {}
    for method in ['constant', 'state']:
        forward_reverse = contrast('stochastic', method+'_forward', method+'_reverse', 'episode_discounted_returns')
        forward_unchanged = contrast('stochastic', method+'_forward', 'no_update', 'episode_discounted_returns')
        success_gain = contrast('stochastic', method+'_forward', 'no_update', 'episode_successes')
        passed = (guard and forward_reverse['paired_98_75'][0] > 0 and
                  forward_unchanged['paired_98_75'][0] > 0 and success_gain['mean'] >= -.05)
        methods[method] = dict(forward_minus_reverse_return=forward_reverse,
                               forward_minus_unchanged_return=forward_unchanged,
                               forward_minus_unchanged_success=success_gain,
                               proxy_false_negative=bool(passed))
    counts = {mode: {b: sum(r[mode, b]['successes'] for r in records) for b in branches}
              for mode in ['deterministic', 'stochastic']}
    return dict(format='automanip-proxy-calibration-analysis-v1',
                negative_control_damage=damage, negative_control_passed=guard,
                methods=methods, pooled_success_counts=counts, episodes_per_pooled_branch=1536,
                state_minus_constant_return=contrast('stochastic', 'state_forward', 'constant_forward', 'episode_discounted_returns'),
                state_minus_constant_success=contrast('stochastic', 'state_forward', 'constant_forward', 'episode_successes'),
                proxy_false_negative_established=any(m['proxy_false_negative'] for m in methods.values()),
                new_simulator_transitions=921600, optimizer_calls=0,
                worker_seconds=[r['wall_seconds'] for r in reports],
                scope='One assisted actor, three fixed data-derived directions. Conditional reset/noise intervals; not training-seed uncertainty. No complete learner/sample-efficiency result.')


def temporal(reports):
    if len(reports) != 2:
        raise ValueError('Exactly two fixed-actor reports required')
    for report in reports:
        validate(report, 4, 102400)
    draws = np.random.default_rng(93809999).integers(0, 8, (1, 10000, 8))
    actors = {}
    for actor, report in zip(['teacher50M', 'bcclone'], reports):
        rows = {r['branch']: r for r in report['records']}
        if set(rows) != {'no_update', 'iid', 'ar1', 'iid_sum_matched'} or any(r['actor_id'] != actor for r in rows.values()):
            raise ValueError('Wrong actor IDs or duplicate/missing branches')

        def contrast(left, right, field):
            return paired_summary([episode_values(rows[left], field) - episode_values(rows[right], field)], draws)

        ratio = rows['ar1']['latent_mean_square'] / rows['iid']['latent_mean_square']
        actors[actor] = dict(success_counts={b: r['successes'] for b, r in rows.items()},
                            episodes_per_branch=512, latent_second_moment_ratio=ratio,
                            noise_calibration_passed=.9 <= ratio <= 1.1,
                            ar_minus_iid_success=contrast('ar1', 'iid', 'episode_successes'),
                            ar_minus_iid_discounted_return=contrast('ar1', 'iid', 'episode_discounted_returns'),
                            ar_minus_sum_matched_success=contrast('ar1', 'iid_sum_matched', 'episode_successes'),
                            iid_minus_deterministic_success=contrast('iid', 'no_update', 'episode_successes'),
                            diagnostics={b: {k: v for k, v in r.items() if k in ['latent_mean_square', 'latent_lag1_correlation', 'proposal_clipping_fraction', 'executed_action_delta_MSE', 'stage_ever_rates']} for b, r in rows.items()})
    bc = actors['bcclone']; teacher = actors['teacher50M']
    gain = bc['ar_minus_iid_success']
    primary = (bc['noise_calibration_passed'] and gain['mean'] >= .05 and gain['paired_97_5'][0] > 0)
    return dict(format='automanip-temporal-noise-analysis-v1', actors=actors,
                primary_improvement_criterion_passed=bool(primary),
                teacher_point_retention_criterion_passed=teacher['ar_minus_iid_success']['mean'] >= -.05,
                new_simulator_transitions=204800, optimizer_calls=0,
                worker_seconds=[r['wall_seconds'] for r in reports],
                scope='Frozen-policy noise intervention, not exploration during training or sample efficiency. Teacher uses declared common intervention scale; not its native training noise.')


def credit_target(reports):
    if len(reports) != 3:
        raise ValueError('Exactly three collection reports required')
    methods = ['gae_legacy', 'gae_state', 'mc_state']
    branches = ['no_update'] + [m+'_'+sign for m in methods for sign in ['forward', 'reverse']]
    records = []
    for seed, report in zip([90282001, 90282002, 90282003], reports):
        validate(report, 7, 179200)
        rows = {r['branch']: r for r in report['records']}
        if set(rows) != set(branches) or any(r['collection_seed'] != seed or r['mode'] != 'stochastic' for r in rows.values()):
            raise ValueError('Wrong collection or duplicate/missing branches')
        records.append(rows)
    draws = np.random.default_rng(93909999).integers(0, 8, (3, 10000, 8))

    def contrast(left, right, field='episode_discounted_returns'):
        return paired_summary([episode_values(r[left], field)-episode_values(r[right], field) for r in records], draws)

    primary = {}
    for name, left, right in [('target', 'mc_state_forward', 'gae_state_forward'),
                              ('baseline', 'gae_state_forward', 'gae_legacy_forward')]:
        result = contrast(left, right)
        primary[name] = dict(discounted_return=result,
                             material_local_benefit=bool(result['mean'] >= .03 and result['paired_97_5'][0] > 0),
                             reversal_supported=bool(result['paired_97_5'][1] < 0),
                             success=contrast(left, right, 'episode_successes'))
    return dict(format='automanip-credit-target-analysis-v1', primary=primary,
                diagnostics={m: dict(forward_minus_reverse=contrast(m+'_forward', m+'_reverse'),
                                     forward_minus_unchanged=contrast(m+'_forward', 'no_update'),
                                     success_minus_unchanged=contrast(m+'_forward', 'no_update', 'episode_successes')) for m in methods},
                pooled_success_counts={b: sum(r[b]['successes'] for r in records) for b in branches},
                episodes_per_pooled_branch=1536, new_simulator_transitions=537600, optimizer_calls=0,
                worker_seconds=[r['wall_seconds'] for r in reports],
                scope='One-step assisted-policy target/baseline effects under fixed readout geometry. Conditional reset/noise intervals, not complete learners or training-seed uncertainty. GAE/MC target estimands differ.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['calibration', 'temporal', 'credit-target'])
    p.add_argument('reports', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads(path.read_text()) for path in a.reports]
    result = {'calibration': calibration, 'temporal': temporal, 'credit-target': credit_target}[a.kind](reports)
    result['inputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in a.reports}
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['actors', 'inputs', 'methods']}, indent=2))


if __name__ == '__main__':
    main()
