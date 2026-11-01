"""One-seed three-control screen; never promote it as replicated efficiency."""
import argparse
import json
from pathlib import Path

from scripts.analyze_scratch_credit import normalized_auc, sustained_crossing


METHODS = ('full', 'masked', 'hold')
GRID = [0, 512000, 1024000, 2048000, 4096000]


def analyze(reports, receipts):
    if len(reports) != 3 or len(receipts) != 3:
        raise ValueError('All three controls required')
    indexed, work = {}, {}
    for report, receipt in zip(reports, receipts):
        method = report['method']
        if (method not in METHODS or method in indexed or report['seed'] != 41
                or receipt['method'] != method or receipt['seed'] != 41
                or report['state'] != 'complete' or receipt['state'] != 'complete'
                or report['confirmation_queries'] or report['final_queries']):
            raise ValueError('Incomplete, duplicate, or mislabeled trial')
        if receipt['actual_work']['environment_completed'] > 4096000:
            raise ValueError('Training interaction cap exceeded')
        points = {}
        for point in report['points']:
            if (point['steps'] in points or point['steps'] > receipt['actual_work']['environment_completed']
                    or point['episodes'] != 128 or point['failures']
                    or not 0 <= point['successes'] <= 128):
                raise ValueError('Invalid checkpoint accounting')
            points[point['steps']] = point
        if report['environment_completed'] != len(points)*128*50:
            raise ValueError('Scoring work differs from completed episodes')
        indexed[method], work[method] = points, receipt
    common = [s for s in GRID if all(s in indexed[m] for m in METHODS)]
    if len(common) < 2 or common != [s for s in GRID if s <= common[-1]]:
        raise ValueError('Incomplete common checkpoint prefix')
    methods = {}
    for method in METHODS:
        points = indexed[method]
        curve = [(s, points[s]['successes']/128) for s in common]
        methods[method] = dict(success_curve=curve, normalized_success_auc=normalized_auc(curve),
            T80=sustained_crossing(curve, .8), T90=sustained_crossing(curve, .9),
            actual_training_transitions=work[method]['actual_work']['environment_completed'],
            optimizer_calls=work[method]['actual_work']['optimizer_completed'],
            trainer_process_seconds=work[method]['trainer_process_seconds'],
            wrapper_seconds=work[method]['wrapper_seconds'],
            scorer_seconds=next(r['wall_seconds'] for r in reports if r['method'] == method))
    contrasts = {}
    for control in ('full', 'masked'):
        contrasts[control] = dict(
            auc_gain=methods['hold']['normalized_success_auc']-methods[control]['normalized_success_auc'],
            final_success_gain=methods['hold']['success_curve'][-1][1]-methods[control]['success_curve'][-1][1])
    passed = all(c['auc_gain'] >= .10 and c['final_success_gain'] >= 0 for c in contrasts.values())
    return dict(format='automanip-gripper-hold-analysis-v1', state='complete', training_seed=41,
        common_steps=common, methods=methods, hold_minus_controls=contrasts,
        single_seed_material_screen_passed=passed,
        new_training_transitions=sum(r['actual_work']['environment_completed'] for r in receipts),
        new_public_transitions=sum(r['environment_completed'] for r in reports),
        confirmation_queries=0, final_queries=0,
        decision='Replicate fixed candidate with both controls before promotion' if passed else
                 'Reject this bounded hold-prior screen; no automatic variant or extension',
        scope='One fresh training seed and one task with explicit analytic actuator prior. Not replicated sample-efficiency evidence; report compute separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--scores', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    receipts = [json.loads((args.training/f'trial-{i}/receipt.json').read_text()) for i in range(3)]
    reports = [json.loads((args.scores/f'trial-{i}/report.json').read_text()) for i in range(3)]
    result = analyze(reports, receipts)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
