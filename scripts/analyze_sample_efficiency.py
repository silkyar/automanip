"""Summarize sample-efficiency score reports: per-trial curves, sustained T80/T90, success AUC, and arm medians.

Usage: python scripts/analyze_sample_efficiency.py <downloaded score volume dir> [<training volume dir>] [--json out.json]
Reads only scorer reports and training receipts; never trains, scores or deserializes native checkpoints.
"""
import argparse
import json
from pathlib import Path
from statistics import median


def sustained_threshold(points, threshold):
    """First checkpoint at or above threshold whose next checkpoint is also at or above it (None if censored)."""
    for (steps, successes), (_, following) in zip(points, points[1:]):
        if successes >= threshold and following >= threshold:
            return steps
    return None


def auc(points, horizon):
    """Trapezoidal mean success fraction over [0, horizon] on observed checkpoints (128 episodes each)."""
    grid = [(s, k/128) for s, k in points if s <= horizon]
    if len(grid) < 2 or grid[-1][0] < horizon:
        return None
    area = sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(grid, grid[1:]))
    return area/horizon


def load(scores, training=None):
    trials = []
    for report_path in sorted(Path(scores).glob('trial-*/report.json'), key=lambda p: int(p.parent.name.split('-')[1])):
        report = json.loads(report_path.read_text())
        points = sorted((p['steps'], p['successes']) for p in report['points'])
        seconds = {p['steps']: p['trainer_seconds'] for p in report['points']}
        trial = dict(index=report['index'], arm=report['arm'], seed=report['seed'], state=report['state'],
                     training_state=report.get('training_state'), points=points, seconds=seconds)
        if training:
            receipt = Path(training)/f"trial-{report['index']}/receipt.json"
            if receipt.exists():
                r = json.loads(receipt.read_text())
                trial.update(transitions=r.get('actual_work', {}).get('environment_completed'),
                             optimizer_steps=r.get('actual_work', {}).get('optimizer_completed'),
                             trainer_process_seconds=r.get('trainer_process_seconds'))
        trials.append(trial)
    return trials


def summarize(trials, ceiling):
    horizon = min((t['points'][-1][0] for t in trials if t['points']), default=0)
    for t in trials:
        t['T80'] = sustained_threshold(t['points'], 103)
        t['T90'] = sustained_threshold(t['points'], 116)
        t['T80_seconds'] = t['seconds'].get(t['T80']) if t['T80'] else None
        t['auc_common'] = auc(t['points'], horizon)
        t['final'] = t['points'][-1] if t['points'] else None
    arms = {}
    for t in trials:
        arms.setdefault(t['arm'], []).append(t)
    summary = {}
    for arm, group in arms.items():
        censored = lambda v: v if v is not None else ceiling  # noqa: E731
        summary[arm] = dict(seeds=[t['seed'] for t in group],
                            T80=[t['T80'] for t in group], T90=[t['T90'] for t in group],
                            T80_reached=sum(t['T80'] is not None for t in group),
                            median_T80_censored_at_ceiling=median(censored(t['T80']) for t in group),
                            median_T90_censored_at_ceiling=median(censored(t['T90']) for t in group),
                            T80_seconds=[t['T80_seconds'] for t in group],
                            mean_auc_common=(sum(t['auc_common'] for t in group)/len(group)
                                             if all(t['auc_common'] is not None for t in group) else None),
                            finals=[t['final'] for t in group])
    return dict(common_horizon=horizon, ceiling=ceiling, arms=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scores')
    parser.add_argument('training', nargs='?')
    parser.add_argument('--ceiling', type=int, default=15_360_000)
    parser.add_argument('--json')
    args = parser.parse_args()
    trials = load(args.scores, args.training)
    summary = summarize(trials, args.ceiling)
    for t in trials:
        curve = ' '.join(f"{s/1e6:.0f}M:{k}" for s, k in t['points'])
        print(f"{t['arm']:>16} seed {t['seed']}: T80={t['T80']} T90={t['T90']} [{t['state']}/{t['training_state']}] {curve}")
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(summary=summary, trials=trials), indent=1, default=str)+'\n')


if __name__ == '__main__':
    main()
