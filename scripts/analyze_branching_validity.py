"""Analyze preserved replay traces; never launches or advances a simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.branching_validity import TOLERANCES, compare, quaternion_error


def analyze(root):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    if report['state'] != 'complete':
        raise ValueError('Incomplete diagnostic; preserve failure report instead')
    rows = []
    for row in report['comparisons']:
        seed, mode, anchor, repeat = (row[k] for k in ('parent', 'mode', 'anchor', 'repeat'))
        with np.load(root/f'parent-{seed}-reference.npz', allow_pickle=False) as saved:
            reference = {k: saved[k][anchor:] for k in (*TOLERANCES.keys(), 'flags') if k != 'rotation'}
            reference['quaternion'] = saved['quaternion'][anchor:]
        with np.load(root/f'parent-{seed}-{mode}-a{anchor}-r{repeat}.npz', allow_pickle=False) as saved:
            replay = {k: saved[k] for k in saved.files}
        checked = compare(reference, replay)
        for key in checked:
            # Recomputing Linux/NumPy reductions on macOS may differ by an ulp.
            # Outcomes and the preregistered pass/fail decision must match exactly.
            equal = (all(np.isclose(value, row[key][name], rtol=0, atol=1e-12)
                         for name, value in checked[key].items()) if key == 'maximum_errors'
                     else checked[key] == row[key])
            if not equal:
                raise ValueError('Analysis disagrees with worker: '+key)
        physical_failures = np.zeros(reference['qpos'].shape[:2], dtype=bool)
        for key, limit in TOLERANCES.items():
            if key == 'rotation':
                error = quaternion_error(reference['quaternion'], replay['quaternion']).max(axis=-1)
            else:
                error = np.abs(reference[key]-replay[key])
                error = error.reshape(*error.shape[:2], -1).max(axis=-1)
            physical_failures |= error > limit
        first = np.where(physical_failures.any(axis=0), physical_failures.argmax(axis=0), -1)
        initial = {key: float(np.abs(reference[key][0]-replay[key][0]).max())
                   for key in TOLERANCES if key != 'rotation'}
        initial['rotation'] = float(quaternion_error(reference['quaternion'][0], replay['quaternion'][0]).max())
        rows.append({**row, 'initial_errors': initial,
                     'first_physical_failure_per_lane': first.tolist(),
                     'physical_failed_lanes': int((first >= 0).sum())})
    summary = {}
    for mode in ('physical', 'augmented'):
        selected = [r for r in rows if r['mode'] == mode]
        first = np.asarray([v for r in selected for v in r['first_physical_failure_per_lane']])
        summary[mode] = dict(
            replay_batches=len(selected), replay_lanes=len(first),
            passed_batches=sum(r['passed'] for r in selected),
            physical_failed_lanes=int((first >= 0).sum()),
            initial_failed_lanes=int((first == 0).sum()),
            median_first_failure_after_anchor=float(np.median(first[first >= 0])) if (first >= 0).any() else None,
            final_success_changed_lanes=sum(r['final_success_changed_lanes'] for r in selected),
            grasp_or_success_changed_lanes=sum(r['grasp_or_success_changed_lanes'] for r in selected),
            maximum_errors={key: max(r['maximum_errors'][key] for r in selected) for key in TOLERANCES},
            maximum_initial_errors={key: max(r['initial_errors'][key] for r in selected) for key in TOLERANCES},
        )
    return dict(format='automanip-branching-validity-analysis-v1', state='complete',
                gate_passed=report['gate_passed'], summaries=summary, comparisons=rows,
                new_transitions=report['transitions_completed'], optimizer_calls=0,
                worker_seconds=report['wall_seconds'], decision=report['decision'],
                scope='Exposed same-lane restoration on fixed parents only; no learned policy or sample-efficiency result')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = analyze(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}, indent=2))
