"""Fresh-process controls for no-op state writing; no simulation execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.branching_validity import compare


def analyze(root, anchor=15):
    root = Path(root)
    traces, reports = {}, []
    for condition in ('no_write', 'physical', 'augmented'):
        for repeat in range(2):
            name = condition+'-'+str(repeat)
            report = json.loads((root/name/'report.json').read_text())
            if report['state'] != 'complete' or report['transitions_completed'] != 3200:
                raise ValueError('Incomplete child: '+name)
            if report['condition'] != condition or report['repeat'] != repeat:
                raise ValueError('Condition identity differs')
            reports.append(report)
            with np.load(root/name/'trace.npz', allow_pickle=False) as saved:
                traces[name] = {key: saved[key] for key in saved.files}
    def portion(trace, start, stop):
        return {key: value[start:stop if key != 'flags' else stop-1]
                for key, value in trace.items()}
    controls = compare(traces['no_write-0'], traces['no_write-1'])
    rows = []
    for condition in ('physical', 'augmented'):
        for repeat in range(2):
            for control_repeat in range(2):
                ref = traces['no_write-'+str(control_repeat)]
                replay = traces[condition+'-'+str(repeat)]
                prefix = compare(portion(ref, 0, anchor+1), portion(replay, 0, anchor+1))
                first = compare(portion(ref, anchor, anchor+2), portion(replay, anchor, anchor+2))
                whole = compare(ref, replay)
                rows.append(dict(condition=condition, repeat=repeat, control_repeat=control_repeat,
                                 prefix=prefix, first_action_after_write=first, full_trajectory=whole))
    prefixes_match = all(r['prefix']['passed'] for r in rows)
    if not prefixes_match:
        conclusion = 'Inconclusive causal attribution: conditions already differ before state writing'
    elif not controls['passed']:
        conclusion = 'Inconclusive causal attribution: capture-only controls themselves fail replay tolerances'
    elif all(not r['first_action_after_write']['passed'] for r in rows):
        conclusion = 'No-op state writing perturbed next-action dynamics in both tested restoration variants; hidden cause unresolved'
    elif all(r['full_trajectory']['passed'] for r in rows):
        conclusion = 'No-op writing passed here; rewind/history mismatch remains to be investigated'
    else:
        conclusion = 'Mixed no-op effects; report individual contrasts without a general cloning conclusion'
    return dict(format='automanip-branching-noop-analysis-v1', state='complete',
                new_transitions=sum(r['transitions_completed'] for r in reports), optimizer_calls=0,
                worker_seconds=[r['wall_seconds'] for r in reports],
                no_write_controls=controls, prefixes_match=prefixes_match, comparisons=rows,
                conclusion=conclusion, benchmark_comparison=False,
                scope='One acquired parent action tape,64fixed initial states,two fresh-process repetitions; no learned-policy result')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); result = analyze(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}, indent=2))
