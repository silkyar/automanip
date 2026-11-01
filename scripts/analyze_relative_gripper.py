"""Paired sampler-coordinate diagnosis; oracle coverage is not policy success."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.branching_validity import compare
from scripts.analyze_prefix_search import PHYSICAL, TRACE
from research.prefix_search import prefix_digest


def load(directory):
    report = json.loads((directory/'report.json').read_text())
    if report['state'] != 'complete' or report['transitions_completed'] != 3200 or report['state_write_calls'] != 0:
        raise ValueError('Invalid child: '+str(directory))
    with np.load(directory/'trace.npz', allow_pickle=False) as saved:
        trace = {k: saved[k] for k in TRACE}
    if prefix_digest(trace) != report['prefix_sha256']:
        raise ValueError('Prefix digest differs')
    return report, trace


def physical(trace, stop=None):
    return {k: value if stop is None else value[:stop-1 if k == 'flags' else stop]
            for k, value in trace.items() if k in PHYSICAL}


def analyze(root, absolute_root, plan, dataset_output=None):
    root, absolute_root = Path(root), Path(absolute_root)
    rows, datasets, work, seconds = [], {}, 0, []
    fidelity = True
    for index, parent in enumerate(plan['parents']):
        directory, old = root/f'collection-{index}', absolute_root/f'collection-{index}'
        _, base = load(old/'candidate-0-r0')
        absolute = [load(old/f'candidate-{c}-r0')[1] for c in range(1, 9)]
        outcomes = {'absolute': np.stack([t['flags'][-1, :, 6].astype(bool) for t in absolute])}
        checks, repeats, traces = [], [], {}
        opening = {mode: [] for mode in ('absolute', 'anchor', 'current')}
        for candidate, t in enumerate(absolute, 1):
            gap = .015+.025*t['executed_actions'][30, :, 7]-t['observations'][30, :, 7]
            opening['absolute'].extend(gap.tolist())
        for mode in ('anchor', 'current'):
            for candidate, repeat in [(c, 0) for c in range(1, 9)]+[(1, 1)]:
                report, trace = load(directory/f'{mode}-candidate-{candidate}-r{repeat}')
                if (report['collection_index'], report['candidate'], report['repeat'], report['mode']) != (index, candidate, repeat, mode):
                    raise ValueError('Child identity differs')
                traces[mode, candidate, repeat] = trace
                work += report['transitions_completed']; seconds.append(report['wall_seconds'])
                check = compare(physical(base, 31), physical(trace, 31))
                check.update(mode=mode, candidate=candidate, repeat=repeat,
                             prefix_actions_identical=bool(np.array_equal(base['executed_actions'][:30], trace['executed_actions'][:30])),
                             arm_actions_identical=bool(np.array_equal(absolute[candidate-1]['executed_actions'][:, :, :7], trace['executed_actions'][:, :, :7])))
                fidelity &= check['passed'] and check['prefix_actions_identical'] and check['arm_actions_identical']
                checks.append(check)
                if repeat == 0:
                    gap = .015+.025*trace['executed_actions'][30, :, 7]-trace['observations'][30, :, 7]
                    opening[mode].extend(gap.tolist())
            a, b = traces[mode, 1, 0], traces[mode, 1, 1]
            check = compare(physical(a), physical(b))
            check.update(mode=mode, action_tape_identical=bool(np.array_equal(a['executed_actions'], b['executed_actions'])))
            fidelity &= check['passed'] and check['action_tape_identical']; repeats.append(check)
            outcomes[mode] = np.stack([traces[mode, c, 0]['flags'][-1, :, 6].astype(bool) for c in range(1, 9)])
        first_checks = []
        for candidate in range(1, 9):
            a, b = traces['anchor', candidate, 0], traces['current', candidate, 0]
            check = compare(physical(a, 32), physical(b, 32))
            check.update(candidate=candidate, first_action_identical=bool(np.array_equal(a['executed_actions'][30], b['executed_actions'][30])))
            fidelity &= check['passed'] and check['first_action_identical']; first_checks.append(check)
        counts = {mode: int(y.any(0).sum()) for mode, y in outcomes.items()}
        choices = np.full(64, -1, dtype=np.int64)
        rng = np.random.default_rng(plan['tie_seed']+index)
        for lane in range(64):
            successful = np.flatnonzero(outcomes['current'][:, lane])+1
            if len(successful): choices[lane] = rng.choice(successful)
        obs, act, means, lanes = [], [], [], []
        for lane, candidate in enumerate(choices):
            if candidate < 0: continue
            t = traces['current', int(candidate), 0]
            obs.extend(t['observations'][30:50, lane]); act.extend(t['executed_actions'][30:50, lane])
            means.extend(t['parent_means'][30:50, lane]); lanes.extend([lane]*20)
        datasets[parent] = dict(teacher_observations=np.asarray(obs, dtype=np.float32).reshape(-1, 48),
                                teacher_actions=np.asarray(act, dtype=np.float32).reshape(-1, 8),
                                teacher_parent_means=np.asarray(means, dtype=np.float32).reshape(-1, 8),
                                teacher_episode=np.asarray(lanes, dtype=np.int64), selected_candidate=choices,
                                retain_observations=base['observations'][:30].transpose(1, 0, 2),
                                retain_parent_means=base['parent_means'][:30].transpose(1, 0, 2),
                                training_outcomes=outcomes['current'])
        rows.append(dict(parent=parent, covered_states=counts,
                         candidate_successes={mode: y.sum(1).tolist() for mode, y in outcomes.items()},
                         opening_fraction={mode: float((np.asarray(gap)>0).mean()) for mode, gap in opening.items()},
                         first_requested_gap_median={mode: float(np.median(gap)) for mode, gap in opening.items()},
                         prefixes=checks, repeated_continuations=repeats, first_anchor_current_matches=first_checks,
                         selected_candidates=choices.tolist()))
    if work != plan['max_transitions']:
        raise ValueError('Transition accounting differs')
    covered = {mode: sum(r['covered_states'][mode] for r in rows) for mode in ('absolute', 'anchor', 'current')}
    gain = (covered['current']-covered['absolute'])/128
    coverage = covered['current'] >= 64 and gain >= .15 and all(r['covered_states']['current'] >= r['covered_states']['absolute'] for r in rows)
    passed = bool(fidelity and coverage)
    if passed and dataset_output is not None:
        destination = Path(dataset_output); destination.mkdir(exist_ok=False)
        for parent, data in datasets.items(): np.savez_compressed(destination/f'parent-{parent}.npz', **data)
    return dict(format='automanip-relative-gripper-analysis-v1', state='complete',
                new_transitions=work, optimizer_calls=0, worker_seconds=seconds,
                fidelity_passed=bool(fidelity), coverage_passed=bool(coverage), screen_passed=passed,
                covered_source_states=covered, source_states=128, current_minus_absolute=gain, parents=rows,
                decision='Register matched student/control test' if passed else 'Withhold distillation; sufficient coordinate excitation did not establish useful coverage or fidelity failed',
                scope='Sampler-coordinate diagnosis on acquired policies; not a learned-policy benchmark or general learning principle')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--absolute-root', required=True)
    p.add_argument('--plan', required=True); p.add_argument('--output', required=True); p.add_argument('--dataset-output')
    a = p.parse_args(); r = analyze(a.root, a.absolute_root, json.loads(Path(a.plan).read_text()), a.dataset_output)
    Path(a.output).write_text(json.dumps(r, indent=2)+'\n')
    print(json.dumps({k: v for k, v in r.items() if k not in ('parents', 'worker_seconds')}, indent=2))
