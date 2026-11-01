"""Training-only action-search coverage and prefix fidelity; never a policy score."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.branching_validity import compare
from research.prefix_search import ANCHOR, candidate_kind, prefix_digest


PHYSICAL = ('position', 'quaternion', 'qpos', 'qvel', 'linear_velocity', 'angular_velocity', 'flags')
TRACE = (*PHYSICAL, 'observations', 'executed_actions', 'parent_means')


def analyze(root, plan, dataset_output=None):
    root = Path(root)
    rows, datasets, transitions, worker_seconds = [], {}, 0, []
    fidelity = True
    for index, parent in enumerate(plan['parents']):
        directory = root/('collection-'+str(index))
        traces, reports = {}, {}
        for candidate, repeat in [(c, 0) for c in range(16)]+[(0, 1), (1, 1)]:
            child = directory/f'candidate-{candidate}-r{repeat}'
            report = json.loads((child/'report.json').read_text())
            if report['state'] != 'complete' or report['transitions_completed'] != 3200 or report['state_write_calls'] != 0:
                raise ValueError('Incomplete or invalid candidate: '+str(child))
            if (report['collection_index'], report['candidate'], report['repeat']) != (index, candidate, repeat):
                raise ValueError('Candidate identity differs')
            with np.load(child/'trace.npz', allow_pickle=False) as saved:
                trace = {k: saved[k] for k in TRACE}
            if prefix_digest(trace) != report['prefix_sha256']:
                raise ValueError('Prefix digest differs')
            if trace['observations'].shape != (51, 64, 48) or trace['executed_actions'].shape != (50, 64, 8) or trace['flags'].shape != (50, 64, 7):
                raise ValueError('Trace interface differs')
            traces[candidate, repeat], reports[candidate, repeat] = trace, report
            transitions += report['transitions_completed']; worker_seconds.append(report['wall_seconds'])
        base = traces[0, 0]
        prefix_checks, repeat_checks = [], []
        def physical(trace, prefix=False):
            return {k: value[:ANCHOR if k == 'flags' else ANCHOR+1] if prefix else value
                    for k, value in trace.items() if k in PHYSICAL}
        for (candidate, repeat), trace in traces.items():
            check = compare(physical(base, True), physical(trace, True))
            check.update(candidate=candidate, repeat=repeat,
                         prefix_actions_identical=bool(np.array_equal(base['executed_actions'][:ANCHOR], trace['executed_actions'][:ANCHOR])),
                         exact_prefix_hash=reports[0, 0]['prefix_sha256'] == reports[candidate, repeat]['prefix_sha256'])
            fidelity &= check['passed'] and check['prefix_actions_identical']
            prefix_checks.append(check)
        for candidate in (0, 1):
            a, b = traces[candidate, 0], traces[candidate, 1]
            check = compare(physical(a), physical(b))
            check.update(candidate=candidate, action_tape_identical=bool(np.array_equal(a['executed_actions'], b['executed_actions'])))
            fidelity &= check['passed'] and check['action_tape_identical']
            repeat_checks.append(check)
        outcomes = np.stack([traces[c, 0]['flags'][-1, :, 6].astype(bool) for c in range(16)])
        choices = np.full(64, -1, dtype=np.int64)
        rng = np.random.default_rng(plan['tie_seed']+index)
        for lane in range(64):
            successful = np.flatnonzero(outcomes[:, lane])
            if len(successful): choices[lane] = 0 if outcomes[0, lane] else rng.choice(successful)
        random_candidates = np.random.default_rng(plan['random_control_seed']+index).integers(1, 16, size=64)
        oracle = choices >= 0
        quiet = outcomes[1]
        row = dict(parent=parent, source_episodes=64, prefix_checks=prefix_checks,
                   repeat_checks=repeat_checks, candidate_success_counts=outcomes.sum(1).tolist(),
                   candidate_kinds=[candidate_kind(c) for c in range(16)],
                   base_successes=int(outcomes[0].sum()), fixed_quiet_successes=int(quiet.sum()),
                   random_proposal_successes=int(outcomes[random_candidates, np.arange(64)].sum()),
                   oracle_covered_states=int(oracle.sum()), oracle_minus_quiet=float(oracle.mean()-quiet.mean()),
                   selected_candidates=choices.tolist(), random_candidates=random_candidates.tolist(),
                   labels_per_successful_episode=20,
                   coverage_by_candidate_prefix=outcomes.cumsum(0).astype(bool).sum(1).tolist(),
                   quiet_bank_covered_states=int(outcomes[1:9].any(0).sum()),
                   policy_offset_bank_covered_states=int(outcomes[9:].any(0).sum()))
        rows.append(row)
        teacher_obs, teacher_actions, teacher_parent, lanes, steps = [], [], [], [], []
        for lane, candidate in enumerate(choices):
            if candidate < 0: continue
            trace = traces[int(candidate), 0]
            teacher_obs.extend(trace['observations'][ANCHOR:50, lane])
            teacher_actions.extend(trace['executed_actions'][ANCHOR:50, lane])
            teacher_parent.extend(trace['parent_means'][ANCHOR:50, lane])
            lanes.extend([lane]*20); steps.extend(range(ANCHOR, 50))
        datasets[parent] = dict(teacher_observations=np.asarray(teacher_obs, dtype=np.float32).reshape(-1, 48),
                                teacher_actions=np.asarray(teacher_actions, dtype=np.float32).reshape(-1, 8),
                                teacher_parent_means=np.asarray(teacher_parent, dtype=np.float32).reshape(-1, 8),
                                teacher_episode=np.asarray(lanes, dtype=np.int64), teacher_step=np.asarray(steps, dtype=np.int64),
                                retain_observations=base['observations'][:ANCHOR].transpose(1, 0, 2).copy(),
                                retain_parent_means=base['parent_means'][:ANCHOR].transpose(1, 0, 2).copy(),
                                selected_candidate=choices, training_outcomes=outcomes)
    if transitions != plan['max_transitions']:
        raise ValueError('Total transition accounting differs')
    covered = sum(r['oracle_covered_states'] for r in rows)
    gain = sum(r['oracle_covered_states']-r['fixed_quiet_successes'] for r in rows)/128
    coverage = covered >= 64 and gain >= .15 and all(r['oracle_minus_quiet'] >= 0 for r in rows)
    passed = bool(fidelity and coverage)
    if passed and dataset_output is not None:
        destination = Path(dataset_output); destination.mkdir(exist_ok=False)
        for parent, data in datasets.items():
            np.savez_compressed(destination/f'parent-{parent}.npz', **data)
    return dict(format='automanip-prefix-search-analysis-v1', state='complete',
                new_transitions=transitions, worker_seconds=worker_seconds, optimizer_calls=0,
                fidelity_passed=bool(fidelity), coverage_passed=bool(coverage), screen_passed=passed,
                covered_source_states=covered, source_states=128, pooled_oracle_minus_fixed_quiet=gain,
                parents=rows, scope='Training-only oracle label coverage on two acquired parents; not a deployable-policy benchmark or sample-efficiency result',
                decision='Register student and matched ordinary-rollout control' if passed else 'Withhold distillation; diagnose failed fidelity or sparse label coverage without a proposal-scale sweep')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True); p.add_argument('--dataset-output')
    a = p.parse_args(); result = analyze(a.root, json.loads(Path(a.plan).read_text()), a.dataset_output)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('parents', 'worker_seconds')}, indent=2))
