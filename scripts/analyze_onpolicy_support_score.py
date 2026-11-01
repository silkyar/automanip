"""Fixed data-support comparison on fresh, paired public rollout blocks."""
import argparse
import json
from pathlib import Path

import numpy as np


ROLES = dict(baseline='base', original_states='rehearsal', student_states='teacher_only')
STAGES = ('grasp', 'lift', 'transport', 'placement', 'release', 'stable_stack_10_steps')


def contrast(left, right):
    gain = (np.asarray(left)-np.asarray(right))/64
    if gain.shape != (2, 4):
        raise ValueError('Expected two parents and four paired blocks')
    indices = np.random.default_rng(96809999).integers(0, 4, size=(10000, 2, 4))
    draws = np.take_along_axis(np.broadcast_to(gain, (10000, 2, 4)), indices, axis=2).mean((1, 2))
    return dict(mean_gain=float(gain.mean()), interval_97_5=np.quantile(draws, [.0125, .9875]).tolist())


def decision(vs_base, vs_original, retained):
    return bool(retained and vs_base['mean_gain'] >= .10 and vs_original['mean_gain'] >= .05
                and min(vs_base['interval_97_5'][0], vs_original['interval_97_5'][0]) > 0)


def analyze(root):
    root = Path(root)
    counts, policies, seconds, work = {}, {}, [], 0
    for name, role in ROLES.items():
        counts[name], policies[name] = [], []
        for index in range(2):
            r = json.loads((root/f'collection-{index}'/role/'report.json').read_text())
            if (r['state'] != 'complete' or r['environment_completed'] != 12800
                    or r['episode_count'] != 256 or len(r['episodes']) != 256
                    or r['method'] != role or r['collection_index'] != index
                    or len(r['batches']) != 4):
                raise ValueError('Incomplete or mislabeled public assessment')
            actual_blocks, grasp_blocks = [], []
            for batch in range(4):
                episodes = r['episodes'][batch*64:(batch+1)*64]
                if any(e['outcome']['failed'] or e['steps'] != 50 for e in episodes):
                    raise ValueError('Execution failure or transition accounting differs')
                successes = sum(e['outcome']['terminal_success'] and
                                (e['outcome']['terminated'] or e['outcome']['truncated']) for e in episodes)
                grasps = sum(e['diagnostics']['ever']['grasp'] for e in episodes)
                b = r['batches'][batch]
                if b['successes'] != successes or b['ever_grasp'] != grasps or b['failures']:
                    raise ValueError('Batch and episode outcomes disagree')
                actual_blocks.append(successes)
                grasp_blocks.append(grasps)
            if sum(actual_blocks) != r['successes'] or sum(grasp_blocks) != r['ever_grasp']:
                raise ValueError('Summary and episode outcomes disagree')
            counts[name].append(actual_blocks)
            policies[name].append(dict(parent=index+2, episodes=256, successes=r['successes'],
                ever_grasp=r['ever_grasp'], weights_sha256=r['weights_sha256'],
                stages={k: sum(e['diagnostics']['ever'][k] for e in r['episodes']) for k in STAGES}))
            work += r['environment_completed']
            seconds.append(r['wall_seconds'])
    vs_base = contrast(counts['student_states'], counts['baseline'])
    vs_original = contrast(counts['student_states'], counts['original_states'])
    retained = all(p['ever_grasp']/256 >= .95 for p in policies['student_states'])
    passed = decision(vs_base, vs_original, retained)
    return dict(format='automanip-onpolicy-support-score-v1', state='complete',
        role_map=ROLES, new_public_transitions=work, optimizer_calls=0,
        worker_seconds=seconds, policies=policies,
        pooled_success={m: float(np.asarray(v).sum()/512) for m, v in counts.items()},
        student_minus_baseline=vs_base, student_minus_original_states=vs_original,
        ever_grasp_retained=retained, primary_screen_passed=passed,
        decision='Replicate fixed assisted mechanism before promotion' if passed else
                 'Close bounded pilot: fixed student-state aggregation failed material screen; no score-driven refit',
        scope='One fixed aggregation step on two acquired actors; not full DAgger or from-scratch learning')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('policies', 'worker_seconds')}, indent=2))
