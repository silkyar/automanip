"""Prespecified two-run development screen; no evaluator or training execution.

Input points have steps, successes, episodes and optional trainer_seconds.
Only exact common measured checkpoints enter AUC; crossings are brackets, not
interpolated observations. The single-seed gate selects follow-up work only.
"""
import argparse
import json
import math
from pathlib import Path


def validate_curve(points):
    if not isinstance(points, list) or not points:
        raise ValueError('nonempty measured curve required')
    previous = -1
    for point in points:
        step, successes, episodes = (point[k] for k in ('steps', 'successes', 'episodes'))
        if type(step) is not int or step <= previous:
            raise ValueError('steps must be strictly increasing nonnegative integers')
        if type(episodes) is not int or episodes != 128:
            raise ValueError('screen requires 128 development episodes per point')
        if type(successes) is not int or not 0 <= successes <= episodes:
            raise ValueError('invalid success count')
        seconds = point.get('trainer_seconds')
        if seconds is not None and (type(seconds) not in (int, float)
                                    or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('invalid trainer time')
        previous = step
    return points


def crossing(points, threshold):
    """First sampled crossing, and whether all later samples stay above it."""
    validate_curve(points)
    for i, point in enumerate(points):
        if point['successes'] / point['episodes'] >= threshold:
            return {
                'reached_at_sample': True,
                'previous_sample_steps': points[i - 1]['steps'] if i else None,
                'first_passing_sample_steps': point['steps'],
                'trainer_seconds_at_sample': point.get('trainer_seconds'),
                'all_later_samples_pass': all(p['successes'] / p['episodes'] >= threshold
                                             for p in points[i:]),
                'continuous_crossing_measured': False,
            }
    return {'reached_at_sample': False, 'not_reached_by_steps': points[-1]['steps'],
            'continuous_crossing_measured': False}


def normalized_auc(points):
    validate_curve(points)
    if len(points) < 2:
        raise ValueError('AUC needs at least two distinct measured steps')
    area = sum((right['steps'] - left['steps'])
               * (left['successes'] / left['episodes'] + right['successes'] / right['episodes']) / 2
               for left, right in zip(points, points[1:]))
    return area / (points[-1]['steps'] - points[0]['steps'])


def summarize_pair(control, candidate):
    validate_curve(control)
    validate_curve(candidate)
    control_by_step = {p['steps']: p for p in control}
    candidate_by_step = {p['steps']: p for p in candidate}
    common_steps = sorted(control_by_step.keys() & candidate_by_step.keys())
    if len(common_steps) < 2:
        raise ValueError('need at least two exact common checkpoint steps')
    common_control = [control_by_step[s] for s in common_steps]
    common_candidate = [candidate_by_step[s] for s in common_steps]
    control_auc, candidate_auc = map(normalized_auc, (common_control, common_candidate))
    auc_gain = candidate_auc - control_auc
    final_steps_match = control[-1]['steps'] == candidate[-1]['steps']
    final_success_difference = candidate[-1]['successes'] - control[-1]['successes']
    final_guard = final_steps_match and final_success_difference >= -3
    control_85, candidate_85 = map(lambda p: crossing(p, .85), (common_control, common_candidate))
    # A right-censored control supplies no measured passing checkpoint. Do not
    # invent one at a later step in order to pass the earlier-crossing gate.
    earlier_85 = (control_85['reached_at_sample'] and candidate_85['reached_at_sample']
                  and candidate_85['first_passing_sample_steps']
                  < control_85['first_passing_sample_steps'])
    follow_up = final_guard and (auc_gain >= .03 - 1e-12 or earlier_85)
    return {
        'common_measured_steps': common_steps,
        'auc_domain_steps': [common_steps[0], common_steps[-1]],
        'auc_integration': 'trapezoidal over exact common measured points; normalized by domain length',
        'control_normalized_auc': control_auc,
        'candidate_normalized_auc': candidate_auc,
        'candidate_auc_gain': auc_gain,
        'final_steps_match': final_steps_match,
        'candidate_minus_control_final_successes': final_success_difference,
        'final_guard_passed': final_guard,
        'candidate_85_at_least_one_common_checkpoint_earlier': earlier_85,
        'follow_up_training_seeds_warranted': bool(follow_up),
        'control_thresholds': {str(t): crossing(control, t) for t in (.8, .85, .9)},
        'candidate_thresholds': {str(t): crossing(candidate, t) for t in (.8, .85, .9)},
        'claim': 'single-seed development screen, not confirmation or a reliable sample-efficiency result',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('curves', type=Path, help='JSON with control and candidate point lists')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    curves = json.loads(args.curves.read_text())
    result = summarize_pair(curves['control'], curves['candidate'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
