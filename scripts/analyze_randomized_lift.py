"""All-lane randomized mechanical outcomes; never select dose responders."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def wilson(k, n):
    if not n: return None
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius, center+radius]


def analyze(root):
    report = json.loads((root/'report.json').read_bytes())
    path = root/'trajectories.npz'
    with np.load(path, allow_pickle=False) as data:
        initial, obs, nxt = (data[key] for key in ('initial', 'observations', 'next_observations'))
        grasp, eligible, assignment = (data[key] for key in ('grasp', 'eligibility_step', 'assignment'))
        actions = data['actions']
        rows = []
        for lane in range(128):
            start = int(eligible[lane])
            row = dict(lane=lane, group='ABCD'[int(assignment[lane])], public_seed=93304000+lane,
                       eligible=start >= 0, intervention_start_step=start,
                       completed_steps=len(nxt), initial_cubeA_position=initial[lane, 25:28].tolist())
            if len(nxt):
                row.update(ever_grasp=bool(grasp[:, lane].any()), final_grasp=bool(grasp[-1, lane]),
                           reset_relative_max_cube_rise_m=float((nxt[:, lane, 27]-initial[lane, 27]).max()),
                           reset_relative_3cm_lift=bool((nxt[:, lane, 27]-initial[lane, 27] >= .03).any()))
            if 0 <= start < len(nxt):
                after = nxt[start:, lane]
                before = obs[start, lane]
                cube = after[:, 27]-before[27]
                tcp = after[:, 20]-before[20]
                cube_hits, tcp_hits = np.flatnonzero(cube >= .03), np.flatnonzero(tcp >= .04)
                previous = np.r_[True, grasp[start:-1, lane]]
                row.update(cube_lift_3cm=bool(len(cube_hits)), delivered_tcp_dose_4cm=bool(len(tcp_hits)),
                           max_cube_rise_m=float(cube.max()), max_tcp_rise_m=float(tcp.max()),
                           first_cube_lift_elapsed_step=int(start+cube_hits[0]+1) if len(cube_hits) else None,
                           first_TCP_dose_elapsed_step=int(start+tcp_hits[0]+1) if len(tcp_hits) else None,
                           cube_rise_at_first_TCP_dose_m=float(cube[tcp_hits[0]]) if len(tcp_hits) else None,
                           grasp_at_first_TCP_dose=bool(grasp[start+tcp_hits[0], lane]) if len(tcp_hits) else None,
                           drop_elapsed_steps=(start+np.flatnonzero(previous & ~grasp[start:, lane])+1).tolist(),
                           remaining_exposure_steps=len(after),
                           final_finger_qpos=nxt[-1, lane, 7:9].tolist(),
                           mean_gripper_action_after_start=float(actions[start:, lane, 7].mean()),
                           final_TCP_z_target_error_m=float(before[20]+.06-after[-1, 20]))
            rows.append(row)
    result = dict(state=report['state'], kind='Randomized assisted mechanical screen, no learning claim',
                  raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), groups={},
                  interpretation='Incomplete or invalid experiment; no directional interpretation',
                  statistical_limits='Working-Bernoulli Wilson intervals, not guaranteed causal/randomization coverage; one policy/batch')
    for group in 'ABCD':
        assigned = [row for row in rows if row['group'] == group]
        selected = [row for row in assigned if row['eligible'] and 'cube_lift_3cm' in row]
        n = len(selected)
        lifts = sum(row['cube_lift_3cm'] for row in selected)
        doses = sum(row['delivered_tcp_dose_4cm'] for row in selected)
        both = sum(row['cube_lift_3cm'] and row['delivered_tcp_dose_4cm'] for row in selected)
        simultaneous = sum(row['cube_rise_at_first_TCP_dose_m'] is not None and
                           row['cube_rise_at_first_TCP_dose_m'] >= .03 and row['grasp_at_first_TCP_dose']
                           for row in selected)
        result['groups'][group] = dict(assigned=len(assigned), eligible=n, cube_lifts=lifts,
            delivered_TCP_doses=doses, both_maxima_lift_and_dose=both,
            cube_lift_and_grasp_at_first_TCP_dose=simultaneous,
            cube_lift_Wilson95=wilson(lifts, n), dose_Wilson95=wilson(doses, n),
            both_maxima_Wilson95=wilson(both, n),
            assigned_reset_relative_lifts=sum(row.get('reset_relative_3cm_lift', False) for row in assigned),
            eligibility_times=[row['intervention_start_step'] for row in selected],
            remaining_exposure_steps=[row['remaining_exposure_steps'] for row in selected], lanes=assigned)
    valid = report['state'] == 'complete' and report['live_FK_valid'] and report['actual_public_transitions'] == 6400
    result['valid'] = valid
    if valid and all(group['eligible'] >= 8 for group in result['groups'].values()):
        a, b, c, d = (result['groups'][name] for name in 'ABCD')
        if c['both_maxima_lift_and_dose'] >= .75*c['eligible'] and a['cube_lifts'] <= .25*a['eligible']:
            result['interpretation'] = 'Directional support for an immediate learned-command barrier in these eligible public states; inspect simultaneity and uncertainty'
        elif (d['both_maxima_lift_and_dose'] >= .75*d['eligible'] and c['cube_lifts'] <= .25*c['eligible']
              and c['delivered_TCP_doses'] >= .75*c['eligible']):
            result['interpretation'] = 'Directional support for closure under this upward controller; inspect simultaneity and uncertainty'
        elif d['delivered_TCP_doses'] >= .75*d['eligible'] and d['cube_lifts'] < .5*d['eligible']:
            result['interpretation'] = 'Upward dose delivered but cube mostly not lifted; load-bearing/contact geometry suspect under this intervention'
        elif d['delivered_TCP_doses'] < .75*d['eligible']:
            result['interpretation'] = 'Insufficient delivered dose; grip hypothesis unresolved'
        else:
            result['interpretation'] = 'Mixed outcomes; no directional gate supported'
        result['grip_only_policy_response_screen'] = b['cube_lifts'] >= .75*b['eligible'] and a['cube_lifts'] <= .25*a['eligible']
    elif valid:
        result['interpretation'] = 'Fewer than eight eligible lanes in at least one group; no directional gate passed'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'], interpretation=result['interpretation'],
        groups={key: {name: value for name, value in group.items() if name not in ('lanes', 'eligibility_times', 'remaining_exposure_steps')}
                for key, group in result['groups'].items()})))
