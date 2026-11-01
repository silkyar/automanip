"""Summarize the registered assisted probe, retaining every public lane."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(root):
    report = json.loads((root/'report.json').read_bytes())
    result = dict(kind='Assisted mechanical probe; no learning efficiency estimate',
                  state=report['state'], eligible_lanes=report.get('eligible_lanes', 0),
                  validity_passed=report.get('validity_passed', False), conditions={},
                  interpretation='No paired causal conclusion: incomplete or invalid probe')
    for condition in ('A', 'B', 'C', 'D'):
        path = root/(condition+'.npz')
        if not path.exists():
            result['conditions'][condition] = dict(missing=True)
            continue
        with np.load(path, allow_pickle=False) as arrays:
            eligibility = arrays['eligibility_step']
            initial = arrays['initial']
            observations = arrays['observations']
            successors = arrays['next_observations']
            grasp = arrays['grasp']
            actions = arrays['actions']
            lanes = []
            for lane in range(8):
                start = int(eligibility[lane])
                entry = dict(lane=lane, public_seed=93284000+lane,
                             eligible=start >= 0, intervention_start_step=start,
                             completed_steps=len(successors))
                if len(successors):
                    entry.update(ever_grasp=bool(grasp[:, lane].any()), final_grasp=bool(grasp[-1, lane]),
                                 reset_relative_max_cube_rise_m=float((successors[:, lane, 27]-initial[lane, 27]).max()),
                                 reset_relative_ever_3cm_lift=bool((successors[:, lane, 27]-initial[lane, 27] >= .03).any()))
                if 0 <= start < len(successors):
                    after = successors[start:, lane]
                    before = observations[start, lane]
                    cube_rise = float((after[:, 27]-before[27]).max())
                    tcp_rise = float((after[:, 20]-before[20]).max())
                    previous_grasp = np.r_[True, grasp[start:-1, lane]]
                    entry.update(max_cube_rise_after_intervention_m=cube_rise,
                                 max_tcp_rise_after_intervention_m=tcp_rise,
                                 cube_lift_3cm=cube_rise >= .03,
                                 delivered_tcp_dose_4cm=tcp_rise >= .04,
                                 drops_after_intervention=int((previous_grasp & ~grasp[start:, lane]).sum()),
                                 mean_gripper_action_after_intervention=float(actions[start:, lane, 7].mean()),
                                 final_finger_qpos=successors[-1, lane, 7:9].tolist(),
                                 final_tcp_z_target_error_m=float(before[20]+.06-after[-1, 20]))
                lanes.append(entry)
            eligible = [lane for lane in lanes if lane['eligible'] and 'cube_lift_3cm' in lane]
            result['conditions'][condition] = dict(
                data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), lanes=lanes,
                eligible_completed=len(eligible), cube_lifts=sum(lane['cube_lift_3cm'] for lane in eligible),
                delivered_tcp_doses=sum(lane['delivered_tcp_dose_4cm'] for lane in eligible),
                both_lift_and_dose=sum(lane['cube_lift_3cm'] and lane['delivered_tcp_dose_4cm'] for lane in eligible))
    n = result['eligible_lanes']
    if result['state'] == 'complete' and result['validity_passed'] and n >= 4:
        a, c, d = (result['conditions'][name] for name in ('A', 'C', 'D'))
        # Conservative reading of 'while A does not': require no A lifts.
        if c['both_lift_and_dose'] >= .75*n and a['cube_lifts'] == 0:
            result['interpretation'] = 'Supports immediate learned-command barrier in these selected public states'
        elif d['both_lift_and_dose'] >= .75*n and c['both_lift_and_dose'] < .75*n:
            result['interpretation'] = 'Closing the gripper changes outcome under the same upward controller; examine C dose before attributing grip failure'
        elif d['delivered_tcp_doses'] >= .75*n and d['cube_lifts'] < .5*n:
            result['interpretation'] = 'Upward dose delivered but cube mostly not lifted; load-bearing grip/contact geometry suspect under this intervention'
        elif d['delivered_tcp_doses'] < .75*n:
            result['interpretation'] = 'Insufficient delivered upward dose; grip hypothesis unresolved'
        else:
            result['interpretation'] = 'Mixed mechanical outcomes; no registered directional hypothesis supported'
    elif result['state'] == 'complete' and result['validity_passed']:
        result['interpretation'] = 'Fewer than four eligible lanes; retain outcomes without directional causal claim'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('state', 'eligible_lanes', 'validity_passed', 'interpretation')}))
