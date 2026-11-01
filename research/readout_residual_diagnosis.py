"""Analytic diagnostics only; deliberately no export or simulator entry point."""
import json
from pathlib import Path
import time

import numpy as np

from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from automanip.stackcube import STACK_TASK
from research.readout_distill import features, RIDGE
from research.relative_gripper_search import inverse_target


def diagnose():
    started = time.monotonic(); rows = []
    for parent in (2, 3):
        source = Path(f'artifacts/cumulative/relative-gripper-20261031/student-inputs/parent-{parent}.npz')
        fit_root = Path(f'.runtime/readout-students-20261031/search-{parent}')
        with np.load(source, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
        with np.load(fit_root/'fit-audit.npz', allow_pickle=False) as z: audit = {k: z[k] for k in z.files}
        artifact = load_policy(f'.runtime/relative-gripper-20261031/inputs/policies/{parent}', PolicySchema(STACK_TASK.fingerprint, 48, 8))
        actor = NumpyPolicy(artifact, PolicySchema(STACK_TASK.fingerprint, 48, 8))
        w, b = actor.layers[-1]; original = np.column_stack([w, b]).T
        t_obs = data['teacher_observations'][audit['teacher_row_indices']]
        target = data['teacher_actions'][audit['teacher_row_indices']]
        r_obs = data['retain_observations'][audit['fitting_episode_indices']].reshape(-1, 48)[audit['rehearsal_row_indices']]
        mask = np.isin(data['teacher_episode'], audit['validation_episode_indices'])
        v_obs, v_target = data['teacher_observations'][mask], data['teacher_actions'][mask]
        vr_obs = data['retain_observations'][audit['validation_episode_indices']].reshape(-1, 48)
        t, r, v, vr = [features(actor, x) for x in (t_obs, r_obs, v_obs, vr_obs)]
        delta_teacher_only = np.linalg.solve(t.T@t/len(t)+RIDGE*np.eye(t.shape[1]), t.T@(target-t@original)/len(t))
        combined = np.vstack([t, r]); combined_obs = np.vstack([t_obs, r_obs])
        hold = np.linalg.solve(combined.T@combined/len(combined)+RIDGE*np.eye(t.shape[1]),
                               combined.T@inverse_target(combined_obs[:, 7])/len(combined))
        teach_gap = .015+.025*v_target[:, 7]-v_obs[:, 7]
        predictions = {}
        for name, delta in [('registered_rehearsal', audit['delta']), ('teacher_only_diagnostic', delta_teacher_only)]:
            raw = v@(original+delta); action = np.clip(raw, -1, 1)
            gap = .015+.025*action[:, 7]-v_obs[:, 7]
            per_axis = np.square(action-v_target).mean(0)
            predictions[name] = dict(heldout_action_mse_per_axis=per_axis.tolist(),
                heldout_finger_target_RMSE_metres=float(.025*np.sqrt(per_axis[7])),
                heldout_positive_finger_increment_fraction=float((gap>1e-6).mean()),
                heldout_median_finger_increment_metres=float(np.median(gap)),
                early_clipped_action_RMS_change=float(np.sqrt(np.square(np.clip(vr@(original+delta), -1, 1)-np.clip(vr@original, -1, 1)).mean())))
        rows.append(dict(parent=parent, heldout_teacher_rows=len(v), heldout_early_rows=len(vr),
                         feature_rows=len(t)+len(r)+len(v)+len(vr),
                         teacher_positive_increment_fraction=float((teach_gap>1e-6).mean()),
                         teacher_median_finger_increment_metres=float(np.median(teach_gap)),
                         hold_coordinate_RMSE_metres=float(.025*np.sqrt(np.square(v@hold-inverse_target(v_obs[:, 7])).mean())),
                         predictions=predictions))
    result = dict(format='automanip-readout-residual-diagnosis-results-v1',state='complete',parents=rows,
                  new_physics=0,linear_solves=4,exported_candidates=0,CPU_seconds=time.monotonic()-started,
                  scope='Offline analytic readout/actuator-error diagnosis only; no new robot policy result or optimized hyperparameter')
    Path('cumulative/results/readout-residual-diagnosis-20261031.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__': print(json.dumps(diagnose(),indent=2))
