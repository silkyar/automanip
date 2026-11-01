"""Conditional entropy pressures plus an analytic counterexample; no fitting."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/discovery'))
from sac_baseline import Actor


def run():
    start = time.monotonic()
    torch.set_num_threads(1)
    base = ROOT/'runs/temperature-sufficiency-20261028/run'
    paths = {'state': base/'actor-step-256000-state.npz', 'trajectory': base/'actor-step-256000-trajectory.npz'}
    expected = {'state': 'ed55d938dcf21a5e80a1268921303ebed23d67dbb556b761cdd736f761f4ae6f',
                'trajectory': 'fdd401f214a41f8f7dcd26df6c7d4797b0dd0e3be3ce7ab8fc6750cf123d6332'}
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]: raise ValueError('Fixed input differs')
    actor = Actor().double().eval().requires_grad_(False)
    with np.load(paths['state'], allow_pickle=False) as saved:
        actor.load_state_dict({name: torch.from_numpy(saved['actor.'+name].copy()).double() for name in actor.state_dict()})
    with np.load(paths['trajectory'], allow_pickle=False) as saved:
        obs = saved['next_obs'][saved['stage_flags'][..., 0].astype(bool)].astype(np.float64)
    if obs.shape != (1904, 48): raise ValueError('Declared input subset differs')
    with torch.no_grad(): mean, logstd = [x.numpy() for x in actor(torch.from_numpy(obs))]
    answers = []
    for order in (32, 64):
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        latent = mean[None]+math.sqrt(2)*np.exp(logstd)[None]*nodes[:, None, None]
        log_jacobian = 2*(math.log(2)-latent-np.logaddexp(0, -2*latent))
        expected_log_density = -logstd-.5*math.log(2*math.pi*math.e)-(log_jacobian*weights[:, None, None]).sum(0)/math.sqrt(math.pi)
        answers.append(expected_log_density)
    difference = float(np.abs(answers[0]-answers[1]).max())
    coordinate_signal = answers[1]-1
    scalar_signal = coordinate_signal.sum(1)
    convergence_passed = difference <= 1e-4
    rows = []
    for index in range(8):
        fraction = float(((coordinate_signal[:, index] > 0) & (scalar_signal < 0)).mean())
        rows.append(dict(index=index, expected_log_density_mean=float(answers[1][:, index].mean()),
                         coordinate_signal_mean=float(coordinate_signal[:, index].mean()),
                         scalar_decrease_coordinate_increase_fraction=fraction,
                         blind_spot_gate=bool(convergence_passed and fraction >= .75 and
                                              coordinate_signal[:, index].mean() > 0 and scalar_signal.mean() < 0)))
    curvature = np.array([100., 1.])
    joint_variance_product = 1.
    optimal_variances = np.sqrt(joint_variance_product*np.prod(curvature))/curvature
    equal_variances = np.ones(2)*math.sqrt(joint_variance_product)
    def entropy(variance): return float(.5*np.log(2*math.pi*math.e*variance).sum())
    optimal_cost = float(.5*(curvature*optimal_variances).sum())
    equal_cost = float(.5*(curvature*equal_variances).sum())
    ratio = equal_cost/optimal_cost
    assert abs(entropy(optimal_variances)-entropy(equal_variances)) < 1e-12
    assert np.max(np.abs(curvature*optimal_variances-np.sqrt(joint_variance_product*np.prod(curvature)))) < 1e-12
    assert abs(ratio-5.05) < 1e-12
    result = dict(state='complete', grasped_saved_state_rows=len(obs), actor_forward_rows=len(obs),
        quadrature_orders=[32, 64], quadrature_log_Jacobian_evaluations=int(len(obs)*8*96),
        quadrature_max_abs_difference=difference, quadrature_gate_passed=convergence_passed,
        scalar_signal_mean=float(scalar_signal.mean()), scalar_decrease_state_fraction=float((scalar_signal < 0).mean()),
        coordinates=rows, negative_control=dict(kind='Unsquashed exact quadratic, not a robot model',
            curvatures=curvature.tolist(), optimal_variances=optimal_variances.tolist(), equal_variances=equal_variances.tolist(),
            common_joint_entropy=entropy(equal_variances), optimal_expected_cost=optimal_cost,
            equal_expected_cost=equal_cost, cost_ratio=ratio, gate_for_general_equal_coordinate_remedy=False),
        optimizer_calls=0, new_simulator_transitions=0, protected_evaluations=0,
        elapsed_seconds=time.monotonic()-start, input_sha256=expected,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Selected current actor/states, not historical optimizer or replay reconstruction',
                     'Per-coordinate constraints change the objective; a scalar blind spot is not a bug',
                     'Quadratic negative control is an analytic counterexample, not empirical generalization'])
    out = ROOT/'research/discovery/coordinate-entropy-gate-20261029.json'
    if out.exists(): raise FileExistsError('Preserve prior receipt')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('state', 'quadrature_max_abs_difference', 'scalar_signal_mean', 'scalar_decrease_state_fraction', 'coordinates', 'negative_control', 'elapsed_seconds')}))


if __name__ == '__main__': run()
