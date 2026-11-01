"""Matched-marginal exploration algebra and fixed-state kinematic coverage."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from panda_kinematics import Kinematics, parse_chain
from sac_baseline import Actor

ROOT = Path(__file__).resolve().parents[2]


def event_probabilities(p1, p2, u1, u2, epsilon=.1):
    vector = (1-epsilon)*p1*p2+epsilon*u1*u2
    coordinate = ((1-epsilon)*p1+epsilon*u1)*((1-epsilon)*p2+epsilon*u2)
    return vector, coordinate


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    epsilon = .1
    positive = event_probabilities(.001, .001, .25, .25)
    negative = event_probabilities(.001, .999, .25, .01)
    single = event_probabilities(.001, 1., .25, 1.)
    assert positive[0]/positive[1] > 9 and negative[0] < negative[1] and single[0] == single[1]
    identity = positive[0]-positive[1]-epsilon*(1-epsilon)*(.25-.001)**2
    assert abs(identity) < 1e-14
    coverage_path = ROOT/'research/discovery/grasp-action-coverage-20261029.json'
    coverage = json.loads(coverage_path.read_bytes())
    trajectory = ROOT/'runs/randomized-lift-20261029/run/trajectories.npz'
    if hashlib.sha256(trajectory.read_bytes()).hexdigest() != coverage['input_sha256']['randomized_trajectory']:
        raise ValueError('Fixed diagnostic states differ')
    with np.load(trajectory, allow_pickle=False) as data:
        lanes = np.flatnonzero(data['eligibility_step'] >= 0)
        observations = data['observations'][data['eligibility_step'][lanes], lanes].astype(np.float64)
    if len(lanes) != 78 or [row['lane'] for row in coverage['states']] != lanes.tolist():
        raise ValueError('Do not change the all-eligible state set')
    actor_path = ROOT/'runs/temperature-sufficiency-20261028/run/actor-step-256000-state.npz'
    if hashlib.sha256(actor_path.read_bytes()).hexdigest() != coverage['input_sha256']['actor_state']:
        raise ValueError('Actor state differs')
    actor = Actor().double().eval().requires_grad_(False)
    with np.load(actor_path, allow_pickle=False) as data:
        actor.load_state_dict({key: torch.from_numpy(data['actor.'+key].copy()).double() for key in actor.state_dict()})
    with torch.no_grad(): mean, logstd = [value.numpy() for value in actor(torch.from_numpy(observations))]
    gaussian = np.tanh(mean[:, None, :7]+np.exp(logstd[:, None, :7])*
                       np.random.default_rng(93314100).normal(size=(78, 512, 7)))
    uniform = np.random.default_rng(93314200).uniform(-1, 1, size=(78, 512, 7))
    coins = np.random.default_rng(93314300).uniform(size=(78, 512, 7)) < epsilon
    mixed = np.where(coins, uniform, gaussian)
    urdf = Path(importlib.util.find_spec('mani_skill').origin).parent/'assets/robots/panda/panda_v3.urdf'
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != 'd040277352476639950e2ddce1dff8a1c0e95df470e5e9ed186cd01c468fa412':
        raise ValueError('Public kinematics differs')
    chain = parse_chain(urdf); kin = Kinematics(chain)
    limits = np.asarray([joint['limits'] for joint in chain if joint['type'] == 'revolute'])
    q = observations[:, :7]
    current = kin.fk(q)
    targets = q[:, None, :]+.1*mixed
    valid = ((targets >= limits[:, 0]) & (targets <= limits[:, 1])).all(-1)
    future = kin.fk(targets.reshape(-1, 7))
    up = valid & (future[:, 2, 3].reshape(78, 512)-current[:, None, 2, 3] >= .005)
    vector_mass, coordinate_mass, rows = [], [], []
    for i, source in enumerate(coverage['states']):
        p = source['closing_proxies']['10mm']['exact_probability']
        p_arm = source['arms']['current_gaussian']['primary_up_probability']
        u_arm = source['arms']['uniform_reference']['primary_up_probability']
        vector = .9*p*p_arm+.1*.4*u_arm
        coordinate = (.9*p+.1*.4)*float(up[i].mean())
        vector_mass.append(vector); coordinate_mass.append(coordinate)
        rows.append(dict(lane=int(lanes[i]), vector_proxy_mass=vector, coordinate_proxy_mass=coordinate,
                         coordinate_up_MC_count=int(up[i].sum()), coordinate_valid_targets=int(valid[i].sum())))
    vmean, cmean = float(np.mean(vector_mass)), float(np.mean(coordinate_mass))
    vmedian, cmedian = float(np.median(vector_mass)), float(np.median(coordinate_mass))
    passed = vmedian > 0 and vmedian >= 5*cmedian and vmean >= cmean
    result = dict(state='complete', primary_proxy='10mm setpoint and valid target-FK rise>=5mm, not physical success',
        gate_passed=passed, vector_mean=vmean, coordinate_mean=cmean,
        vector_median=vmedian, coordinate_median=cmedian,
        median_ratio=vmedian/cmedian if cmedian > 0 else None,
        toy_rare_pair=dict(vector_probability=positive[0], coordinate_probability=positive[1], ratio=positive[0]/positive[1]),
        toy_mixed_negative=dict(vector_probability=negative[0], coordinate_probability=negative[1]),
        single_coordinate_equal=single[0] == single[1], exact_identity_abs_error=abs(identity),
        states=rows, analytic_FK_pose_evaluations=kin.fk_pose_evaluations,
        actor_forward_rows=78, newly_evaluated_arm_vectors=78*512,
        regenerated_original_scalar_variates=2*78*512*7, new_mask_scalar_draws=78*512*7,
        reused_coverage_receipt_sha256=hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
        original_measurements_reused_not_replications=True, MC_resolution_per_state=1/512,
        optimizer_calls=0, new_simulator_transitions=0, elapsed_seconds=time.monotonic()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Matched conditional marginals only at the same fixed policy/state',
                 'Finite MC point gate, not a guaranteed distributional inequality or population effect',
                 'No temporal-coherence or learning-rate claim', 'No candidate or GPU admission from gate alone'])
    destination = ROOT/'research/discovery/joint-exploration-gate-20261029.json'
    if destination.exists(): raise FileExistsError('Preserve prior gate')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('state', 'gate_passed', 'vector_mean', 'coordinate_mean',
                                                 'vector_median', 'coordinate_median', 'median_ratio',
                                                 'toy_rare_pair', 'toy_mixed_negative', 'elapsed_seconds')}))


if __name__ == '__main__': run()
