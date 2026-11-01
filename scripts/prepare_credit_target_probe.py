"""Prepare matched target/baseline actor directions from old training data.

Only safe actor JSON is sent to the existing public rollout worker. No new
experience is collected here, and no failed historical screen is rewritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]


def prepare(output):
    import numpy as np
    import torch
    from automanip.policy_io import load_policy, PolicySchema
    from automanip.numpy_inference import NumpyPolicy
    from automanip.stackcube import STACK_TASK
    from research.discovery.amp_v1 import read_actor, export_actor, PRODUCTION_PINS
    from research.crossfit_value_probe import crossfit_baselines
    from research.old_state_metric_probe import constrained_direction

    torch.set_num_threads(2)
    started = time.monotonic()
    output = Path(output); output.mkdir(exist_ok=False, parents=True)
    old_inputs = ROOT/'.runtime/proxy-calibration-20261030/inputs'
    old_plan = json.loads((old_inputs/'plan.json').read_text())
    base, _ = read_actor(old_inputs/'policies/no_update')
    schema = PolicySchema(STACK_TASK.fingerprint, 48, 8)
    base_artifact = load_policy(old_inputs/'policies/no_update', schema)
    if base_artifact.weights_sha256 != old_plan['source_actor_weights_sha256']:
        raise ValueError('Source actor changed')
    files_read = {}; parity = []; fitting = []; policies = {}

    def record_input(path):
        files_read[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def copy_policy(key, old_key):
        folder = output/'policies'/key; folder.mkdir(parents=True)
        source = old_inputs/old_plan['policies'][old_key]
        for name in ['manifest.json', 'weights.json']:
            path = source/name
            record_input(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != old_plan['policy_files'][str(path.relative_to(old_inputs))]:
                raise ValueError('Previous safe policy drift')
            (folder/name).write_bytes(path.read_bytes())
        load_policy(folder, schema)
        policies[key] = str(folder.relative_to(output))

    copy_policy('no_update', 'no_update')
    for seed in [90282001, 90282002, 90282003]:
        collection = ROOT/'runs/stackcube-first-update-probe-20261028/run'/str(seed)
        state_file = collection/'update-state.npz'
        collection_file = collection/'collection.npz'
        statistics_file = ROOT/f'artifacts/cumulative/gradient-aggregation-cpu-v1/{seed}-statistics.npz'
        for path in [state_file, collection_file, statistics_file]:
            record_input(path)
        with np.load(state_file, allow_pickle=False) as z:
            for key, value in base.net.state_dict().items():
                if not np.array_equal(value.numpy(), z['initial_actor_mean.'+key]):
                    raise ValueError('Initial mean differs')
            std = torch.from_numpy(z['initial_actor_logstd'].copy()).exp().flatten().double()
            targets = torch.from_numpy(z['returns'].copy()).double()
        with np.load(collection_file, allow_pickle=False) as z:
            if (z['dones'][1:] != 0).any():
                raise ValueError('Unexpected interior reset')
            obs = torch.from_numpy(z['obs'].copy().reshape(-1, 48))
            actions = torch.from_numpy(z['actions'].copy().reshape(-1, 8))
        with np.load(statistics_file, allow_pickle=False) as z:
            current = torch.from_numpy(z['current'].copy())
            legacy_direction = torch.from_numpy(z['full_rollout_direction'].copy())
        xs = []; eps = []
        with torch.no_grad():
            for start in range(0, len(obs), 2048):
                block = obs[start:start+2048]
                hidden = base.net[:-1](block)
                xs.append(torch.cat([hidden.double(), torch.ones((len(block), 1), dtype=torch.float64)], axis=1))
                eps.append((actions[start:start+2048].double()-base(block).double())/std)
        phi = torch.cat(xs); innovation = torch.cat(eps)
        baseline = crossfit_baselines(phi, targets, 1024, .001)
        residual = targets - baseline['state']
        gradient = (innovation*residual[:, None]).T@phi/len(phi)
        ridge = .001*float(current.trace())/len(current)
        gae_direction, stats = constrained_direction(gradient, current, torch.zeros_like(current), 0., .01, ridge)
        fitting.append(dict(collection_seed=seed, gae_state=stats, fitting_rows=len(phi),
                            baseline_MSE=float(residual.square().mean()), baseline_folds='whole episode even/odd; own labels excluded'))
        for method, direction in [('gae_legacy', legacy_direction), ('gae_state', gae_direction)]:
            for sign, label in [(1., 'forward'), (-1., 'reverse')]:
                actor = copy.deepcopy(base)
                with torch.no_grad():
                    actor.net[-1].weight.add_((sign*std[:, None]*direction[:, :-1]).float())
                    actor.net[-1].bias.add_((sign*std*direction[:, -1]).float())
                key = f'{seed}/{method}_{label}'; folder = output/'policies'/key
                export_actor(actor, folder); artifact = load_policy(folder, schema)
                examples = obs[:32]
                with torch.no_grad():
                    expected = actor(examples).clamp(-1, 1).numpy()
                error = float(np.abs(expected - np.asarray(NumpyPolicy(artifact, schema).act(examples.tolist()))).max())
                if error > 1e-5:
                    raise ValueError('Export parity failed')
                parity.append(dict(policy=key, max_action_error=error))
                policies[key] = str(folder.relative_to(output))
        for label in ['forward', 'reverse']:
            copy_policy(f'{seed}/mc_state_{label}', f'{seed}/state_{label}')
    safe_files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((output/'policies').rglob('*')) if p.is_file()}
    plan = dict(format='automanip-credit-target-public-probe-v1', created_at=time.time(),
                execution_authorized=True,
                user_authorization='2026-10-30 continue autonomous research, distinct GPU experiments, cost not an issue',
                purpose='Separate historical GAE baseline, fitted state baseline, and GAE versus finite MC target at identical readout geometry and mean KL. No hard heldout/fitted proxy admission rule.',
                production_pins=PRODUCTION_PINS, source_actor_weights_sha256=base_artifact.weights_sha256,
                policies=policies, policy_files=safe_files, safe_export_parity=parity,
                preparation_inputs=files_read, preparation_fits=fitting,
                preparation_seconds=time.monotonic()-started,
                value_linear_solves=6, actor_linear_solves=3, new_fitting_interactions=0,
                collection_seeds=[90282001, 90282002, 90282003],
                diagnostic_seed_bases=[93900000, 93901000, 93902000],
                noise_seed_bases=[93910000, 93911000, 93912000],
                gaussian_std=old_plan['gaussian_std'], modes=['stochastic'],
                branch_order=['no_update','gae_legacy_forward','gae_legacy_reverse','gae_state_forward','gae_state_reverse','mc_state_forward','mc_state_reverse'],
                horizon=50, vector_environments=64, batches_per_branch=8, episodes_per_branch=512,
                max_public_transitions=537600, max_public_transitions_per_collection=179200,
                optimizer_calls=0, protected_benchmark_queries=0, confirmation_queries=0, final_queries=0,
                target_difference='GAE uses archived production returns=GAE_advantage+initialV including original bootstrap. MC uses sum of future rewards gamma.9 to50 with zero tail. Different estimands, not an unbiased equivalence claim.',
                baseline_control='GAE-state and MC-state both fit same257 actor features, ridge.001trace/d, same episode crossfit. GAE-legacy retains original untrainedV and historical advantage centering.',
                geometry='All gradients aggregate51200 old rows; same current covariance, ridge, mean Gaussian KL=.01, frozen features/std. This is not a comparison of complete PPO learners.',
                primary_contrasts=['mc_state_forward-minus-gae_state_forward: target choice at matched baseline class', 'gae_state_forward-minus-gae_legacy_forward: baseline choice at fixed GAE target'],
                primary_rule='A contrast supports material local benefit only if mean discounted-return gain>=.03 and paired97.5%bootstrap lower bound>0. Otherwise inconclusive unless upper bound<0. Forward/reverse and success are diagnostic, no revised stepsize.',
                intervals='10000 batch-bootstrap draws within each fixed collection;8batches of64episodes. Conditional on three fixed directions, not independent training seeds.',
                bootstrap_seed=93909999,
                assistance=old_plan['assistance'],
                worker_sha256=hashlib.sha256((ROOT/'research/proxy_calibration_probe.py').read_bytes()).hexdigest(),
                timeout_seconds=240, startup_timeout_seconds=60, internal_seconds=200,
                followup='Reassess before iterative learner; no automatic retraining, retries or holdout queries')
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(policies=len(policies), files=len(safe_files), cpu_seconds=plan['preparation_seconds'],max_action_error=max(p['max_action_error'] for p in parity))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args().output)
