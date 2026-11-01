"""Current frozen surrogate gradients, not reconstructed training or Q truth."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from automanip.numpy_inference import NumpyPolicy
from automanip.policy_io import PolicySchema, load_policy
from replay_learner_lowalpha import Critic
from sac_baseline import Actor


def describe(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()), positive_fraction=float((x > 0).mean()))


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    base = ROOT/'runs/temperature-sufficiency-20261028/run'
    state_path = base/'actor-step-256000-state.npz'
    trajectory_path = base/'actor-step-256000-trajectory.npz'
    expected = {state_path: 'ed55d938dcf21a5e80a1268921303ebed23d67dbb556b761cdd736f761f4ae6f',
                trajectory_path: 'fdd401f214a41f8f7dcd26df6c7d4797b0dd0e3be3ce7ab8fc6750cf123d6332'}
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Input snapshot differs')
    with np.load(state_path, allow_pickle=False) as saved:
        models = [Actor().double().eval(), Critic().double().eval(), Critic().double().eval()]
        for prefix, model in zip(('actor', 'q1', 'q2'), models):
            state = {key: torch.from_numpy(saved[prefix+'.'+key].copy()).double()
                     for key in model.state_dict()}
            model.load_state_dict(state)
            model.requires_grad_(False)
        alpha = float(np.exp(saved['log_alpha']))
    actor, q1, q2 = models
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        observations = trajectory['next_obs'][trajectory['stage_flags'][..., 0].astype(bool)].astype(np.float64)
    if observations.shape != (1904, 48): raise ValueError('Declared observation subset differs')
    obs = torch.from_numpy(observations)
    with torch.no_grad(): mean, logstd = actor(obs)
    deterministic = torch.tanh(mean).detach().requires_grad_(True)
    values = torch.minimum(q1(obs, deterministic), q2(obs, deterministic))
    action_gradient, = torch.autograd.grad(values.sum(), deterministic)
    jacobian = 1-deterministic.detach().square()
    deterministic_mean_q_gradient = action_gradient*jacobian
    schema = PolicySchema('b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f', 48, 8)
    safe = load_policy(base/'actor-step-256000', schema)
    if safe.weights_sha256 != '98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a':
        raise ValueError('Safe policy differs')
    parity = np.max(np.abs(np.asarray(NumpyPolicy(safe, schema).act(observations))-deterministic.detach().numpy()))
    if parity > 1e-10: raise ValueError('Safe inference parity failed')

    rng = np.random.default_rng(93294000)
    noise = torch.from_numpy(rng.normal(size=(16, len(obs), 8)))
    repeated = obs.repeat(16, 1)
    mu = mean.detach().repeat(16, 1).requires_grad_(True)
    std_log = logstd.detach().repeat(16, 1)
    z = noise.reshape(-1, 8)
    latent = mu+std_log.exp()*z
    action = latent.tanh()
    logprob = (-.5*(z.square()+2*std_log+math.log(2*math.pi))
               -2*(math.log(2)-latent-F.softplus(-2*latent))).sum(-1, keepdim=True)
    qs = torch.minimum(q1(repeated, action), q2(repeated, action))
    sampled_action_gradient, = torch.autograd.grad(qs.sum(), action, retain_graph=True)
    grad_q, = torch.autograd.grad(qs.sum(), mu, retain_graph=True)
    grad_entropy_cost, = torch.autograd.grad((alpha*logprob).sum(), mu, retain_graph=True)
    grad_loss, = torch.autograd.grad((alpha*logprob-qs).sum(), mu)
    entropy_error = float((grad_entropy_cost-2*alpha*action.detach()).abs().max())
    chain_error = float((grad_q-sampled_action_gradient*(1-action.detach().square())).abs().max())
    decomposition_error = float((-grad_loss-(grad_q-grad_entropy_cost)).abs().max())
    if max(entropy_error, chain_error, decomposition_error) > 1e-10: raise ValueError('Gradient decomposition differs')

    # Fixed, declared small directional check on the first sixteen stored states.
    direction = torch.from_numpy(np.random.default_rng(93294001).normal(size=(16, 8)))
    direction /= direction.norm(dim=1, keepdim=True)
    chosen = deterministic.detach()[:16]
    epsilon = 1e-6
    with torch.no_grad():
        def value(a): return torch.minimum(q1(obs[:16], a), q2(obs[:16], a)).squeeze(-1)
        difference = (value(chosen+epsilon*direction)-value(chosen-epsilon*direction))/(2*epsilon)
    analytic = (action_gradient[:16]*direction).sum(-1)
    per_joint = []
    for index in range(8):
        per_joint.append(dict(index=index,
            deterministic_action=describe(deterministic.detach().numpy()[:, index]),
            deterministic_tanh_derivative=describe(jacobian.numpy()[:, index]),
            deterministic_Q_action_gradient=describe(action_gradient.numpy()[:, index]),
            deterministic_Q_mean_gradient=describe(deterministic_mean_q_gradient.numpy()[:, index]),
            sampled_Q_mean_ascent=describe(grad_q.numpy()[:, index]),
            sampled_entropy_mean_ascent=describe(-grad_entropy_cost.numpy()[:, index]),
            sampled_total_mean_ascent=describe(-grad_loss.numpy()[:, index]),
            sampled_action_mean=float(action.detach()[:, index].mean())))
    result = dict(state='complete', kind='Frozen current surrogate-gradient diagnostic, not learning or critic validation',
        saved_grasped_successor_rows=len(obs), alpha=alpha, safe_policy_parity_max_abs=float(parity),
        samples_per_observation=16, sample_seed=93294000, per_joint=per_joint,
        entropy_gradient_identity_max_abs_error=entropy_error,
        Q_tanh_chain_rule_max_abs_error=chain_error,
        surrogate_decomposition_max_abs_error=decomposition_error,
        finite_difference_directional_max_abs_error=float((difference-analytic).abs().max()),
        deterministic_minQ=describe(values.detach().numpy()),
        actor_forward_rows=len(obs), safe_numpy_policy_forward_rows=len(obs),
        twin_critic_forward_rows=2*(len(obs)+16*len(obs)+32),
        autograd_requests_rows=dict(deterministic_Q_action=len(obs), sampled_Q_action=16*len(obs),
            sampled_Q_mean=16*len(obs), sampled_entropy_mean=16*len(obs), sampled_surrogate_mean=16*len(obs)),
        new_simulator_transitions=0, optimizer_calls=0, elapsed_seconds=time.monotonic()-start,
        input_sha256={str(path.relative_to(ROOT)): digest for path, digest in expected.items()},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT/'research/discovery/replay_learner_lowalpha.py',
                                    ROOT/'research/discovery/sac_baseline.py')},
        limitations=['No historical minibatch/target/optimizer reconstruction',
                     'Critic preferences are not physical counterfactual returns',
                     'Saved behavior states are not the full replay distribution',
                     'Mean-space directions are not complete network/Adam updates',
                     'Monte Carlo rows are dependent repeated states, not independent episodes'])
    destination = ROOT/'research/discovery/actor-gradient-snapshot-20261029.json'
    if destination.exists(): raise FileExistsError('Preserve existing probe result')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('state', 'saved_grasped_successor_rows', 'alpha',
        'safe_policy_parity_max_abs', 'entropy_gradient_identity_max_abs_error',
        'finite_difference_directional_max_abs_error', 'elapsed_seconds')}))


if __name__ == '__main__': main()
