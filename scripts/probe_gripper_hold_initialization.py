"""CPU mechanism gate using the exact pinned production Agent definition."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from research.gripper_hold_initialization import configure, maximum_fixed_change, verify_optimizer
from scripts.prepare_scratch_credit import BASE, BASE_SHA


def production_agent():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Pinned production recipe differs')
    definitions = [n for n in ast.parse(raw).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in ('layer_init', 'Agent')]
    if len(definitions) != 2:
        raise ValueError('Production definitions missing')
    namespace = dict(np=np, torch=torch, nn=nn, Normal=Normal)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(BASE), 'exec'), namespace)
    env = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(48,)),
                          single_action_space=SimpleNamespace(shape=(8,)))
    torch.manual_seed(41)
    return namespace['Agent'](env)


def probe():
    started = time.process_time()
    full = production_agent()
    control, candidate = copy.deepcopy(full), copy.deepcopy(full)
    cm, _ = configure(control.actor_mean, False)
    hm, _ = configure(candidate.actor_mean, True)
    critic_same = all(torch.equal(a, b) for a, b in zip(control.critic.parameters(), candidate.critic.parameters()))
    logstd_same = torch.equal(control.actor_logstd, candidate.actor_logstd)
    x = torch.randn(1001, 48)
    x[:, 7] = torch.linspace(-.01, .04, len(x))
    with torch.no_grad():
        difference = candidate.actor_mean(x)-control.actor_mean(x)
        error = float(((difference[:, 7]-(40*x[:, 7]-.6))*.025).abs().max())
        other_error = float(difference[:, :7].abs().max())
    before = [p.detach().clone() for p, _ in hm]
    optimizer = torch.optim.Adam(candidate.parameters(), lr=.0003, eps=1e-5)
    verify_optimizer(optimizer)
    # Synthetic optimizer stress, not a task-learning experiment.
    for _ in range(64):
        obs = torch.randn(64, 48)
        obs[:, 7] = torch.linspace(-.01, .04, 64)
        with torch.no_grad():
            actions, _, _, _ = candidate.get_action_and_value(obs)
        optimizer.zero_grad()
        _, logprob, entropy, value = candidate.get_action_and_value(obs, actions)
        loss = -(logprob*torch.linspace(-1, 1, 64)).mean()-.003*entropy.mean()+.5*value.square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(candidate.parameters(), .5)
        optimizer.step()
    drift = maximum_fixed_change(hm, before)
    # Actuator prior is a mean map: Gaussian noise and clipping still move it.
    z = torch.linspace(.000005, .999995, 100000)
    noise = Normal(0., 1.).icdf(z)*np.exp(-.5)
    stochastic = []
    for q in [-.01, .015, .04]:
        hold = 40*q-.6
        stochastic.append(dict(q=q,
            zero_mean_expected_target=float((.015+.025*noise.clamp(-1, 1)).mean()),
            hold_mean_expected_target=float((.015+.025*(noise+hold).clamp(-1, 1)).mean())))
    passed = error < 3e-7 and other_error == 0 and drift == 0 and critic_same and logstd_same
    return dict(state='complete',passed=passed,production_source_sha256=BASE_SHA,
        hold_target_max_error_metres=error,non_gripper_difference=other_error,
        fixed_coordinate_drift_after_64_Adam_steps=drift,critic_identical=critic_same,
        logstd_identical=logstd_same,matched_masks=all(torch.equal(a, b) for (_, a), (_, b) in zip(cm, hm)),
        synthetic_optimizer_steps=64,CPU_seconds=time.process_time()-started,
        new_environment_transitions=0,training_seed=41,stochastic_actuator_check=stochastic,
        limitation='Known mean hold map only; clipping makes noisy commands drift near limits. CPU gate does not establish learning or exploration benefit.')


if __name__ == '__main__':
    result = probe()
    output = Path('cumulative/results/gripper-hold-CPU-20261031.json')
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
