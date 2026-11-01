"""Training-only state-conditioned diagonal exploration; deterministic actor unchanged.

Conventional Gaussian policy-family extension, not gSDE or a new algorithm.
Independent noise each step preserves conditional Gaussian likelihoods.
"""
from __future__ import annotations

import torch
from torch import nn


class LogStdResidual(nn.Module):
    def __init__(self, obs_dim, action_dim, width=64):
        super().__init__()
        # Constructing the extra head must not perturb the base initialization or
        # first sampling RNG stream. Head is created on CPU before Agent.to(...).
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(), nn.Linear(width, action_dim))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return self.net(obs)


def gaussian(mean, base_logstd, residual, obs):
    logstd = (base_logstd.expand_as(mean) + residual(obs)).clamp(-5., 2.)
    return torch.distributions.Normal(mean, logstd.exp())


def wrap_agent(base_class):
    """Return an adapter for the pinned public PPO Agent; no simulator imports.

    The owner can substitute this class before construction in an isolated
    trainer. actor_mean/critic definitions and deterministic artifact are intact.
    Gaussian entropy remains latent entropy and executed samples are still clipped
    by the original trainer. The new head sees only the existing observations.
    """
    class StateVarianceAgent(base_class):
        def __init__(self, envs):
            super().__init__(envs)
            self.logstd_residual = LogStdResidual(
                envs.single_observation_space.shape[0], envs.single_action_space.shape[0])

        def get_action(self, x, deterministic=False):
            mean = self.actor_mean(x)
            if deterministic:
                return mean
            return gaussian(mean, self.actor_logstd, self.logstd_residual, x).sample()

        def get_action_and_value(self, x, action=None):
            distribution = gaussian(self.actor_mean(x), self.actor_logstd, self.logstd_residual, x)
            if action is None:
                action = distribution.sample()
            return action, distribution.log_prob(action).sum(1), distribution.entropy().sum(1), self.critic(x)

    return StateVarianceAgent
