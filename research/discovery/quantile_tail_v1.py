"""Conventional quantile-return and lower-tail kernels; no environment imports.

One-step tail actor optimization is not automatically static episodic CVaR.
Distributional-return risk is distinct from model/ensemble epistemic uncertainty.
"""
from __future__ import annotations

import torch
from torch import nn


class QuantileCritic(nn.Module):
    def __init__(self, observation_dim=48, action_dim=8, width=128, quantiles=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_dim+action_dim, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, quantiles))
        self.register_buffer("taus", (torch.arange(quantiles, dtype=torch.float32)+.5)/quantiles)

    def forward(self, observation, action):
        return self.net(torch.cat((observation, action), -1))


def quantile_loss(prediction, target, taus):
    """Exact pinball loss (no Huber bias on rare low-return events)."""
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape[0] != target.shape[0]:
        raise ValueError("Expected matching batches of quantiles/target samples")
    if len(taus) != prediction.shape[1]:
        raise ValueError("Quantile probabilities mismatch")
    residual = target.detach().unsqueeze(1)-prediction.unsqueeze(2)
    weight = (taus.view(1, -1, 1)-(residual.detach() < 0).to(prediction.dtype)).abs()
    return (weight*residual.abs()).mean()


def lower_tail(quantiles, fraction=.1):
    """Integrate fixed equal-mass quantile bins, including a fractional last bin.

    Inputs retain their semantic quantile-index order; crossing is diagnosed,
    not silently hidden by sorting arbitrary network outputs.
    """
    if not 0 < fraction <= 1 or quantiles.ndim != 2 or quantiles.shape[1] < 1:
        raise ValueError("Invalid quantile array or tail mass")
    width = quantiles.shape[1]
    weights = (fraction*width-torch.arange(width, device=quantiles.device)).clamp(0., 1.)/(fraction*width)
    return (quantiles*weights).sum(-1, keepdim=True)


def episode_cvar_score_weights(returns, fraction=.1):
    """Finite-sample lower-tail score estimator with VaR baseline.

    Each row is one complete ordinary episode, including failures. No critic or
    success label is used. Correct asymptotic CVaR PG uses (G-q_alpha)I[G<=q]/alpha,
    rather than treating selected bad returns as an unbiased mean-return sample.
    Ties at the quantile are assigned zero weight by their zero residual.
    This is an estimator, not an unbiased finite-batch guarantee.
    """
    if returns.ndim != 1 or len(returns) < 2 or not 0 < fraction <= 1:
        raise ValueError("Expected complete-episode return vector and valid mass")
    if not torch.isfinite(returns).all():
        raise ValueError("Nonfinite episode return")
    threshold = torch.quantile(returns.detach(), fraction)
    weights = ((returns.detach()-threshold)*(returns.detach() <= threshold))/fraction
    return weights, threshold
