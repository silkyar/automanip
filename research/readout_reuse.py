"""Head-only actor adaptation: exact output-layer residual, unchanged runtime.

No simulator, deployment or evaluator imports. This is an established restricted
fine-tuning baseline, not a new residual architecture or a retention guarantee.
"""
import torch
from torch import nn


def configure_actor(actor, mode):
    if mode not in ('full','readout'):raise ValueError('Closed actor adaptation mode')
    layers=[layer for layer in actor.modules() if isinstance(layer,nn.Linear)]
    if not layers:raise ValueError('Expected existing MLP')
    for p in actor.parameters():p.requires_grad_(mode=='full')
    if mode=='readout':
        for p in layers[-1].parameters():p.requires_grad_(True)
    return dict(mode=mode,trainable_actor_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
                frozen_actor_parameters=sum(p.numel() for p in actor.parameters() if not p.requires_grad),
                inference='Existing single MLP; source output weights plus learned difference are exported together',
                retention_guaranteed=False)


def residual_equivalence(source, adapted, observations):
    """Check mu_new = mu_source + deltaW * source_features + deltaBias.

    Only valid with identical hidden tensors; clipping follows the full sum.
    """
    left=list(source.net.children());right=list(adapted.net.children())
    if len(left)!=len(right) or not isinstance(left[-1],nn.Linear) or not isinstance(right[-1],nn.Linear):raise ValueError('Actor architecture differs')
    for a,b in zip(left[:-1],right[:-1]):
        if type(a)!=type(b) or any(not torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())):raise ValueError('Source hidden representation changed')
    with torch.no_grad():
        hidden=observations
        for layer in left[:-1]:hidden=layer(hidden)
        residual=hidden@(right[-1].weight-left[-1].weight).T+(right[-1].bias-left[-1].bias)
        fused=adapted(observations)
        reconstructed=source(observations)+residual
        return dict(max_unclipped_difference=float((fused-reconstructed).abs().max()),
                    max_clipped_difference=float((fused.clamp(-1,1)-reconstructed.clamp(-1,1)).abs().max()))
