"""Known actuator hold map inside a plain MLP; no simulator or task queries.

The fixed path uses one existing unit per hidden layer. Both masked controls
freeze exactly the same coordinates; only the fixed output gain differs.
This is an explicit controller prior, not a learned skill or pretrained actor.
"""
import torch
from torch import nn


GAIN = .003


def configure(actor, hold):
    if type(hold) is not bool:
        raise ValueError('Explicit hold/no-hold comparison required')
    if len(actor) != 7 or any(not isinstance(actor[i], nn.Tanh) for i in (1, 3, 5)):
        raise ValueError('Expected plain three-hidden-layer tanh actor')
    layers = [actor[i] for i in (0, 2, 4, 6)]
    if [(l.in_features, l.out_features) for l in layers] != [(48, 256), (256, 256), (256, 256), (256, 8)]:
        raise ValueError('Frozen StackCube actor shape differs')
    masks = []
    with torch.no_grad():
        for i, layer in enumerate(layers):
            wm, bm = torch.ones_like(layer.weight), torch.ones_like(layer.bias)
            if i < 3:
                wm[-1, :] = 0
                bm[-1] = 0
                layer.weight[-1, :] = 0
                layer.bias[-1] = 0
            if i > 0:
                wm[:, -1] = 0
                layer.weight[:, -1] = 0
            if i == 0:
                layer.weight[-1, 7] = 40*GAIN
                layer.bias[-1] = -.6*GAIN
            elif i < 3:
                layer.weight[-1, -1] = 1
            else:
                layer.weight[7, -1] = 1/GAIN if hold else 0
            masks.extend([(layer.weight, wm), (layer.bias, bm)])
    # No optimizer weight decay is allowed: a zero gradient alone would not
    # protect fixed nonzero coordinates from decoupled weight decay.
    handles = [p.register_hook(lambda grad, mask=mask: grad*mask) for p, mask in masks]
    return masks, handles


def verify_optimizer(optimizer):
    if any(group.get('weight_decay', 0) != 0 for group in optimizer.param_groups):
        raise ValueError('Fixed actuator path requires zero weight decay')


def maximum_fixed_change(masks, snapshots):
    return max(float(((p.detach()-old)*(1-mask)).abs().max())
               for (p, mask), old in zip(masks, snapshots))
