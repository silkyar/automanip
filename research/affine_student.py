"""Function-preserving input reparameterization, folded into plain MLP weights."""
import numpy as np
import torch

from automanip.policy_io import ValidatedPolicy
from research.readout_distill import export_readout
from research.representation_distill import Student


class AffineStudent(Student):
    def __init__(self, artifact, mean, scale):
        super().__init__(artifact)
        mean, scale = np.asarray(mean), np.asarray(scale)
        if mean.shape != (48,) or scale.shape != (48,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid fixed training normalization')
        self.register_buffer('input_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('input_scale', torch.tensor(scale, dtype=torch.float32))
        with torch.no_grad():
            first = self.layers[0]; weight = first.weight.clone(); bias = first.bias.clone()
            first.weight.copy_(weight*self.input_scale[None])
            first.bias.copy_(bias+weight@self.input_mean)
    def forward(self, observations):
        return super().forward((observations-self.input_mean)/self.input_scale)
    def raw_parameters(self):
        first = self.layers[0]
        weight = first.weight/self.input_scale[None]
        bias = first.bias-weight@self.input_mean
        return [weight, bias, *(p for layer in self.layers[1:] for p in layer.parameters())]


def export_affine_student(model, artifact, schema, directory):
    raw = model.raw_parameters(); tensors = {}
    for i in range(len(model.layers)):
        for name, tensor in [('weight', raw[2*i]), ('bias', raw[2*i+1])]:
            tensors[f'layers.{i}.{name}'] = tuple(tensor.detach().numpy().reshape(-1).tolist())
    acting = ValidatedPolicy(artifact.hidden_sizes, tensors, 'folded-numeric-weights')
    return export_readout(acting, schema, np.zeros((artifact.hidden_sizes[-1]+1, schema.action_dim)), directory)
