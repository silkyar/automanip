"""Trusted vectorized execution of already validated MLP tensors only."""
import numpy as np


class NumpyPolicy:
    def __init__(self, artifact, schema):
        self.schema = schema
        dims = (schema.observation_dim, *artifact.hidden_sizes, schema.action_dim)
        self.layers = [(np.asarray(artifact.tensors[f"layers.{i}.weight"], dtype=np.float64).reshape(right,left),
                        np.asarray(artifact.tensors[f"layers.{i}.bias"], dtype=np.float64))
                       for i, (left,right) in enumerate(zip(dims,dims[1:]))]

    def act(self, observations):
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.schema.observation_dim or not np.isfinite(x).all():
            raise ValueError("invalid observations")
        for i, (weight,bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if not np.isfinite(x).all():
                raise ValueError("nonfinite policy output")
            if i < len(self.layers)-1:
                x = np.tanh(x)
        return np.clip(x,-1,1).tolist()
