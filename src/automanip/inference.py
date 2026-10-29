"""Trusted, dependency-free deterministic MLP inference for smoke evaluation."""
import math
from automanip.policy_io import PolicySchema, ValidatedPolicy


class MLPPolicy:
    def __init__(self, artifact: ValidatedPolicy, schema: PolicySchema):
        self.artifact = artifact
        self.schema = schema

    def act(self, observations):
        result = []
        dims = [self.schema.observation_dim, *self.artifact.hidden_sizes, self.schema.action_dim]
        for row in observations:
            if len(row) != dims[0] or any(not math.isfinite(v) for v in row):
                raise ValueError("invalid observation")
            values = row
            for layer, (inputs, outputs) in enumerate(zip(dims, dims[1:])):
                weights = self.artifact.tensors[f"layers.{layer}.weight"]
                biases = self.artifact.tensors[f"layers.{layer}.bias"]
                values = [biases[j] + sum(weights[j*inputs+k]*values[k] for k in range(inputs))
                          for j in range(outputs)]
                if any(not math.isfinite(v) for v in values):
                    raise ValueError("nonfinite network output")
                if layer < len(dims)-2:
                    values = [math.tanh(v) for v in values]
            result.append([max(-1.0, min(1.0, v)) for v in values])
        return result
