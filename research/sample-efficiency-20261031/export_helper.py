import torch
from torch import nn
from pathlib import Path
import hashlib,json
TASK_FINGERPRINT="b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f"

def export_actor(actor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    linear = [layer for layer in actor.net if isinstance(layer, nn.Linear)]
    for i, layer in enumerate(linear):
        for key in ("weight", "bias"):
            value = getattr(layer, key).detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite student actor")
            tensors[f"layers.{i}.{key}"] = dict(shape=list(value.shape), values=value.flatten().tolist())
    raw = json.dumps(tensors, allow_nan=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = dict(format="automanip-json-f32-v1", task_fingerprint=TASK_FINGERPRINT,
                    observation_dim=48, action_dim=8, hidden_sizes=actor.hidden_sizes,
                    activation="tanh", weights_sha256=digest)
    (directory/"weights.json").write_bytes(raw)
    (directory/"manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return dict(path=str(directory), weights_sha256=digest)
