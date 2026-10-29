"""Bounded JSON tensor interchange for the CPU MVP.

Only finite float32-representable numbers and evaluator-defined MLP parameters
are accepted. No pickle, dynamic imports, paths from manifests, or candidate code.
This deliberately simple format can later be replaced by safetensors.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Mapping

MAX_MANIFEST_BYTES = 8192
MAX_WEIGHTS_BYTES = 32 * 1024 * 1024
MAX_PARAMETERS = 500_000


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes):
    try:
        return json.loads(data, object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc


def _read(path: Path, limit: int) -> bytes:
    # Reading limit+1 bounds memory even when a file changes after stat().
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("artifact exceeds byte limit")
    return data


@dataclass(frozen=True)
class PolicySchema:
    """Created by the evaluator, after measuring its observation interface."""
    task_fingerprint: str
    observation_dim: int
    action_dim: int = 7

    def __post_init__(self):
        if not isinstance(self.task_fingerprint, str) or len(self.task_fingerprint) != 64:
            raise ValueError("invalid task fingerprint")
        for dim in (self.observation_dim, self.action_dim):
            if type(dim) is not int or not 1 <= dim <= 4096:
                raise ValueError("invalid interface dimension")


@dataclass(frozen=True)
class ValidatedPolicy:
    hidden_sizes: tuple[int, ...]
    tensors: Mapping[str, tuple[float, ...]]
    weights_sha256: str
    # This is a data object, not an executable candidate policy.


def load_policy(directory: Path, schema: PolicySchema) -> ValidatedPolicy:
    """Validate a fixed-name manifest and JSON weights against trusted schema.

    Digests identify artifacts; a candidate-supplied digest proves no authenticity.
    Process/filesystem isolation remains a separate deployment requirement.
    """
    directory = Path(directory)
    manifest = _json(_read(directory / "manifest.json", MAX_MANIFEST_BYTES))
    required = {"format", "task_fingerprint", "observation_dim", "action_dim",
                "hidden_sizes", "activation", "weights_sha256"}
    if type(manifest) is not dict or set(manifest) != required:
        raise ValueError("unexpected manifest fields")
    if manifest["format"] != "automanip-json-f32-v1" or manifest["activation"] != "tanh":
        raise ValueError("unsupported architecture or format")
    if manifest["task_fingerprint"] != schema.task_fingerprint:
        raise ValueError("task mismatch")
    for key, expected in (("observation_dim", schema.observation_dim), ("action_dim", schema.action_dim)):
        if type(manifest[key]) is not int or manifest[key] != expected:
            raise ValueError("interface mismatch")
    hidden = manifest["hidden_sizes"]
    if type(hidden) is not list or not 1 <= len(hidden) <= 3:
        raise ValueError("expected one to three hidden layers")
    if any(type(n) is not int or not 1 <= n <= 256 for n in hidden):
        raise ValueError("invalid hidden width")
    dims = [schema.observation_dim, *hidden, schema.action_dim]
    shapes = {}
    for i, (inputs, outputs) in enumerate(zip(dims, dims[1:])):
        shapes[f"layers.{i}.weight"] = [outputs, inputs]
        shapes[f"layers.{i}.bias"] = [outputs]
    if sum(math.prod(shape) for shape in shapes.values()) > MAX_PARAMETERS:
        raise ValueError("parameter budget exceeded")
    raw = _read(directory / "weights.json", MAX_WEIGHTS_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if manifest["weights_sha256"] != digest:
        raise ValueError("weights digest mismatch")
    weights = _json(raw)
    if type(weights) is not dict or set(weights) != set(shapes):
        raise ValueError("unexpected tensor names")
    validated = {}
    for name, expected_shape in shapes.items():
        tensor = weights[name]
        if type(tensor) is not dict or set(tensor) != {"shape", "values"}:
            raise ValueError("invalid tensor record")
        shape = tensor["shape"]
        if type(shape) is not list or any(type(n) is not int for n in shape) or shape != expected_shape:
            raise ValueError("tensor shape mismatch")
        values = tensor["values"]
        if type(values) is not list or len(values) != math.prod(shape):
            raise ValueError("tensor length mismatch")
        converted = []
        for value in values:
            if type(value) not in (int, float):
                raise ValueError("tensor values must be numbers")
            try:
                number = struct.unpack("<f", struct.pack("<f", value))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError("value outside float32 range") from exc
            if not math.isfinite(number):
                raise ValueError("nonfinite weight")
            converted.append(number)
        validated[name] = tuple(converted)
    return ValidatedPolicy(tuple(hidden), MappingProxyType(validated), digest)
