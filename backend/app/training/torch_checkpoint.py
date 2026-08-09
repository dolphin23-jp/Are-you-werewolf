"""Safe NPZ checkpoints for the optional PyTorch Transformer policy.

The format stores JSON architecture metadata plus raw tensor arrays. Loading uses
``numpy.load(..., allow_pickle=False)`` and never deserializes Python objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from app.training.policy_contract import PolicyHeadSizes
from app.training.torch_policy import TransformerPolicyConfig, TorchTransformerPolicy

_CHECKPOINT_VERSION = 1
_METADATA_KEY = "__metadata__"
_TENSOR_PREFIX = "tensor::"


def save_torch_policy(model: TorchTransformerPolicy, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": _CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "sizes": asdict(model.sizes),
    }
    encoded_metadata = json.dumps(metadata, sort_keys=True).encode("utf-8")
    arrays: dict[str, np.ndarray[Any, Any]] = {
        _METADATA_KEY: np.frombuffer(encoded_metadata, dtype=np.uint8).copy(),
    }
    for name, tensor in model.state_dict().items():
        arrays[f"{_TENSOR_PREFIX}{name}"] = tensor.detach().cpu().numpy().copy()

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)


def load_torch_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> TorchTransformerPolicy:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = _decode_metadata(archive)
        model = TorchTransformerPolicy(
            TransformerPolicyConfig(**_mapping(metadata, "config")),
            sizes=PolicyHeadSizes(**_mapping(metadata, "sizes")),
        )
        expected = set(model.state_dict())
        available = {
            key.removeprefix(_TENSOR_PREFIX)
            for key in archive.files
            if key.startswith(_TENSOR_PREFIX)
        }
        if expected != available:
            missing = sorted(expected - available)
            extra = sorted(available - expected)
            raise ValueError(
                f"checkpoint tensors do not match model; missing={missing}, extra={extra}"
            )

        state: dict[str, Tensor] = {}
        for name, template in model.state_dict().items():
            array = np.array(archive[f"{_TENSOR_PREFIX}{name}"], copy=True)
            tensor = torch.from_numpy(array).to(dtype=template.dtype)
            if tensor.shape != template.shape:
                raise ValueError(f"checkpoint tensor {name} has unexpected shape")
            state[name] = tensor

    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def _decode_metadata(archive: Any) -> dict[str, Any]:
    if _METADATA_KEY not in archive.files:
        raise ValueError("checkpoint metadata is missing")
    raw = np.asarray(archive[_METADATA_KEY])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("checkpoint metadata has invalid encoding")
    try:
        parsed = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint metadata is invalid") from exc
    if not isinstance(parsed, dict) or parsed.get("version") != _CHECKPOINT_VERSION:
        raise ValueError("unsupported Transformer checkpoint version")
    return parsed


def _mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {key} metadata is invalid")
    return value
