"""Safe batch-boundary snapshots for long-running Transformer self-play.

The run-state format is intentionally pickle-free. It stores the model tensors,
Adam optimizer tensors, PPO minibatch RNG state, immutable training settings,
and completed-episode progress in one atomic NPZ file. A resumed run therefore
continues from the last committed batch instead of silently resetting optimizer
moments or replaying already-consumed seeds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from app.engine.game import PlayerSpec
from app.training.policy_contract import PolicyHeadSizes
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_self_play import TorchSelfPlayTrainingLoop
from app.training.torch_trainer import TorchPPOConfig

_RUN_STATE_VERSION = 1
_METADATA_KEY = "__metadata__"
_MODEL_PREFIX = "model::"
_OPTIMIZER_PREFIX = "optimizer::"
_TRAINER_GENERATOR_KEY = "trainer_generator"


@dataclass(frozen=True)
class TorchRunProgress:
    completed_episodes: int
    batch_number: int
    base_seed: int
    parent_policy_id: str | None = None
    next_pool_generation: int | None = None

    def __post_init__(self) -> None:
        if self.completed_episodes < 0:
            raise ValueError("completed_episodes cannot be negative")
        if self.batch_number < 0:
            raise ValueError("batch_number cannot be negative")
        if self.next_pool_generation is not None and self.next_pool_generation < 0:
            raise ValueError("next_pool_generation cannot be negative")


def save_torch_run_state(
    loop: TorchSelfPlayTrainingLoop,
    progress: TorchRunProgress,
    path: str | Path,
) -> None:
    """Atomically save a self-contained training snapshot without pickle."""

    optimizer_state, trainer_generator = loop.optimizer.checkpoint_state()
    raw_state = optimizer_state.get("state")
    raw_groups = optimizer_state.get("param_groups")
    if not isinstance(raw_state, dict) or not isinstance(raw_groups, list):
        raise ValueError("unexpected optimizer state structure")

    metadata = {
        "version": _RUN_STATE_VERSION,
        "model_config": asdict(loop.model.config),
        "policy_sizes": asdict(loop.model.sizes),
        "ppo_config": asdict(loop.optimizer.config),
        "runtime": {
            "max_discussion_ticks": loop.max_discussion_ticks,
            "max_parallel_games": loop.max_parallel_games,
            "temperature": loop.temperature,
        },
        "progress": asdict(progress),
        "optimizer_param_groups": _json_safe(raw_groups),
    }
    encoded_metadata = json.dumps(metadata, sort_keys=True).encode("utf-8")
    arrays: dict[str, np.ndarray[Any, Any]] = {
        _METADATA_KEY: np.frombuffer(encoded_metadata, dtype=np.uint8).copy(),
        _TRAINER_GENERATOR_KEY: trainer_generator.detach().cpu().numpy().copy(),
    }
    for name, tensor in loop.model.state_dict().items():
        arrays[f"{_MODEL_PREFIX}{name}"] = tensor.detach().cpu().numpy().copy()

    for raw_index, fields in raw_state.items():
        if not isinstance(raw_index, int) or not isinstance(fields, dict):
            raise ValueError("unexpected optimizer parameter state")
        for field_name, value in fields.items():
            if not isinstance(field_name, str) or not isinstance(value, Tensor):
                raise ValueError("optimizer state must contain named tensors only")
            arrays[
                f"{_OPTIMIZER_PREFIX}{raw_index}::{field_name}"
            ] = value.detach().cpu().numpy().copy()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    temporary.replace(destination)


def load_torch_run_state(
    path: str | Path,
    player_specs: list[PlayerSpec],
    *,
    device: str | torch.device = "cpu",
) -> tuple[TorchSelfPlayTrainingLoop, TorchRunProgress]:
    """Restore model, learner, RNG stream, runtime settings, and progress."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = _decode_metadata(archive)
        model = TorchTransformerPolicy(
            TransformerPolicyConfig(**_mapping(metadata, "model_config")),
            sizes=PolicyHeadSizes(**_mapping(metadata, "policy_sizes")),
        ).to(device)
        _restore_model_tensors(model, archive)

        ppo_config = TorchPPOConfig(**_mapping(metadata, "ppo_config"))
        runtime = _mapping(metadata, "runtime")
        max_discussion_ticks = _required_int(runtime, "max_discussion_ticks")
        max_parallel_games = _required_int(runtime, "max_parallel_games")
        temperature = _required_float(runtime, "temperature")
        loop = TorchSelfPlayTrainingLoop(
            player_specs,
            model=model,
            ppo_config=ppo_config,
            max_discussion_ticks=max_discussion_ticks,
            max_parallel_games=max_parallel_games,
            temperature=temperature,
        )

        optimizer_state = {
            "state": _optimizer_tensor_state(archive),
            "param_groups": _optimizer_param_groups(metadata),
        }
        generator_array = np.array(archive[_TRAINER_GENERATOR_KEY], copy=True)
        if generator_array.dtype != np.uint8 or generator_array.ndim != 1:
            raise ValueError("run-state trainer generator has invalid encoding")
        loop.optimizer.restore_checkpoint_state(
            optimizer_state,
            torch.from_numpy(generator_array),
        )
        progress = _progress(metadata)
    return loop, progress


def _restore_model_tensors(model: TorchTransformerPolicy, archive: Any) -> None:
    expected = set(model.state_dict())
    available = {
        key.removeprefix(_MODEL_PREFIX)
        for key in archive.files
        if key.startswith(_MODEL_PREFIX)
    }
    if expected != available:
        missing = sorted(expected - available)
        extra = sorted(available - expected)
        raise ValueError(
            f"run-state model tensors do not match; missing={missing}, extra={extra}"
        )

    state: dict[str, Tensor] = {}
    for name, template in model.state_dict().items():
        array = np.array(archive[f"{_MODEL_PREFIX}{name}"], copy=True)
        tensor = torch.from_numpy(array).to(dtype=template.dtype)
        if tensor.shape != template.shape:
            raise ValueError(f"run-state model tensor {name} has unexpected shape")
        state[name] = tensor
    model.load_state_dict(state, strict=True)


def _optimizer_tensor_state(archive: Any) -> dict[int, dict[str, Tensor]]:
    state: dict[int, dict[str, Tensor]] = {}
    for key in archive.files:
        if not key.startswith(_OPTIMIZER_PREFIX):
            continue
        payload = key.removeprefix(_OPTIMIZER_PREFIX)
        raw_index, separator, field_name = payload.partition("::")
        if not separator or not field_name:
            raise ValueError("invalid optimizer tensor key in run state")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("invalid optimizer parameter index in run state") from exc
        array = np.array(archive[key], copy=True)
        state.setdefault(index, {})[field_name] = torch.from_numpy(array)
    return state


def _optimizer_param_groups(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("optimizer_param_groups")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("run-state optimizer param groups are invalid")
    return [dict(item) for item in raw]


def _progress(metadata: dict[str, Any]) -> TorchRunProgress:
    raw = _mapping(metadata, "progress")
    parent = raw.get("parent_policy_id")
    next_generation = raw.get("next_pool_generation")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("run-state parent_policy_id is invalid")
    if next_generation is not None and not isinstance(next_generation, int):
        raise ValueError("run-state next_pool_generation is invalid")
    return TorchRunProgress(
        completed_episodes=_required_int(raw, "completed_episodes"),
        batch_number=_required_int(raw, "batch_number"),
        base_seed=_required_int(raw, "base_seed"),
        parent_policy_id=parent,
        next_pool_generation=next_generation,
    )


def _decode_metadata(archive: Any) -> dict[str, Any]:
    if _METADATA_KEY not in archive.files:
        raise ValueError("run-state metadata is missing")
    if _TRAINER_GENERATOR_KEY not in archive.files:
        raise ValueError("run-state trainer generator is missing")
    raw = np.asarray(archive[_METADATA_KEY])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("run-state metadata has invalid encoding")
    try:
        parsed = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("run-state metadata is invalid") from exc
    if not isinstance(parsed, dict) or parsed.get("version") != _RUN_STATE_VERSION:
        raise ValueError("unsupported Transformer run-state version")
    return parsed


def _mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"run-state {key} metadata is invalid")
    return value


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"run-state {key} must be an integer")
    return value


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"run-state {key} must be numeric")
    return float(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("optimizer metadata keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    raise ValueError(f"optimizer metadata contains unsupported value {type(value).__name__}")
