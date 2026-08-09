"""Pickle-free batch-boundary recovery for Transformer historical self-play."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy, PopulationWeight
from app.training.policy_contract import PolicyHeadSizes
from app.training.torch_historical import TorchHistoricalTrainingLoop
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import TorchPPOConfig

_HISTORICAL_RUN_STATE_VERSION = 1
_METADATA_KEY = "__metadata__"
_MODEL_PREFIX = "model::"
_OPTIMIZER_PREFIX = "optimizer::"
_TRAINER_GENERATOR_KEY = "trainer_generator"


@dataclass(frozen=True)
class TorchHistoricalRunProgress:
    """Deterministic progress at a completed historical learner-batch boundary."""

    completed_batches: int
    base_seed: int
    episodes_per_batch: int
    requested_teams: tuple[Team, ...]
    parent_policy_id: str | None = None
    next_pool_generation: int | None = None

    def __post_init__(self) -> None:
        if self.completed_batches < 0:
            raise ValueError("completed_batches cannot be negative")
        if self.episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")
        if not self.requested_teams:
            raise ValueError("requested_teams cannot be empty")
        if self.next_pool_generation is not None and self.next_pool_generation < 0:
            raise ValueError("next_pool_generation cannot be negative")

    @property
    def next_learner_team(self) -> Team:
        return self.requested_teams[
            self.completed_batches % len(self.requested_teams)
        ]

    @property
    def next_start_seed(self) -> int:
        return self.base_seed + self.completed_batches * self.episodes_per_batch


def save_torch_historical_run_state(
    loop: TorchHistoricalTrainingLoop,
    progress: TorchHistoricalRunProgress,
    path: str | Path,
) -> None:
    """Atomically save model, optimizer, both RNG streams and historical progress."""

    optimizer_state, trainer_generator = loop.optimizer.checkpoint_state()
    raw_state = optimizer_state.get("state")
    raw_groups = optimizer_state.get("param_groups")
    if not isinstance(raw_state, dict) or not isinstance(raw_groups, list):
        raise ValueError("unexpected optimizer state structure")

    metadata = {
        "version": _HISTORICAL_RUN_STATE_VERSION,
        "kind": "historical",
        "model_config": asdict(loop.model.config),
        "policy_sizes": asdict(loop.model.sizes),
        "ppo_config": asdict(loop.optimizer.config),
        "runtime": {
            "max_discussion_ticks": loop.max_discussion_ticks,
            "max_parallel_games": loop.max_parallel_games,
            "max_inference_batch_size": loop.max_inference_batch_size,
            "temperature": loop.temperature,
        },
        "progress": {
            "completed_batches": progress.completed_batches,
            "base_seed": progress.base_seed,
            "episodes_per_batch": progress.episodes_per_batch,
            "requested_teams": [team.value for team in progress.requested_teams],
            "parent_policy_id": progress.parent_policy_id,
            "next_pool_generation": progress.next_pool_generation,
        },
        "opponent_strategy": _strategy_payload(loop.opponent_strategy),
        "opponent_rng_state": _json_safe(loop.checkpoint_opponent_rng_state()),
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


def load_torch_historical_run_state(
    path: str | Path,
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TorchHistoricalTrainingLoop, TorchHistoricalRunProgress]:
    """Restore an exact historical learner-batch boundary snapshot."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = _decode_metadata(archive)
        model = TorchTransformerPolicy(
            TransformerPolicyConfig(**_mapping(metadata, "model_config")),
            sizes=PolicyHeadSizes(**_mapping(metadata, "policy_sizes")),
        ).to(device)
        _restore_model_tensors(model, archive)

        runtime = _mapping(metadata, "runtime")
        strategy = _strategy_from_payload(metadata.get("opponent_strategy"))
        loop = TorchHistoricalTrainingLoop(
            player_specs,
            model,
            pool,
            opponent_strategy=strategy,
            opponent_seed=0,
            ppo_config=TorchPPOConfig(**_mapping(metadata, "ppo_config")),
            max_discussion_ticks=_required_int(runtime, "max_discussion_ticks"),
            max_parallel_games=_required_int(runtime, "max_parallel_games"),
            max_inference_batch_size=_optional_positive_int(
                runtime,
                "max_inference_batch_size",
            ),
            temperature=_required_float(runtime, "temperature"),
            trainer_seed=0,
        )

        optimizer_state = {
            "state": _optimizer_tensor_state(archive),
            "param_groups": _optimizer_param_groups(metadata),
        }
        generator_array = np.array(archive[_TRAINER_GENERATOR_KEY], copy=True)
        if generator_array.dtype != np.uint8 or generator_array.ndim != 1:
            raise ValueError("historical run-state trainer generator is invalid")
        loop.optimizer.restore_checkpoint_state(
            optimizer_state,
            torch.from_numpy(generator_array),
        )
        loop.restore_opponent_rng_state(
            _tuple_tree(metadata.get("opponent_rng_state"))
        )
        progress = _progress(metadata)
    return loop, progress


def _strategy_payload(
    strategy: PopulationMetaStrategy | None,
) -> dict[str, Any] | None:
    if strategy is None:
        return None
    return {
        "temperature": strategy.temperature,
        "weights": {
            team.value: [
                {"policy_id": item.policy_id, "weight": item.weight}
                for item in strategy.weights(team)
            ]
            for team in Team
        },
    }


def _strategy_from_payload(raw: Any) -> PopulationMetaStrategy | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("historical run-state opponent strategy is invalid")
    temperature = raw.get("temperature")
    weights = raw.get("weights")
    if not isinstance(temperature, (int, float)) or not isinstance(weights, dict):
        raise ValueError("historical run-state opponent strategy is invalid")

    by_team: dict[Team, tuple[PopulationWeight, ...]] = {}
    for team in Team:
        items = weights.get(team.value)
        if not isinstance(items, list):
            raise ValueError("historical run-state opponent weights are invalid")
        parsed: list[PopulationWeight] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("historical run-state opponent weight is invalid")
            policy_id = item.get("policy_id")
            weight = item.get("weight")
            if not isinstance(policy_id, str) or not isinstance(weight, (int, float)):
                raise ValueError("historical run-state opponent weight is invalid")
            parsed.append(PopulationWeight(policy_id, float(weight)))
        by_team[team] = tuple(parsed)
    return PopulationMetaStrategy(by_team=by_team, temperature=float(temperature))


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
            f"historical run-state model tensors do not match; "
            f"missing={missing}, extra={extra}"
        )

    state: dict[str, Tensor] = {}
    for name, template in model.state_dict().items():
        array = np.array(archive[f"{_MODEL_PREFIX}{name}"], copy=True)
        tensor = torch.from_numpy(array).to(dtype=template.dtype)
        if tensor.shape != template.shape:
            raise ValueError(
                f"historical run-state model tensor {name} has unexpected shape"
            )
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
            raise ValueError("invalid historical optimizer tensor key")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("invalid historical optimizer parameter index") from exc
        state.setdefault(index, {})[field_name] = torch.from_numpy(
            np.array(archive[key], copy=True)
        )
    return state


def _optimizer_param_groups(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("optimizer_param_groups")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("historical optimizer param groups are invalid")
    return [dict(item) for item in raw]


def _progress(metadata: dict[str, Any]) -> TorchHistoricalRunProgress:
    raw = _mapping(metadata, "progress")
    teams = raw.get("requested_teams")
    if not isinstance(teams, list) or not teams or not all(
        isinstance(item, str) for item in teams
    ):
        raise ValueError("historical requested_teams are invalid")
    try:
        requested_teams = tuple(Team(item) for item in teams)
    except ValueError as exc:
        raise ValueError("historical requested_teams are invalid") from exc

    parent = raw.get("parent_policy_id")
    next_generation = raw.get("next_pool_generation")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("historical parent_policy_id is invalid")
    if next_generation is not None and not isinstance(next_generation, int):
        raise ValueError("historical next_pool_generation is invalid")
    return TorchHistoricalRunProgress(
        completed_batches=_required_int(raw, "completed_batches"),
        base_seed=_required_int(raw, "base_seed"),
        episodes_per_batch=_required_int(raw, "episodes_per_batch"),
        requested_teams=requested_teams,
        parent_policy_id=parent,
        next_pool_generation=next_generation,
    )


def _decode_metadata(archive: Any) -> dict[str, Any]:
    if _METADATA_KEY not in archive.files:
        raise ValueError("historical run-state metadata is missing")
    if _TRAINER_GENERATOR_KEY not in archive.files:
        raise ValueError("historical run-state trainer generator is missing")
    raw = np.asarray(archive[_METADATA_KEY])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("historical run-state metadata has invalid encoding")
    try:
        parsed = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("historical run-state metadata is invalid") from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("version") != _HISTORICAL_RUN_STATE_VERSION
        or parsed.get("kind") != "historical"
    ):
        raise ValueError("unsupported historical run-state version")
    return parsed


def _mapping(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"historical run-state {key} metadata is invalid")
    return value


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"historical run-state {key} must be an integer")
    return value


def _optional_positive_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"historical run-state {key} must be a positive integer or null"
        )
    return value


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"historical run-state {key} must be numeric")
    return float(value)


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("historical optimizer metadata keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    raise ValueError(
        f"historical run-state contains unsupported value {type(value).__name__}"
    )
