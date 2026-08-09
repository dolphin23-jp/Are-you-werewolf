"""Atomic pickle-free recovery state for multi-faction Transformer oracle cycles."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.torch_historical import TorchHistoricalTrainingLoop
from app.training.torch_historical_run_state import (
    TorchHistoricalRunProgress,
    load_torch_historical_run_state,
    save_torch_historical_run_state,
)
from app.training.torch_oracle_cycle import TorchOracleRunProgress
from app.training.torch_pool import TorchPolicyPool

_ORACLE_RUN_STATE_VERSION = 1
_METADATA_KEY = "__metadata__"
_HISTORICAL_STATE_KEY = "__historical_state__"


def save_torch_oracle_run_state(
    loop: TorchHistoricalTrainingLoop | None,
    progress: TorchOracleRunProgress,
    path: str | Path,
) -> None:
    """Atomically save outer oracle progress plus the exact active learner state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": _ORACLE_RUN_STATE_VERSION,
        "kind": "psro_oracle_cycle",
        "progress": {
            "teams": [team.value for team in progress.teams],
            "team_index": progress.team_index,
            "completed_episodes": progress.completed_episodes,
            "episodes_per_oracle": progress.episodes_per_oracle,
            "oracle_batch_size": progress.oracle_batch_size,
            "base_seed": progress.base_seed,
            "opponent_seed": progress.opponent_seed,
            "trainer_seed": progress.trainer_seed,
            "next_pool_generation": progress.next_pool_generation,
            "active_parent_policy_id": progress.active_parent_policy_id,
            "completed_policy_ids": list(progress.completed_policy_ids),
            "active_wins": progress.active_wins,
            "active_losses": progress.active_losses,
            "active_draws": progress.active_draws,
            "active_days_sum": progress.active_days_sum,
            "active_decisions_sum": progress.active_decisions_sum,
        },
    }
    encoded_metadata = json.dumps(metadata, sort_keys=True).encode("utf-8")
    arrays: dict[str, np.ndarray[Any, Any]] = {
        _METADATA_KEY: np.frombuffer(encoded_metadata, dtype=np.uint8).copy(),
    }

    if progress.is_complete:
        if loop is not None:
            raise ValueError("completed oracle cycle cannot save an active learner")
    else:
        if loop is None:
            raise ValueError("active oracle cycle requires a learner state")
        historical_progress = _historical_progress(progress)
        arrays[_HISTORICAL_STATE_KEY] = np.frombuffer(
            _serialize_historical_state(
                loop,
                historical_progress,
                destination.parent,
            ),
            dtype=np.uint8,
        ).copy()

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
    temporary.replace(destination)


def load_torch_oracle_run_state(
    path: str | Path,
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TorchHistoricalTrainingLoop | None, TorchOracleRunProgress]:
    """Restore the exact active oracle sub-batch boundary, or a completed cycle."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = _decode_metadata(archive)
        progress = _progress(metadata)
        if progress.is_complete:
            if _HISTORICAL_STATE_KEY in archive.files:
                raise ValueError("completed oracle run-state contains active learner data")
            return None, progress
        if _HISTORICAL_STATE_KEY not in archive.files:
            raise ValueError("active oracle run-state is missing learner data")
        inner = np.asarray(archive[_HISTORICAL_STATE_KEY])
        if inner.dtype != np.uint8 or inner.ndim != 1:
            raise ValueError("oracle historical learner state has invalid encoding")
        inner_bytes = inner.tobytes()

    loop, historical_progress = _deserialize_historical_state(
        inner_bytes,
        player_specs,
        pool,
        device=device,
        directory=source.parent,
    )
    _validate_historical_progress(loop, historical_progress, progress)
    return loop, progress


def _historical_progress(
    progress: TorchOracleRunProgress,
) -> TorchHistoricalRunProgress:
    team = progress.active_team
    if team is None or progress.active_parent_policy_id is None:
        raise ValueError("completed oracle cycle has no historical learner progress")
    return TorchHistoricalRunProgress(
        completed_batches=progress.completed_batches,
        base_seed=progress.active_team_base_seed,
        episodes_per_batch=progress.oracle_batch_size,
        requested_teams=(team,),
        parent_policy_id=progress.active_parent_policy_id,
        next_pool_generation=progress.next_pool_generation,
    )


def _validate_historical_progress(
    loop: TorchHistoricalTrainingLoop,
    historical: TorchHistoricalRunProgress,
    progress: TorchOracleRunProgress,
) -> None:
    team = progress.active_team
    if team is None:
        raise ValueError("completed oracle cycle cannot contain historical progress")
    if loop.opponent_strategy is None:
        raise ValueError("PSRO oracle run-state requires a fixed opponent strategy")
    if historical.requested_teams != (team,):
        raise ValueError("oracle learner faction does not match outer progress")
    if historical.completed_batches != progress.completed_batches:
        raise ValueError("oracle learner batch count does not match outer progress")
    if historical.base_seed != progress.active_team_base_seed:
        raise ValueError("oracle learner seed base does not match outer progress")
    if historical.episodes_per_batch != progress.oracle_batch_size:
        raise ValueError("oracle learner batch size does not match outer progress")
    if historical.parent_policy_id != progress.active_parent_policy_id:
        raise ValueError("oracle learner parent does not match outer progress")
    if historical.next_pool_generation != progress.next_pool_generation:
        raise ValueError("oracle learner pool boundary does not match outer progress")
    if (
        progress.completed_episodes < progress.episodes_per_oracle
        and historical.next_start_seed != progress.next_start_seed
    ):
        raise ValueError("oracle learner next seed does not match outer progress")


def _serialize_historical_state(
    loop: TorchHistoricalTrainingLoop,
    progress: TorchHistoricalRunProgress,
    directory: Path,
) -> bytes:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="oracle-inner-",
        suffix=".npz",
        dir=directory,
    )
    os.close(descriptor)
    inner_path = Path(raw_path)
    try:
        save_torch_historical_run_state(loop, progress, inner_path)
        return inner_path.read_bytes()
    finally:
        inner_path.unlink(missing_ok=True)
        inner_path.with_suffix(inner_path.suffix + ".tmp").unlink(missing_ok=True)


def _deserialize_historical_state(
    payload: bytes,
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    *,
    device: str | torch.device,
    directory: Path,
) -> tuple[TorchHistoricalTrainingLoop, TorchHistoricalRunProgress]:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="oracle-inner-load-",
        suffix=".npz",
        dir=directory,
    )
    os.close(descriptor)
    inner_path = Path(raw_path)
    try:
        inner_path.write_bytes(payload)
        return load_torch_historical_run_state(
            inner_path,
            player_specs,
            pool,
            device=device,
        )
    finally:
        inner_path.unlink(missing_ok=True)


def _decode_metadata(archive: Any) -> dict[str, Any]:
    if _METADATA_KEY not in archive.files:
        raise ValueError("oracle run-state metadata is missing")
    raw = np.asarray(archive[_METADATA_KEY])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("oracle run-state metadata has invalid encoding")
    try:
        parsed = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("oracle run-state metadata is invalid") from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("version") != _ORACLE_RUN_STATE_VERSION
        or parsed.get("kind") != "psro_oracle_cycle"
    ):
        raise ValueError("unsupported oracle run-state version")
    return parsed


def _progress(metadata: dict[str, Any]) -> TorchOracleRunProgress:
    raw = metadata.get("progress")
    if not isinstance(raw, dict):
        raise ValueError("oracle run-state progress metadata is invalid")
    raw_teams = raw.get("teams")
    if not isinstance(raw_teams, list) or not all(
        isinstance(item, str) for item in raw_teams
    ):
        raise ValueError("oracle run-state teams are invalid")
    try:
        teams = tuple(Team(item) for item in raw_teams)
    except ValueError as exc:
        raise ValueError("oracle run-state teams are invalid") from exc
    raw_completed_ids = raw.get("completed_policy_ids")
    if not isinstance(raw_completed_ids, list) or not all(
        isinstance(item, str) for item in raw_completed_ids
    ):
        raise ValueError("oracle completed policy IDs are invalid")
    parent = raw.get("active_parent_policy_id")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("oracle active parent policy ID is invalid")
    return TorchOracleRunProgress(
        teams=teams,
        team_index=_required_int(raw, "team_index"),
        completed_episodes=_required_int(raw, "completed_episodes"),
        episodes_per_oracle=_required_int(raw, "episodes_per_oracle"),
        oracle_batch_size=_required_int(raw, "oracle_batch_size"),
        base_seed=_required_int(raw, "base_seed"),
        opponent_seed=_required_int(raw, "opponent_seed"),
        trainer_seed=_required_int(raw, "trainer_seed"),
        next_pool_generation=_required_int(raw, "next_pool_generation"),
        active_parent_policy_id=parent,
        completed_policy_ids=tuple(raw_completed_ids),
        active_wins=_required_int(raw, "active_wins"),
        active_losses=_required_int(raw, "active_losses"),
        active_draws=_required_int(raw, "active_draws"),
        active_days_sum=_required_float(raw, "active_days_sum"),
        active_decisions_sum=_required_float(raw, "active_decisions_sum"),
    )


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"oracle run-state {key} must be an integer")
    return value


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"oracle run-state {key} must be numeric")
    return float(value)
