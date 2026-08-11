"""Export and extract immutable post-hoc strategy-analysis snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

_SNAPSHOT_VERSION = 1
_TEAM_KEYS = ("village", "werewolf", "fox")


def export_strategy_snapshot(
    *,
    pool_dir: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
    iteration: int | None = None,
    source_label: str | None = None,
    git_commit: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy only frozen policy files and metadata into a portable tar.gz.

    The source pool and population run are read-only. The function does not import
    PyTorch and can therefore run alongside a GPU training process without taking
    accelerator memory.
    """

    pool_root = Path(pool_dir)
    run_root = Path(run_dir)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not pool_root.is_dir():
        raise ValueError(f"missing pool directory: {pool_root}")
    if not run_root.is_dir():
        raise ValueError(f"missing run directory: {run_root}")

    source_manifest = _load_json(pool_root / "manifest.json")
    population, source = _resolve_population(run_root, iteration=iteration)
    selected_ids = tuple(
        dict.fromkeys(
            policy_id
            for team in _TEAM_KEYS
            for policy_id in population[team]
        )
    )

    entries_raw = source_manifest.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError("pool manifest is missing entries")
    entries_by_id = {
        str(entry.get("policy_id")): entry
        for entry in entries_raw
        if isinstance(entry, dict) and isinstance(entry.get("policy_id"), str)
    }
    selected_entries: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for policy_id in selected_ids:
        entry = entries_by_id.get(policy_id)
        if entry is None:
            raise ValueError(f"policy {policy_id} is missing from pool manifest")
        checkpoint = entry.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ValueError(f"policy {policy_id} has an invalid checkpoint path")
        checkpoint_path = pool_root / checkpoint
        if not checkpoint_path.is_file():
            raise ValueError(f"missing checkpoint for {policy_id}: {checkpoint_path}")
        selected_entries.append(dict(entry))
        checkpoint_hashes[policy_id] = _sha256(checkpoint_path)

    snapshot = {
        "version": _SNAPSHOT_VERSION,
        "source": {
            **source,
            "label": source_label,
            "git_commit": git_commit,
        },
        "population": {team: list(population[team]) for team in _TEAM_KEYS},
        "checkpoint_sha256": checkpoint_hashes,
    }
    subset_manifest = {
        "version": source_manifest.get("version"),
        "entries": selected_entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="werewolf-strategy-snapshot-") as temporary:
        root = Path(temporary)
        pool_out = root / "pool"
        pool_out.mkdir(parents=True)
        (root / "snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (pool_out / "manifest.json").write_text(
            json.dumps(subset_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for entry in selected_entries:
            checkpoint = str(entry["checkpoint"])
            checkpoint_out = pool_out / checkpoint
            checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pool_root / checkpoint, checkpoint_out)

        mode = "w:gz" if output.suffix == ".gz" else "w"
        with tarfile.open(output, mode) as archive:
            archive.add(root / "snapshot.json", arcname="snapshot.json")
            archive.add(pool_out, arcname="pool")

    return snapshot


def extract_strategy_snapshot(
    archive_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Safely extract a snapshot and verify checkpoint hashes."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination, filter="data")

    snapshot = _load_json(destination / "snapshot.json")
    if snapshot.get("version") != _SNAPSHOT_VERSION:
        raise ValueError("unsupported strategy snapshot version")
    hashes = snapshot.get("checkpoint_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("strategy snapshot is missing checkpoint hashes")
    manifest = _load_json(destination / "pool" / "manifest.json")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("snapshot pool manifest is missing entries")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid snapshot pool entry")
        policy_id = entry.get("policy_id")
        checkpoint = entry.get("checkpoint")
        if not isinstance(policy_id, str) or not isinstance(checkpoint, str):
            raise ValueError("invalid snapshot pool policy entry")
        expected = hashes.get(policy_id)
        if not isinstance(expected, str):
            raise ValueError(f"missing hash for snapshot policy {policy_id}")
        actual = _sha256(destination / "pool" / checkpoint)
        if actual != expected:
            raise ValueError(f"checkpoint hash mismatch for {policy_id}")
    return snapshot


def _resolve_population(
    run_root: Path,
    *,
    iteration: int | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    state_path = run_root / "population.run.json"
    state = _load_json(state_path)
    completed = int(state.get("completed_iterations", -1))
    phase = state.get("phase")

    if iteration is not None:
        if iteration <= 0:
            raise ValueError("iteration must be positive")
        summary_path = run_root / f"iteration-{iteration:04d}" / "summary.json"
        summary = _load_json(summary_path)
        raw_population = summary.get("restricted_population")
        population = _parse_population(raw_population)
        return population, {
            "mode": "completed_iteration",
            "iteration": iteration,
            "completed_iterations_at_export": completed,
            "run_phase_at_export": phase,
        }

    if phase in {"measure", "oracle"}:
        raw_population = {
            "village": state.get("village_policy_ids"),
            "werewolf": state.get("werewolf_policy_ids"),
            "fox": state.get("fox_policy_ids"),
        }
        population = _parse_population(raw_population)
        return population, {
            "mode": "active_frozen_population",
            "iteration": completed + 1,
            "completed_iterations_at_export": completed,
            "run_phase_at_export": phase,
        }

    if phase != "idle":
        raise ValueError(f"unsupported population run phase: {phase!r}")
    if completed <= 0:
        raise ValueError("idle population run has no completed iteration to snapshot")
    summary_path = run_root / f"iteration-{completed:04d}" / "summary.json"
    summary = _load_json(summary_path)
    population = _parse_population(summary.get("restricted_population"))
    return population, {
        "mode": "latest_completed_iteration",
        "iteration": completed,
        "completed_iterations_at_export": completed,
        "run_phase_at_export": phase,
    }


def _parse_population(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("population is missing faction policy lists")
    parsed: dict[str, tuple[str, ...]] = {}
    for team in _TEAM_KEYS:
        value = raw.get(team)
        if not isinstance(value, list) or not value or not all(
            isinstance(policy_id, str) for policy_id in value
        ):
            raise ValueError(f"invalid {team} population")
        parsed[team] = tuple(value)
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
