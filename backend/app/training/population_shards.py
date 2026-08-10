"""Helpers for crash-safe sharded population payoff evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from app.training.population_payoff import PolicyProfile, ProfilePayoff

_TABLE_VERSION = 1


def select_profile_shard(
    profiles: Sequence[PolicyProfile],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[PolicyProfile, ...]:
    """Return one deterministic round-robin profile shard."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return tuple(profiles[shard_index::shard_count])


def merge_payoff_records(
    *groups: Iterable[ProfilePayoff],
) -> tuple[ProfilePayoff, ...]:
    """Merge aggregate records only when one side is a valid cumulative extension.

    Sharded workers start from a snapshot of the master table and append terminal
    results. Therefore a newer record must contain every count from the older
    record. Equal-length but different records indicate concurrent/conflicting
    evaluation and are rejected rather than guessed around.
    """

    merged: dict[PolicyProfile, ProfilePayoff] = {}
    for group in groups:
        for record in group:
            _validate_record(record)
            current = merged.get(record.profile)
            if current is None:
                merged[record.profile] = record
                continue
            if record == current:
                continue
            if record.games == current.games:
                raise ValueError(f"conflicting equal-length payoff record: {record.profile}")
            newer, older = (
                (record, current) if record.games > current.games else (current, record)
            )
            if not _extends(newer, older):
                raise ValueError(f"payoff records are not cumulative extensions: {record.profile}")
            merged[record.profile] = newer
    return tuple(sorted(merged.values(), key=lambda item: item.profile))


def write_payoff_records(path: str | Path, records: Iterable[ProfilePayoff]) -> None:
    """Atomically write aggregate payoff records in PopulationPayoffTable v1 format."""

    destination = Path(path)
    validated = merge_payoff_records(records)
    payload = {
        "version": _TABLE_VERSION,
        "records": [
            {
                **asdict(record),
                "profile": asdict(record.profile),
            }
            for record in validated
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _validate_record(record: ProfilePayoff) -> None:
    counts = (
        record.village_wins,
        record.werewolf_wins,
        record.fox_wins,
        record.draws,
    )
    if record.games < 0 or any(value < 0 for value in counts):
        raise ValueError(f"negative payoff count: {record.profile}")
    if sum(counts) != record.games:
        raise ValueError(f"terminal counts do not equal games: {record.profile}")
    if record.total_days < 0:
        raise ValueError(f"negative total_days: {record.profile}")


def _extends(newer: ProfilePayoff, older: ProfilePayoff) -> bool:
    return (
        newer.profile == older.profile
        and newer.games >= older.games
        and newer.village_wins >= older.village_wins
        and newer.werewolf_wins >= older.werewolf_wins
        and newer.fox_wins >= older.fox_wins
        and newer.draws >= older.draws
        and newer.total_days >= older.total_days
    )
