from pathlib import Path

import pytest

from app.training.population_payoff import PolicyProfile, PopulationPayoffTable, ProfilePayoff
from app.training.population_shards import (
    merge_payoff_records,
    select_profile_shard,
    write_payoff_records,
)


def _record(
    profile: PolicyProfile,
    *,
    games: int,
    village: int,
    werewolf: int,
    fox: int,
    draws: int,
    days: int,
) -> ProfilePayoff:
    return ProfilePayoff(profile, games, village, werewolf, fox, draws, days)


def test_profile_shards_are_disjoint_and_cover_all_profiles() -> None:
    profiles = tuple(PolicyProfile(f"v{i}", "w", "f") for i in range(11))
    shards = [
        select_profile_shard(profiles, shard_count=4, shard_index=index)
        for index in range(4)
    ]

    assert tuple(item for shard in shards for item in shard) != profiles
    assert set().union(*(set(shard) for shard in shards)) == set(profiles)
    assert sum(len(shard) for shard in shards) == len(profiles)
    assert shards[0] == profiles[0::4]
    assert shards[3] == profiles[3::4]


def test_profile_shard_rejects_invalid_indices() -> None:
    profile = PolicyProfile("v", "w", "f")
    with pytest.raises(ValueError, match="shard_count"):
        select_profile_shard((profile,), shard_count=0, shard_index=0)
    with pytest.raises(ValueError, match="shard_index"):
        select_profile_shard((profile,), shard_count=2, shard_index=2)


def test_merge_payoff_records_prefers_valid_cumulative_extension() -> None:
    profile = PolicyProfile("v", "w", "f")
    older = _record(
        profile,
        games=2,
        village=1,
        werewolf=1,
        fox=0,
        draws=0,
        days=7,
    )
    newer = _record(
        profile,
        games=4,
        village=2,
        werewolf=1,
        fox=1,
        draws=0,
        days=15,
    )

    assert merge_payoff_records((newer,), (older,)) == (newer,)
    assert merge_payoff_records((older,), (newer,)) == (newer,)


def test_merge_payoff_records_rejects_conflicting_histories() -> None:
    profile = PolicyProfile("v", "w", "f")
    first = _record(
        profile,
        games=2,
        village=2,
        werewolf=0,
        fox=0,
        draws=0,
        days=6,
    )
    conflict = _record(
        profile,
        games=2,
        village=1,
        werewolf=1,
        fox=0,
        draws=0,
        days=6,
    )

    with pytest.raises(ValueError, match="conflicting equal-length"):
        merge_payoff_records((first,), (conflict,))


def test_write_payoff_records_round_trips_population_table(tmp_path: Path) -> None:
    profile = PolicyProfile("v", "w", "f")
    record = _record(
        profile,
        games=3,
        village=1,
        werewolf=1,
        fox=0,
        draws=1,
        days=12,
    )
    path = tmp_path / "shard.json"

    write_payoff_records(path, (record,))

    assert PopulationPayoffTable(path).records == (record,)
