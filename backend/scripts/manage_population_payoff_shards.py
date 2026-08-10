"""Prepare, inspect, and merge crash-safe population payoff shard tables."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.population_shards import (
    merge_payoff_records,
    select_profile_shard,
    write_payoff_records,
)
from app.training.torch_pool import TorchPolicyPool


def _profiles(pool: TorchPolicyPool, last: int) -> tuple[PolicyProfile, ...]:
    village = pool.policy_ids_for_team(Team.VILLAGE, last=last)
    werewolf = pool.policy_ids_for_team(Team.WEREWOLF, last=last)
    fox = pool.policy_ids_for_team(Team.FOX, last=last)
    if not village or not werewolf or not fox:
        raise ValueError("each faction must have at least one eligible policy")
    return tuple(
        PolicyProfile(village_id, werewolf_id, fox_id)
        for village_id, werewolf_id, fox_id in itertools.product(village, werewolf, fox)
    )


def _shard_path(directory: Path, shard_index: int) -> Path:
    return directory / f"shard-{shard_index:02d}.json"


def _prepare(
    master: PopulationPayoffTable,
    profiles: tuple[PolicyProfile, ...],
    shard_dir: Path,
    shard_count: int,
) -> None:
    master_by_profile = {record.profile: record for record in master.records}
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard_index in range(shard_count):
        selected = set(
            select_profile_shard(
                profiles,
                shard_count=shard_count,
                shard_index=shard_index,
            )
        )
        path = _shard_path(shard_dir, shard_index)
        existing = PopulationPayoffTable(path).records if path.exists() else ()
        unexpected = [record.profile for record in existing if record.profile not in selected]
        if unexpected:
            raise ValueError(f"shard {shard_index} contains profiles outside its assignment")
        seeded = tuple(
            record for profile, record in master_by_profile.items() if profile in selected
        )
        merged = merge_payoff_records(seeded, existing)
        write_payoff_records(path, merged)
        print(
            f"prepared shard={shard_index}/{shard_count} profiles={len(selected)} "
            f"saved_games={sum(record.games for record in merged)} table={path}"
        )


def _merged_records(
    master: PopulationPayoffTable,
    profiles: tuple[PolicyProfile, ...],
    shard_dir: Path,
    shard_count: int,
):
    groups = [master.records]
    for shard_index in range(shard_count):
        selected = set(
            select_profile_shard(
                profiles,
                shard_count=shard_count,
                shard_index=shard_index,
            )
        )
        path = _shard_path(shard_dir, shard_index)
        if not path.exists():
            continue
        records = PopulationPayoffTable(path).records
        unexpected = [record.profile for record in records if record.profile not in selected]
        if unexpected:
            raise ValueError(f"shard {shard_index} contains profiles outside its assignment")
        groups.append(records)
    return merge_payoff_records(*groups)


def _print_status(records, profiles: tuple[PolicyProfile, ...], games_per_profile: int) -> None:
    selected = set(profiles)
    relevant = [record for record in records if record.profile in selected]
    games = sum(min(record.games, games_per_profile) for record in relevant)
    target = len(profiles) * games_per_profile
    completed = sum(record.games >= games_per_profile for record in relevant)
    print(
        f"games={games}/{target} ({100.0 * games / target:.1f}%) "
        f"profiles={len(relevant)}/{len(profiles)} complete_profiles={completed}/{len(profiles)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "status", "merge"))
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--games-per-profile", type=int, required=True)
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be positive")
    if args.shards <= 0:
        parser.error("--shards must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")

    pool = TorchPolicyPool(args.pool_dir, device="cpu")
    profiles = _profiles(pool, args.last)
    master = PopulationPayoffTable(args.table)

    if args.action == "prepare":
        _prepare(master, profiles, args.shard_dir, args.shards)
        return

    merged = _merged_records(master, profiles, args.shard_dir, args.shards)
    _print_status(merged, profiles, args.games_per_profile)
    if args.action == "status":
        return

    by_profile = {record.profile: record for record in merged}
    incomplete = [
        profile
        for profile in profiles
        if profile not in by_profile or by_profile[profile].games < args.games_per_profile
    ]
    if incomplete:
        raise SystemExit(f"cannot merge incomplete shards: {len(incomplete)} profiles unfinished")
    write_payoff_records(args.table, merged)
    print(f"merged_table={args.table}")


if __name__ == "__main__":
    main()
