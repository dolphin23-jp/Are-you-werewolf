"""Measure three-faction matchup profiles from immutable policy generations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_payoff import (
    PolicyProfile,
    ProfilePayoff,
    PopulationPayoffTable,
    evaluate_policy_profile,
)


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _profile_seed_offset(profile: PolicyProfile) -> int:
    encoded = f"{profile.village}|{profile.werewolf}|{profile.fox}".encode()
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:4], "big")


def _next_seed(base_seed: int, profile: PolicyProfile, games_before: int) -> int:
    return base_seed + _profile_seed_offset(profile) + games_before


def _require_record(
    table: PopulationPayoffTable,
    profile: PolicyProfile,
) -> ProfilePayoff:
    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile measurement did not produce a record")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--extra-games", type=int, default=0)
    parser.add_argument("--uncertainty-prior", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")
    if args.extra_games < 0:
        parser.error("--extra-games cannot be negative")
    if args.uncertainty_prior <= 0:
        parser.error("--uncertainty-prior must be positive")

    pool = NumpyPolicyPool(args.pool_dir)
    if not pool.entries:
        parser.error("--pool-dir contains no policy generations")
    village_ids = pool.policy_ids_for_team(Team.VILLAGE, last=args.last)
    werewolf_ids = pool.policy_ids_for_team(Team.WEREWOLF, last=args.last)
    fox_ids = pool.policy_ids_for_team(Team.FOX, last=args.last)
    if not village_ids or not werewolf_ids or not fox_ids:
        parser.error("each faction must have at least one eligible policy")

    table = PopulationPayoffTable(args.table)
    profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            village_ids,
            werewolf_ids,
            fox_ids,
        )
    )

    for profile in profiles:
        existing = table.get(profile)
        existing_games = existing.games if existing is not None else 0
        missing = max(0, args.games_per_profile - existing_games)
        if missing:
            seed_start = _next_seed(args.seed, profile, existing_games)
            seeds = tuple(range(seed_start, seed_start + missing))
            evaluate_policy_profile(
                _player_specs(),
                pool,
                table,
                profile,
                seeds=seeds,
                max_discussion_ticks=args.discussion_ticks,
            )

    for _ in range(args.extra_games):
        selected = max(
            profiles,
            key=lambda profile: (
                _require_record(table, profile).max_posterior_payoff_std(
                    prior=args.uncertainty_prior
                ),
                -_require_record(table, profile).games,
                profile,
            ),
        )
        before = _require_record(table, selected)
        evaluate_policy_profile(
            _player_specs(),
            pool,
            table,
            selected,
            seeds=(_next_seed(args.seed, selected, before.games),),
            max_discussion_ticks=args.discussion_ticks,
        )

    for profile in profiles:
        record = _require_record(table, profile)
        print(
            f"profile={profile.village}/{profile.werewolf}/{profile.fox} "
            f"games={record.games} wins(v/w/f)="
            f"{record.village_wins}/{record.werewolf_wins}/{record.fox_wins} "
            f"draws={record.draws} payoffs(v/w/f)="
            f"{record.mean_payoff(Team.VILLAGE):.3f}/"
            f"{record.mean_payoff(Team.WEREWOLF):.3f}/"
            f"{record.mean_payoff(Team.FOX):.3f} "
            f"uncertainty={record.max_posterior_payoff_std(prior=args.uncertainty_prior):.4f}"
        )

    print(
        f"measured_profiles={len(profiles)} extra_games={args.extra_games} "
        f"village={','.join(village_ids)} "
        f"werewolf={','.join(werewolf_ids)} "
        f"fox={','.join(fox_ids)} table={args.table}"
    )


if __name__ == "__main__":
    main()
