"""Measure three-faction matchup profiles from immutable policy generations."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffTable,
    evaluate_policy_profile,
)


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")

    pool = NumpyPolicyPool(args.pool_dir)
    if not pool.entries:
        parser.error("--pool-dir contains no policy generations")
    selected = tuple(entry.policy_id for entry in pool.entries[-args.last :])
    table = PopulationPayoffTable(args.table)
    profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(selected, repeat=3)
    )

    for profile_index, profile in enumerate(profiles):
        existing = table.get(profile)
        existing_games = existing.games if existing is not None else 0
        missing = max(0, args.games_per_profile - existing_games)
        if missing:
            seed_start = args.seed + profile_index * 100000 + existing_games
            seeds = tuple(range(seed_start, seed_start + missing))
            record = evaluate_policy_profile(
                _player_specs(),
                pool,
                table,
                profile,
                seeds=seeds,
                max_discussion_ticks=args.discussion_ticks,
            )
        else:
            record = existing
        if record is None:
            raise RuntimeError("profile measurement did not produce a record")
        print(
            f"profile={profile.village}/{profile.werewolf}/{profile.fox} "
            f"games={record.games} wins(v/w/f)="
            f"{record.village_wins}/{record.werewolf_wins}/{record.fox_wins} "
            f"draws={record.draws} payoffs(v/w/f)="
            f"{record.mean_payoff(Team.VILLAGE):.3f}/"
            f"{record.mean_payoff(Team.WEREWOLF):.3f}/"
            f"{record.mean_payoff(Team.FOX):.3f}"
        )

    print(
        f"measured_profiles={len(profiles)} selected_policies={','.join(selected)} "
        f"table={args.table}"
    )


if __name__ == "__main__":
    main()
