"""Render a consolidated empirical population/meta-strategy report."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy, diagnose_meta_strategy
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.torch_pool import TorchPolicyPool


def _short(policy_id: str) -> str:
    if policy_id.startswith("g") and policy_id[1:].isdigit():
        return f"g{int(policy_id[1:])}"
    return policy_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--last", type=int, default=3)
    parser.add_argument("--games-per-profile", type=int)
    args = parser.parse_args()

    if args.last <= 0:
        parser.error("--last must be positive")
    if args.games_per_profile is not None and args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")

    pool = TorchPolicyPool(args.pool_dir, device="cpu")
    targets = {
        team: pool.policy_ids_for_team(team, last=args.last)
        for team in Team
    }
    profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            targets[Team.VILLAGE],
            targets[Team.WEREWOLF],
            targets[Team.FOX],
        )
    )
    table = PopulationPayoffTable(args.table)
    strategy = PopulationMetaStrategy.load(args.strategy)
    records = []
    for profile in profiles:
        record = table.get(profile)
        if record is None:
            raise SystemExit(f"missing payoff profile: {profile}")
        if args.games_per_profile is not None and record.games != args.games_per_profile:
            raise SystemExit(
                f"profile {profile} has {record.games} games; expected {args.games_per_profile}"
            )
        records.append(record)

    total_games = sum(record.games for record in records)
    wins = {team: sum(record.wins(team) for record in records) for team in Team}
    draws = sum(record.draws for record in records)

    print("===== POPULATION FIXED EVALUATION =====")
    for team in Team:
        print(f"{team.value}_targets=" + ",".join(_short(item) for item in targets[team]))
    print(f"profiles={len(records)} total_games={total_games}")

    print("\n===== SIMPLE OUTCOME TOTALS =====")
    for team in Team:
        count = wins[team]
        print(f"{team.value}_wins={count}/{total_games} ({100.0 * count / total_games:.2f}%)")
    print(f"draws={draws}/{total_games} ({100.0 * draws / total_games:.2f}%)")

    print("\n===== POLICY MARGINAL SIMPLE WIN RATES =====")
    for team in Team:
        print(f"[{team.value}]")
        for policy_id in targets[team]:
            selected = [
                record
                for record in records
                if record.profile.policy_id(team) == policy_id
            ]
            games = sum(record.games for record in selected)
            policy_wins = sum(record.wins(team) for record in selected)
            print(
                f"{_short(policy_id)} wins={policy_wins}/{games} "
                f"({100.0 * policy_wins / games:.2f}%)"
            )

    print("\n===== META WEIGHTS =====")
    for team in Team:
        rendered = ", ".join(
            f"{_short(item.policy_id)}={item.probability:.6f}"
            for item in strategy.weights(team)
        )
        print(f"{team.value}: {rendered}")

    diagnostics = diagnose_meta_strategy(table, strategy)
    print("\n===== RESTRICTED DEVIATION DIAGNOSTICS =====")
    for team in Team:
        item = diagnostics.for_team(team)
        print(
            f"{team.value}: mixture={item.mixture_payoff:+.6f} "
            f"best={_short(item.best_policy_id)} "
            f"best_payoff={item.best_policy_payoff:+.6f} "
            f"deviation_gain={item.deviation_gain:.6f}"
        )
    print(f"max_restricted_deviation_gain={diagnostics.max_deviation_gain:.6f}")

    weights = {
        team: {item.policy_id: item.probability for item in strategy.weights(team)}
        for team in Team
    }
    meta_outcomes = {team: 0.0 for team in Team}
    meta_draw = 0.0
    for record in records:
        probability = (
            weights[Team.VILLAGE][record.profile.village]
            * weights[Team.WEREWOLF][record.profile.werewolf]
            * weights[Team.FOX][record.profile.fox]
        )
        for team in Team:
            meta_outcomes[team] += probability * record.wins(team) / record.games
        meta_draw += probability * record.draws / record.games

    print("\n===== META-MIXTURE OUTCOME RATES =====")
    for team in Team:
        print(f"{team.value}={100.0 * meta_outcomes[team]:.3f}%")
    print(f"draw={100.0 * meta_draw:.3f}%")
    print(f"sum={100.0 * (sum(meta_outcomes.values()) + meta_draw):.6f}%")
    print("===== END POPULATION FIXED EVALUATION =====")


if __name__ == "__main__":
    main()
