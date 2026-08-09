"""Solve a payoff-driven opponent mixture from a measured population cube."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.engine.roles import Team
from app.training.meta_strategy import (
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.population_payoff import PopulationPayoffTable


def _parse_ids(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("policy id list cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--village")
    parser.add_argument("--werewolf")
    parser.add_argument("--fox")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()

    try:
        village = _parse_ids(args.village)
        werewolf = _parse_ids(args.werewolf)
        fox = _parse_ids(args.fox)
    except ValueError as exc:
        parser.error(str(exc))

    table = PopulationPayoffTable(args.table)
    strategy = solve_logit_response_mixture(
        table,
        village=village,
        werewolf=werewolf,
        fox=fox,
        temperature=args.temperature,
        iterations=args.iterations,
        damping=args.damping,
    )
    diagnostics = diagnose_meta_strategy(table, strategy)
    strategy.save(args.output)

    for team in Team:
        rendered = ", ".join(
            f"{item.policy_id}:{item.probability:.4f}"
            for item in strategy.weights(team)
        )
        diagnostic = diagnostics.for_team(team)
        print(f"{team.value}={rendered}")
        print(
            f"{team.value}_diagnostic="
            f"mixture_payoff:{diagnostic.mixture_payoff:.4f},"
            f"best:{diagnostic.best_policy_id},"
            f"best_payoff:{diagnostic.best_policy_payoff:.4f},"
            f"deviation_gain:{diagnostic.deviation_gain:.4f}"
        )
    print(f"max_restricted_deviation_gain={diagnostics.max_deviation_gain:.4f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
