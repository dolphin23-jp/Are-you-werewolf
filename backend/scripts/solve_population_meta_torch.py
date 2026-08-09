"""Solve an empirical meta-strategy for a Transformer policy population."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.roles import Team
from app.training.meta_strategy import (
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.population_payoff import PopulationPayoffTable
from app.training.torch_pool import TorchPolicyPool


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last", type=int)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.last is not None and args.last <= 0:
        parser.error("--last must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    village = pool.policy_ids_for_team(Team.VILLAGE, last=args.last)
    werewolf = pool.policy_ids_for_team(Team.WEREWOLF, last=args.last)
    fox = pool.policy_ids_for_team(Team.FOX, last=args.last)
    if not village or not werewolf or not fox:
        parser.error("each faction must have at least one eligible pool policy")

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
