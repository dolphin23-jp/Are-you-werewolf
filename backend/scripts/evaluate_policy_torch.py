"""Evaluate a saved Transformer policy one faction at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.evaluation import evaluate_faction
from app.training.policy_contract import LearnedPolicyModel
from app.training.torch_checkpoint import load_torch_policy
from app.training.uniform_model import UniformPolicyModel


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _opponent(path: Path | None, *, device: torch.device) -> LearnedPolicyModel:
    if path is None:
        return UniformPolicyModel()
    return load_torch_policy(path, device=device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--team",
        choices=("all", Team.VILLAGE.value, Team.WEREWOLF.value, Team.FOX.value),
        default="all",
    )
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    candidate = load_torch_policy(args.candidate, device=device).eval()
    opponent = _opponent(args.opponent, device=device)
    teams = tuple(Team) if args.team == "all" else (Team(args.team),)
    seeds = tuple(range(args.seed, args.seed + args.games))

    for team in teams:
        stats = evaluate_faction(
            _player_specs(),
            candidate,
            opponent,
            team=team,
            seeds=seeds,
            max_discussion_ticks=args.discussion_ticks,
        )
        print(
            f"team={team.value} games={stats.games} wins={stats.wins} "
            f"losses={stats.losses} draws={stats.draws} "
            f"win_rate={stats.win_rate:.3f} mean_days={stats.mean_days:.2f} "
            f"device={device}"
        )


if __name__ == "__main__":
    main()
