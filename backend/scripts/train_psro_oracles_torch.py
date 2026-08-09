"""Train one Transformer response oracle for each meta-game faction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.torch_oracle import train_torch_population_oracle
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import TorchPPOConfig


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--meta-strategy", type=Path, required=True)
    parser.add_argument("--episodes-per-oracle", type=int, default=4)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.episodes_per_oracle <= 0:
        parser.error("--episodes-per-oracle must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    if not pool.entries:
        parser.error("--pool-dir contains no policy generations")
    strategy = PopulationMetaStrategy.load(args.meta_strategy)
    ppo_config = TorchPPOConfig(
        learning_rate=args.learning_rate,
        epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
    )

    for team_index, team in enumerate(Team):
        stats = train_torch_population_oracle(
            _player_specs(),
            pool,
            strategy,
            team=team,
            episodes=args.episodes_per_oracle,
            start_seed=args.seed + team_index * args.episodes_per_oracle,
            opponent_seed=args.opponent_seed + team_index,
            trainer_seed=args.seed + team_index,
            ppo_config=ppo_config,
            max_discussion_ticks=args.discussion_ticks,
        )
        update = stats.update
        print(
            f"team={team.value} parent={stats.parent_policy_id} "
            f"oracle={stats.oracle_entry.policy_id} "
            f"record={stats.wins}-{stats.losses}-{stats.draws} "
            f"mean_days={stats.mean_days:.2f} "
            f"mean_decisions={stats.mean_decisions:.1f} "
            f"policy_loss={update.mean_policy_loss:.4f} "
            f"value_loss={update.mean_value_loss:.4f} "
            f"grad_norm={update.gradient_norm:.4f} device={device}"
        )


if __name__ == "__main__":
    main()
