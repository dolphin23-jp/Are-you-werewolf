"""Train a saved NumPy policy against historical generations from a policy pool."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.historical_train import HistoricalNumpyTrainingLoop
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("historical-policy.npz"))
    parser.add_argument("--meta-strategy", type=Path)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--episodes-per-batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument(
        "--team",
        choices=("all", Team.VILLAGE.value, Team.WEREWOLF.value, Team.FOX.value),
        default="all",
    )
    args = parser.parse_args()

    if args.batches <= 0:
        parser.error("--batches must be positive")
    if args.episodes_per_batch <= 0:
        parser.error("--episodes-per-batch must be positive")

    model = NumpyMLPPolicy.load(args.load)
    pool = NumpyPolicyPool(args.pool_dir)
    if not pool.entries:
        parser.error("--pool-dir must contain at least one saved generation")
    opponent_strategy = (
        PopulationMetaStrategy.load(args.meta_strategy)
        if args.meta_strategy is not None
        else None
    )

    loop = HistoricalNumpyTrainingLoop(
        _player_specs(),
        model,
        pool,
        opponent_strategy=opponent_strategy,
        opponent_seed=args.opponent_seed,
        max_discussion_ticks=args.discussion_ticks,
        ppo_config=PPOConfig(
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
        ),
    )
    requested_teams = tuple(Team) if args.team == "all" else (Team(args.team),)
    latest = pool.latest()
    parent_id = latest.policy_id if latest is not None else None

    for batch_index in range(args.batches):
        learner_team = requested_teams[batch_index % len(requested_teams)]
        start_seed = args.seed + batch_index * args.episodes_per_batch
        stats = loop.train_batch(
            learner_team=learner_team,
            start_seed=start_seed,
            episodes=args.episodes_per_batch,
        )
        model.save(args.output)
        entry = pool.add(
            model,
            parent_id=parent_id,
            specialized_team=learner_team,
        )
        parent_id = entry.policy_id
        update = stats.update
        opponents = ",".join(sorted(set(stats.opponent_policy_ids)))
        print(
            f"batch={batch_index + 1}/{args.batches} team={learner_team.value} "
            f"record={stats.wins}-{stats.losses}-{stats.draws} "
            f"mean_days={stats.mean_days:.2f} mean_decisions={stats.mean_decisions:.1f} "
            f"policy_loss={update.mean_policy_loss:.4f} "
            f"value_loss={update.mean_value_loss:.4f} "
            f"grad_norm={update.gradient_norm:.4f} "
            f"opponents={opponents} saved={entry.policy_id}"
        )


if __name__ == "__main__":
    main()
