"""Train the lightweight structured werewolf policy with LLM-free self-play."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.engine.game import PlayerSpec
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.self_play_train import NumpySelfPlayTrainingLoop


def _player_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--output", type=Path, default=Path("self-play-policy.npz"))
    parser.add_argument("--pool-dir", type=Path)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    restored = NumpyMLPPolicy.load(args.load) if args.load is not None else None
    loop = NumpySelfPlayTrainingLoop(
        _player_specs(),
        model=restored,
        hidden_size=args.hidden_size,
        model_seed=args.seed,
        max_discussion_ticks=args.discussion_ticks,
        ppo_config=PPOConfig(
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
        ),
    )
    pool = NumpyPolicyPool(args.pool_dir) if args.pool_dir is not None else None
    latest = pool.latest() if pool is not None else None
    parent_id = latest.policy_id if latest is not None else None

    completed = 0
    batch_number = 0
    while completed < args.episodes:
        count = min(args.batch_size, args.episodes - completed)
        stats = loop.train_batch(
            start_seed=args.seed + completed,
            episodes=count,
        )
        completed += count
        batch_number += 1
        loop.model.save(args.output)
        policy_id = "-"
        if pool is not None:
            entry = pool.add(loop.model, parent_id=parent_id)
            parent_id = entry.policy_id
            policy_id = entry.policy_id
        update = stats.update
        print(
            f"batch={batch_number} episodes={completed}/{args.episodes} "
            f"wins(v/w/f)={stats.village_wins}/{stats.werewolf_wins}/{stats.fox_wins} "
            f"draws={stats.draws} mean_days={stats.mean_days:.2f} "
            f"mean_decisions={stats.mean_decisions:.1f} "
            f"policy_loss={update.mean_policy_loss:.4f} "
            f"value_loss={update.mean_value_loss:.4f} "
            f"ratio={update.mean_ratio:.4f} clip={update.clip_fraction:.3f} "
            f"grad_norm={update.gradient_norm:.4f} checkpoint={args.output} "
            f"policy_id={policy_id}"
        )


if __name__ == "__main__":
    main()
