"""Train a Transformer policy against immutable historical generations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.torch_checkpoint import load_torch_policy, save_torch_policy
from app.training.torch_historical import TorchHistoricalTrainingLoop
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
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("historical-transformer.npz"))
    parser.add_argument("--meta-strategy", type=Path)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--episodes-per-batch", type=int, default=2)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
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
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    model = load_torch_policy(args.load, device=device).eval()
    pool = TorchPolicyPool(args.pool_dir, device=device)
    if not pool.entries:
        parser.error("--pool-dir must contain at least one saved generation")
    opponent_strategy = (
        PopulationMetaStrategy.load(args.meta_strategy)
        if args.meta_strategy is not None
        else None
    )
    loop = TorchHistoricalTrainingLoop(
        _player_specs(),
        model,
        pool,
        opponent_strategy=opponent_strategy,
        opponent_seed=args.opponent_seed,
        max_discussion_ticks=args.discussion_ticks,
        max_parallel_games=args.parallel_games,
        max_inference_batch_size=args.inference_batch_size,
        trainer_seed=args.seed,
        ppo_config=TorchPPOConfig(
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
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
        save_torch_policy(model, args.output)
        entry = pool.add(
            model,
            parent_id=parent_id,
            specialized_team=learner_team,
        )
        parent_id = entry.policy_id
        update = stats.update
        opponents = ",".join(sorted(set(stats.opponent_policy_ids)))
        inference_limit = loop.max_inference_batch_size or "unbounded"
        print(
            f"batch={batch_index + 1}/{args.batches} team={learner_team.value} "
            f"record={stats.wins}-{stats.losses}-{stats.draws} "
            f"mean_days={stats.mean_days:.2f} mean_decisions={stats.mean_decisions:.1f} "
            f"parallel_games={loop.max_parallel_games} "
            f"rollout_s={stats.rollout_seconds:.3f} "
            f"rollout_eps_s={stats.rollout_episodes_per_second:.2f} "
            f"rollout_decisions_s={stats.rollout_decisions_per_second:.1f} "
            f"inference_limit={inference_limit} "
            f"inference_mean_batch={stats.mean_inference_batch:.1f} "
            f"inference_max_batch={stats.max_inference_batch} "
            f"inference_max_pending={stats.max_pending_inference_requests} "
            f"opponent_loads={stats.opponent_checkpoint_loads} "
            f"learner_s={stats.learner_seconds:.3f} "
            f"learner_decisions_s={stats.learner_decisions_per_second:.1f} "
            f"policy_loss={update.mean_policy_loss:.4f} "
            f"value_loss={update.mean_value_loss:.4f} "
            f"kl={update.mean_approx_kl:.6f} "
            f"entropy={update.mean_path_entropy:.4f} "
            f"value_ev={update.rollout_value_explained_variance:.4f} "
            f"grad_norm={update.gradient_norm:.4f} "
            f"opponents={opponents} saved={entry.policy_id} device={device}"
        )


if __name__ == "__main__":
    main()
