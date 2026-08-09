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
from app.training.torch_historical_run_state import (
    TorchHistoricalRunProgress,
    load_torch_historical_run_state,
    save_torch_historical_run_state,
)
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


def _validate_pool_boundary(
    pool: TorchPolicyPool,
    progress: TorchHistoricalRunProgress,
) -> None:
    expected = progress.next_pool_generation
    if expected is None:
        raise ValueError("historical run-state is missing next pool generation")
    actual = pool.next_generation
    if actual not in (expected, expected + 1):
        raise ValueError(
            "policy pool does not match the committed historical generation boundary"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("historical-transformer.npz"))
    parser.add_argument("--run-state", type=Path, default=Path("historical.run.npz"))
    parser.add_argument("--resume", action="store_true")
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
    if args.resume and args.load is not None:
        parser.error("--load cannot be combined with --resume")
    if not args.resume and args.load is None:
        parser.error("--load is required for a fresh historical run")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    if not pool.entries:
        parser.error("--pool-dir must contain at least one saved generation")

    if args.resume:
        if not args.run_state.exists():
            parser.error("--run-state does not exist")
        try:
            loop, progress = load_torch_historical_run_state(
                args.run_state,
                _player_specs(),
                pool,
                device=device,
            )
            _validate_pool_boundary(pool, progress)
        except ValueError as exc:
            parser.error(str(exc))
        if args.batches < progress.completed_batches:
            parser.error(
                "--batches cannot be lower than the completed run-state batch count"
            )
        if args.meta_strategy is not None:
            supplied = PopulationMetaStrategy.load(args.meta_strategy)
            if supplied != loop.opponent_strategy:
                parser.error("--meta-strategy does not match the saved run-state")
    else:
        assert args.load is not None
        model = load_torch_policy(args.load, device=device).eval()
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
        requested_teams = (
            tuple(Team) if args.team == "all" else (Team(args.team),)
        )
        latest = pool.latest()
        progress = TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=args.seed,
            episodes_per_batch=args.episodes_per_batch,
            requested_teams=requested_teams,
            parent_policy_id=latest.policy_id if latest is not None else None,
            next_pool_generation=pool.next_generation,
        )
        save_torch_historical_run_state(loop, progress, args.run_state)

    for batch_index in range(progress.completed_batches, args.batches):
        learner_team = progress.next_learner_team
        start_seed = progress.next_start_seed
        stats = loop.train_batch(
            learner_team=learner_team,
            start_seed=start_seed,
            episodes=progress.episodes_per_batch,
        )
        save_torch_policy(loop.model, args.output)
        expected_generation = progress.next_pool_generation
        if expected_generation is None:
            parser.error("run-state is missing next pool generation")
        try:
            entry = pool.ensure_generation(
                loop.model,
                generation=expected_generation,
                parent_id=progress.parent_policy_id,
                specialized_team=learner_team,
            )
        except ValueError as exc:
            parser.error(str(exc))
        progress = TorchHistoricalRunProgress(
            completed_batches=batch_index + 1,
            base_seed=progress.base_seed,
            episodes_per_batch=progress.episodes_per_batch,
            requested_teams=progress.requested_teams,
            parent_policy_id=entry.policy_id,
            next_pool_generation=expected_generation + 1,
        )
        save_torch_historical_run_state(loop, progress, args.run_state)

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
