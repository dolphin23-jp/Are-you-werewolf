"""Train a crash-resumable Transformer response oracle for each meta-game faction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.torch_oracle_cycle import (
    finalize_torch_oracle,
    start_torch_oracle_cycle,
    train_torch_oracle_subbatch,
    validate_torch_oracle_pool_boundary,
)
from app.training.torch_oracle_run_state import (
    load_torch_oracle_run_state,
    save_torch_oracle_run_state,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--meta-strategy", type=Path)
    parser.add_argument("--run-state", type=Path, default=Path("psro-oracles.run.npz"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes-per-oracle", type=int, default=4)
    parser.add_argument("--oracle-batch-size", type=int)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int)
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
    if args.oracle_batch_size is not None and args.oracle_batch_size <= 0:
        parser.error("--oracle-batch-size must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if not args.resume and args.meta_strategy is None:
        parser.error("--meta-strategy is required for a fresh oracle cycle")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    specs = _player_specs()
    pool = TorchPolicyPool(args.pool_dir, device=device)
    if not pool.entries:
        parser.error("--pool-dir contains no policy generations")

    if args.resume:
        if not args.run_state.exists():
            parser.error("--run-state does not exist")
        try:
            loop, progress = load_torch_oracle_run_state(
                args.run_state,
                specs,
                pool,
                device=device,
            )
            validate_torch_oracle_pool_boundary(pool, progress)
        except ValueError as exc:
            parser.error(str(exc))
        if args.meta_strategy is not None and loop is not None:
            supplied = PopulationMetaStrategy.load(args.meta_strategy)
            if supplied != loop.opponent_strategy:
                parser.error("--meta-strategy does not match the saved oracle cycle")
    else:
        assert args.meta_strategy is not None
        strategy = PopulationMetaStrategy.load(args.meta_strategy)
        oracle_batch_size = args.oracle_batch_size or args.episodes_per_oracle
        try:
            loop, progress = start_torch_oracle_cycle(
                specs,
                pool,
                strategy,
                episodes_per_oracle=args.episodes_per_oracle,
                oracle_batch_size=oracle_batch_size,
                base_seed=args.seed,
                opponent_seed=args.opponent_seed,
                trainer_seed=args.seed,
                ppo_config=TorchPPOConfig(
                    learning_rate=args.learning_rate,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                ),
                max_discussion_ticks=args.discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
        save_torch_oracle_run_state(loop, progress, args.run_state)

    if progress.is_complete:
        completed = ",".join(progress.completed_policy_ids)
        print(f"oracle_cycle=complete policies={completed} device={device}")
        return

    while not progress.is_complete:
        if loop is None:
            parser.error("active oracle cycle is missing its learner")
        team = progress.active_team
        if team is None:
            parser.error("active oracle cycle is missing its faction")

        if progress.next_batch_episodes > 0:
            batch_number = progress.completed_batches + 1
            try:
                stats, progress = train_torch_oracle_subbatch(loop, progress)
            except ValueError as exc:
                parser.error(str(exc))
            save_torch_oracle_run_state(loop, progress, args.run_state)
            update = stats.update
            inference_limit = loop.max_inference_batch_size or "unbounded"
            print(
                f"team={team.value} oracle_batch={batch_number} "
                f"episodes={progress.completed_episodes}/{progress.episodes_per_oracle} "
                f"record={progress.active_wins}-{progress.active_losses}-{progress.active_draws} "
                f"mean_days={progress.mean_days:.2f} "
                f"mean_decisions={progress.mean_decisions:.1f} "
                f"parallel_games={loop.max_parallel_games} "
                f"rollout_s={stats.rollout_seconds:.3f} "
                f"rollout_eps_s={stats.rollout_episodes_per_second:.2f} "
                f"inference_limit={inference_limit} "
                f"inference_mean_batch={stats.mean_inference_batch:.1f} "
                f"inference_max_batch={stats.max_inference_batch} "
                f"opponent_loads={stats.opponent_checkpoint_loads} "
                f"learner_s={stats.learner_seconds:.3f} "
                f"policy_loss={update.mean_policy_loss:.4f} "
                f"value_loss={update.mean_value_loss:.4f} "
                f"kl={update.mean_approx_kl:.6f} "
                f"entropy={update.mean_path_entropy:.4f} "
                f"value_ev={update.rollout_value_explained_variance:.4f} "
                f"grad_norm={update.gradient_norm:.4f} device={device}"
            )
            continue

        parent_id = progress.active_parent_policy_id
        record = (
            progress.active_wins,
            progress.active_losses,
            progress.active_draws,
            progress.mean_days,
            progress.mean_decisions,
        )
        try:
            next_loop, next_progress, entry = finalize_torch_oracle(
                specs,
                pool,
                loop,
                progress,
            )
        except ValueError as exc:
            parser.error(str(exc))
        save_torch_oracle_run_state(next_loop, next_progress, args.run_state)
        wins, losses, draws, mean_days, mean_decisions = record
        print(
            f"team={team.value} parent={parent_id} oracle={entry.policy_id} "
            f"record={wins}-{losses}-{draws} "
            f"mean_days={mean_days:.2f} mean_decisions={mean_decisions:.1f}"
        )
        loop = next_loop
        progress = next_progress

    completed = ",".join(progress.completed_policy_ids)
    print(f"oracle_cycle=complete policies={completed} device={device}")


if __name__ == "__main__":
    main()
