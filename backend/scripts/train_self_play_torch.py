"""Train the player/event Transformer with LLM-free structured self-play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.training.torch_checkpoint import load_torch_policy, save_torch_policy
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_run_state import (
    TorchRunProgress,
    load_torch_run_state,
    save_torch_run_state,
)
from app.training.torch_self_play import TorchSelfPlayBatchStats, TorchSelfPlayTrainingLoop
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


def _append_metrics(
    path: Path,
    *,
    batch_number: int,
    completed: int,
    start_seed: int,
    parallel_games: int,
    stats: TorchSelfPlayBatchStats,
) -> None:
    update = stats.update
    payload = {
        "batch": batch_number,
        "completed_episodes": completed,
        "batch_episodes": stats.episodes,
        "start_seed": start_seed,
        "parallel_games": parallel_games,
        "wins": {
            "village": stats.village_wins,
            "werewolf": stats.werewolf_wins,
            "fox": stats.fox_wins,
            "draw": stats.draws,
        },
        "mean_days": stats.mean_days,
        "mean_decisions": stats.mean_decisions,
        "rollout_seconds": stats.rollout_seconds,
        "learner_seconds": stats.learner_seconds,
        "rollout_episodes_per_second": stats.rollout_episodes_per_second,
        "rollout_decisions_per_second": stats.rollout_decisions_per_second,
        "learner_decisions_per_second": stats.learner_decisions_per_second,
        "ppo": {
            "decisions": update.decisions,
            "epochs": update.epochs,
            "mean_policy_loss": update.mean_policy_loss,
            "mean_value_loss": update.mean_value_loss,
            "mean_ratio": update.mean_ratio,
            "clip_fraction": update.clip_fraction,
            "gradient_norm": update.gradient_norm,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _pool_snapshot(
    pool: TorchPolicyPool,
    model: TorchTransformerPolicy,
    progress: TorchRunProgress,
) -> tuple[str, int]:
    generation = progress.next_pool_generation
    if generation is None:
        raise ValueError("run state does not expect a policy pool")
    policy_id = f"g{generation:06d}"
    try:
        existing = pool.get(policy_id)
    except KeyError:
        entry = pool.add(
            model,
            generation=generation,
            parent_id=progress.parent_policy_id,
        )
        return entry.policy_id, generation + 1

    if existing.parent_id != progress.parent_policy_id or existing.specialized_team is not None:
        raise ValueError(
            f"existing pool entry {policy_id} does not match resumable run lineage"
        )
    existing_model = pool.load(policy_id)
    if not _models_equal(existing_model, model):
        raise ValueError(
            f"existing pool entry {policy_id} differs from deterministically replayed batch"
        )
    return existing.policy_id, generation + 1


def _models_equal(
    left: TorchTransformerPolicy,
    right: TorchTransformerPolicy,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(
        torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu())
        for name in left_state
    )


def _validate_resume_pool(
    pool: TorchPolicyPool | None,
    progress: TorchRunProgress,
) -> None:
    expects_pool = progress.next_pool_generation is not None
    if expects_pool != (pool is not None):
        raise ValueError("resumed run must use the same policy-pool mode")
    if pool is None:
        return
    if progress.parent_policy_id is not None:
        pool.get(progress.parent_policy_id)
    generation = progress.next_pool_generation
    if generation is None:
        raise RuntimeError("validated pool progress lost its generation")
    if pool.next_generation > generation + 1:
        raise ValueError("policy pool advanced beyond the resumable run state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--feedforward", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load", type=Path)
    parser.add_argument("--output", type=Path, default=Path("self-play-transformer.npz"))
    parser.add_argument("--run-state", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pool-dir", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.resume and args.load is not None:
        parser.error("--resume and --load are mutually exclusive")

    run_state_path = args.run_state or args.output.with_suffix(args.output.suffix + ".run.npz")
    if run_state_path == args.output:
        parser.error("--run-state must differ from --output")

    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    specs = _player_specs()
    pool = (
        TorchPolicyPool(args.pool_dir, device=device)
        if args.pool_dir is not None
        else None
    )

    if args.resume:
        if not run_state_path.exists():
            parser.error(f"run state does not exist: {run_state_path}")
        try:
            loop, progress = load_torch_run_state(
                run_state_path,
                specs,
                device=device,
            )
            _validate_resume_pool(pool, progress)
        except (KeyError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        if progress.completed_episodes > args.episodes:
            parser.error(
                "--episodes is a total target and cannot be below resumed progress"
            )
    else:
        if run_state_path.exists():
            parser.error(
                f"run state already exists: {run_state_path}; use --resume or another path"
            )
        try:
            ppo_config = TorchPPOConfig(
                learning_rate=args.learning_rate,
                epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                normalize_advantages=args.normalize_advantages,
            )
        except ValueError as exc:
            parser.error(str(exc))

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        if args.load is not None:
            model = load_torch_policy(args.load, device=device)
        else:
            model = TorchTransformerPolicy(
                TransformerPolicyConfig(
                    d_model=args.d_model,
                    nhead=args.nhead,
                    num_layers=args.layers,
                    dim_feedforward=args.feedforward,
                    dropout=0.0,
                )
            ).to(device)

        loop = TorchSelfPlayTrainingLoop(
            specs,
            model=model,
            max_discussion_ticks=args.discussion_ticks,
            max_parallel_games=args.parallel_games,
            trainer_seed=args.seed,
            ppo_config=ppo_config,
        )
        latest = pool.latest() if pool is not None else None
        progress = TorchRunProgress(
            completed_episodes=0,
            batch_number=0,
            base_seed=args.seed,
            parent_policy_id=latest.policy_id if latest is not None else None,
            next_pool_generation=pool.next_generation if pool is not None else None,
        )
        save_torch_run_state(loop, progress, run_state_path)

    while progress.completed_episodes < args.episodes:
        count = min(args.batch_size, args.episodes - progress.completed_episodes)
        batch_start_seed = progress.base_seed + progress.completed_episodes
        stats = loop.train_batch(
            start_seed=batch_start_seed,
            episodes=count,
        )
        completed = progress.completed_episodes + count
        batch_number = progress.batch_number + 1
        save_torch_policy(loop.model, args.output)

        policy_id = "-"
        parent_policy_id = progress.parent_policy_id
        next_pool_generation = progress.next_pool_generation
        if pool is not None:
            try:
                parent_policy_id, next_pool_generation = _pool_snapshot(
                    pool,
                    loop.model,
                    progress,
                )
            except ValueError as exc:
                parser.error(str(exc))
            policy_id = parent_policy_id

        progress = TorchRunProgress(
            completed_episodes=completed,
            batch_number=batch_number,
            base_seed=progress.base_seed,
            parent_policy_id=parent_policy_id,
            next_pool_generation=next_pool_generation,
        )
        save_torch_run_state(loop, progress, run_state_path)

        if args.metrics_jsonl is not None:
            _append_metrics(
                args.metrics_jsonl,
                batch_number=batch_number,
                completed=completed,
                start_seed=batch_start_seed,
                parallel_games=loop.max_parallel_games,
                stats=stats,
            )
        update = stats.update
        print(
            f"batch={batch_number} episodes={completed}/{args.episodes} "
            f"parallel_games={loop.max_parallel_games} device={device} "
            f"wins(v/w/f)={stats.village_wins}/{stats.werewolf_wins}/{stats.fox_wins} "
            f"draws={stats.draws} mean_days={stats.mean_days:.2f} "
            f"mean_decisions={stats.mean_decisions:.1f} "
            f"rollout_s={stats.rollout_seconds:.3f} "
            f"rollout_eps_s={stats.rollout_episodes_per_second:.2f} "
            f"rollout_decisions_s={stats.rollout_decisions_per_second:.1f} "
            f"learner_s={stats.learner_seconds:.3f} "
            f"learner_decisions_s={stats.learner_decisions_per_second:.1f} "
            f"policy_loss={update.mean_policy_loss:.4f} "
            f"value_loss={update.mean_value_loss:.4f} "
            f"ratio={update.mean_ratio:.4f} clip={update.clip_fraction:.3f} "
            f"grad_norm={update.gradient_norm:.4f} checkpoint={args.output} "
            f"run_state={run_state_path} policy_id={policy_id}"
        )


if __name__ == "__main__":
    main()
