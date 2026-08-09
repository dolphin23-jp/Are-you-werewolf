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
    parser.add_argument("--pool-dir", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")

    try:
        device = _resolve_device(args.device)
        ppo_config = TorchPPOConfig(
            learning_rate=args.learning_rate,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=args.normalize_advantages,
        )
    except (RuntimeError, ValueError) as exc:
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
        _player_specs(),
        model=model,
        max_discussion_ticks=args.discussion_ticks,
        max_parallel_games=args.parallel_games,
        trainer_seed=args.seed,
        ppo_config=ppo_config,
    )
    pool = (
        TorchPolicyPool(args.pool_dir, device=device)
        if args.pool_dir is not None
        else None
    )
    latest = pool.latest() if pool is not None else None
    parent_id = latest.policy_id if latest is not None else None

    completed = 0
    batch_number = 0
    while completed < args.episodes:
        count = min(args.batch_size, args.episodes - completed)
        batch_start_seed = args.seed + completed
        stats = loop.train_batch(
            start_seed=batch_start_seed,
            episodes=count,
        )
        completed += count
        batch_number += 1
        save_torch_policy(loop.model, args.output)
        policy_id = "-"
        if pool is not None:
            entry = pool.add(loop.model, parent_id=parent_id)
            parent_id = entry.policy_id
            policy_id = entry.policy_id
        if args.metrics_jsonl is not None:
            _append_metrics(
                args.metrics_jsonl,
                batch_number=batch_number,
                completed=completed,
                start_seed=batch_start_seed,
                parallel_games=args.parallel_games,
                stats=stats,
            )
        update = stats.update
        print(
            f"batch={batch_number} episodes={completed}/{args.episodes} "
            f"parallel_games={args.parallel_games} device={device} "
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
            f"policy_id={policy_id}"
        )


if __name__ == "__main__":
    main()
