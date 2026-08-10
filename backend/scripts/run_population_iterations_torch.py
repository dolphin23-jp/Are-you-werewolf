"""Run resumable empirical Transformer population iterations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.engine.game import PlayerSpec
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population_research import (
    TorchPopulationResearchConfig,
    TorchPopulationResearchEvent,
    TorchPopulationResearchRun,
)
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


def _print_event(event: TorchPopulationResearchEvent) -> None:
    print(f"iteration={event.iteration} event={event.kind} {event.message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recent-policies",
        type=int,
        help=(
            "restricted policies per faction; on --resume this may only change "
            "at an idle iteration boundary"
        ),
    )
    parser.add_argument("--games-per-profile", type=int, default=3)
    parser.add_argument("--extra-games", type=int, default=0)
    parser.add_argument("--uncertainty-prior", type=float, default=0.5)
    parser.add_argument("--oracle-episodes", type=int, default=4)
    parser.add_argument("--oracle-batch-size", type=int)
    parser.add_argument("--evaluation-seed", type=int, default=30000)
    parser.add_argument("--oracle-seed", type=int, default=40000)
    parser.add_argument("--opponent-seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--meta-temperature", type=float, default=0.25)
    parser.add_argument("--meta-iterations", type=int, default=100)
    parser.add_argument("--meta-damping", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    try:
        device = _resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    run = TorchPopulationResearchRun(
        _player_specs(),
        pool,
        args.run_dir,
        device=device,
    )

    if args.resume:
        try:
            state = run.resume()
            if (
                args.recent_policies is not None
                and args.recent_policies != state.config.recent_policies
            ):
                state = run.set_recent_policies(args.recent_policies)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"resume completed_iterations={state.completed_iterations} "
            f"phase={state.phase.value} recent_policies={state.config.recent_policies} "
            f"device={device}"
        )
    else:
        oracle_batch_size = args.oracle_batch_size or args.oracle_episodes
        config_overrides = {}
        if args.recent_policies is not None:
            config_overrides["recent_policies"] = args.recent_policies
        try:
            state = run.start(
                TorchPopulationResearchConfig(
                    **config_overrides,
                    games_per_profile=args.games_per_profile,
                    extra_games=args.extra_games,
                    uncertainty_prior=args.uncertainty_prior,
                    oracle_episodes=args.oracle_episodes,
                    oracle_batch_size=oracle_batch_size,
                    evaluation_seed=args.evaluation_seed,
                    oracle_seed=args.oracle_seed,
                    opponent_seed=args.opponent_seed,
                    max_discussion_ticks=args.discussion_ticks,
                    max_parallel_games=args.parallel_games,
                    max_inference_batch_size=args.inference_batch_size,
                    meta_temperature=args.meta_temperature,
                    meta_iterations=args.meta_iterations,
                    meta_damping=args.meta_damping,
                    ppo_config=TorchPPOConfig(
                        learning_rate=args.learning_rate,
                        epochs=args.ppo_epochs,
                        minibatch_size=args.minibatch_size,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        normalize_advantages=args.normalize_advantages,
                    ),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"start completed_iterations={state.completed_iterations} "
            f"phase={state.phase.value} recent_policies={state.config.recent_policies} "
            f"device={device}"
        )

    try:
        final = run.run_until(args.iterations, on_event=_print_event)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"research_run=complete completed_iterations={final.completed_iterations} "
        f"run_state={run.state_path} payoff_table={run.payoff_path} device={device}"
    )


if __name__ == "__main__":
    main()
