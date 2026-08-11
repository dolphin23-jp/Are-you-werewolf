"""Benchmark population rollout throughput without profiler overhead or run mutation."""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population import (
    TorchProfileEvaluationRequest,
    evaluate_torch_policy_profiles,
)
from app.training.torch_population_multiprocess import (
    evaluate_torch_policy_profiles_multiprocess,
)


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


def _completed_iteration(run_dir: Path) -> int:
    state = json.loads((run_dir / "population.run.json").read_text(encoding="utf-8"))
    return int(state["completed_iterations"])


def _policy_ids(strategy: PopulationMetaStrategy, team: Team) -> tuple[str, ...]:
    return tuple(item.policy_id for item in strategy.weights(team))


def _usage_seconds() -> float:
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--profiles", type=int, default=25)
    parser.add_argument("--games-per-profile", type=int, default=20)
    parser.add_argument("--parallel-games", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inference-coalesce-ms", type=float, default=2.0)
    parser.add_argument("--discussion-ticks", type=int)
    parser.add_argument("--seed", type=int, default=191_001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.profiles <= 0:
        parser.error("--profiles must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.inference_coalesce_ms < 0:
        parser.error("--inference-coalesce-ms cannot be negative")

    try:
        device = _resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    iteration = args.iteration or _completed_iteration(args.run_dir)
    strategy = PopulationMetaStrategy.load(
        args.run_dir / f"iteration-{iteration:04d}" / "meta.json"
    )
    population = {team: _policy_ids(strategy, team) for team in Team}
    all_profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            population[Team.VILLAGE],
            population[Team.WEREWOLF],
            population[Team.FOX],
        )
    )
    selected = all_profiles[: min(args.profiles, len(all_profiles))]

    state = json.loads(
        (args.run_dir / "population.run.json").read_text(encoding="utf-8")
    )
    config = state.get("config", {})
    discussion_ticks = (
        int(config.get("max_discussion_ticks", 8))
        if args.discussion_ticks is None
        else args.discussion_ticks
    )

    requests: list[TorchProfileEvaluationRequest] = []
    next_seed = args.seed
    for profile in selected:
        seeds = tuple(range(next_seed, next_seed + args.games_per_profile))
        next_seed += args.games_per_profile
        requests.append(TorchProfileEvaluationRequest(profile, seeds))

    pool = TorchPolicyPool(args.pool_dir, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with tempfile.TemporaryDirectory(prefix="werewolf-rollout-benchmark-") as tmp_dir:
        table = PopulationPayoffTable(Path(tmp_dir) / "payoffs.json")
        process_cpu_before = _usage_seconds()
        wall_before = perf_counter()
        if args.workers == 1:
            stats = evaluate_torch_policy_profiles(
                _player_specs(),
                pool,
                table,
                tuple(requests),
                max_discussion_ticks=discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
            )
        else:
            stats = evaluate_torch_policy_profiles_multiprocess(
                _player_specs(),
                pool,
                table,
                tuple(requests),
                worker_count=args.workers,
                max_discussion_ticks=discussion_ticks,
                max_parallel_games=args.parallel_games,
                max_inference_batch_size=args.inference_batch_size,
                inference_coalesce_seconds=args.inference_coalesce_ms / 1000.0,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = perf_counter() - wall_before
        process_cpu_seconds = _usage_seconds() - process_cpu_before

    print("===== POPULATION ROLLOUT BENCHMARK =====")
    print(f"iteration={iteration} device={device}")
    print(
        f"profiles={len(selected)} games={stats.games} "
        f"parallel_games={args.parallel_games} "
        f"inference_batch_size={args.inference_batch_size} "
        f"workers={args.workers} "
        f"inference_coalesce_ms={args.inference_coalesce_ms:.3f}"
    )
    print(
        f"wall_s={wall_seconds:.3f} process_cpu_s={process_cpu_seconds:.3f} "
        f"cpu_core_equivalents={process_cpu_seconds / wall_seconds:.3f}"
    )
    print(
        f"wall_games_s={stats.games / wall_seconds:.3f} "
        f"rollout_games_s={stats.games_per_second:.3f} "
        f"rollout_s={stats.rollout_seconds:.3f}"
    )
    print(
        f"inference_calls={stats.inference_calls} "
        f"inference_mean_batch={stats.mean_inference_batch:.2f} "
        f"inference_max_batch={stats.max_inference_batch} "
        f"max_pending={stats.max_pending_inference_requests}"
    )
    print(
        f"rollout_chunks={stats.rollout_chunks} "
        f"checkpoint_loads={stats.checkpoint_loads} "
        f"torch_num_threads={torch.get_num_threads()}"
    )
    if device.type == "cuda":
        print(
            f"cuda_peak_allocated_mb="
            f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f}"
        )
    print("===== END POPULATION ROLLOUT BENCHMARK =====")


if __name__ == "__main__":
    main()
