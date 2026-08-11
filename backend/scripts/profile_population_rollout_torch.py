"""Profile population rollout bottlenecks without mutating the active run."""

from __future__ import annotations

import argparse
import cProfile
import io
import itertools
import os
import pstats
import resource
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
    import json

    state = json.loads((run_dir / "population.run.json").read_text(encoding="utf-8"))
    return int(state["completed_iterations"])


def _policy_ids(strategy: PopulationMetaStrategy, team: Team) -> tuple[str, ...]:
    return tuple(item.policy_id for item in strategy.weights(team))


def _usage_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--profiles", type=int, default=8)
    parser.add_argument("--games-per-profile", type=int, default=4)
    parser.add_argument("--parallel-games", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--discussion-ticks", type=int)
    parser.add_argument("--seed", type=int, default=91_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.profiles <= 0:
        parser.error("--profiles must be positive")
    if args.games_per_profile <= 0:
        parser.error("--games-per-profile must be positive")
    if args.parallel_games <= 0:
        parser.error("--parallel-games must be positive")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")

    try:
        device = _resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    iteration = args.iteration or _completed_iteration(args.run_dir)
    strategy = PopulationMetaStrategy.load(
        args.run_dir / f"iteration-{iteration:04d}" / "meta.json"
    )
    population = {
        team: _policy_ids(strategy, team)
        for team in Team
    }
    all_profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            population[Team.VILLAGE],
            population[Team.WEREWOLF],
            population[Team.FOX],
        )
    )
    selected = all_profiles[: min(args.profiles, len(all_profiles))]

    output_dir = args.output_dir or Path(
        f"/tmp/werewolf-rollout-profile-{os.getpid()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table = PopulationPayoffTable(output_dir / "payoffs.json")
    pool = TorchPolicyPool(args.pool_dir, device=device)

    state_path = args.run_dir / "population.run.json"
    import json

    state = json.loads(state_path.read_text(encoding="utf-8"))
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

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    python_profile = cProfile.Profile()
    process_cpu_before = _usage_seconds()
    wall_before = perf_counter()
    python_profile.enable()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as torch_profile:
        stats = evaluate_torch_policy_profiles(
            _player_specs(),
            pool,
            table,
            tuple(requests),
            max_discussion_ticks=discussion_ticks,
            max_parallel_games=args.parallel_games,
            max_inference_batch_size=args.inference_batch_size,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    python_profile.disable()
    wall_seconds = perf_counter() - wall_before
    process_cpu_seconds = _usage_seconds() - process_cpu_before

    python_binary = output_dir / "python-profile.prof"
    python_text = output_dir / "python-profile.txt"
    python_profile.dump_stats(python_binary)
    stream = io.StringIO()
    pstats.Stats(python_profile, stream=stream).sort_stats("cumulative").print_stats(60)
    python_text.write_text(stream.getvalue(), encoding="utf-8")

    torch_cpu_text = output_dir / "torch-cpu.txt"
    torch_cpu_text.write_text(
        torch_profile.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=60,
        ),
        encoding="utf-8",
    )
    torch_cuda_text: Path | None = None
    if device.type == "cuda":
        torch_cuda_text = output_dir / "torch-cuda.txt"
        try:
            cuda_table = torch_profile.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=60,
            )
        except KeyError:
            cuda_table = torch_profile.key_averages().table(
                sort_by="self_device_time_total",
                row_limit=60,
            )
        torch_cuda_text.write_text(cuda_table, encoding="utf-8")

    trace_path = output_dir / "torch-trace.json"
    torch_profile.export_chrome_trace(str(trace_path))

    print("===== POPULATION ROLLOUT PROFILE =====")
    print(f"iteration={iteration} device={device}")
    print(
        f"profiles={len(selected)} games={stats.games} "
        f"parallel_games={args.parallel_games} "
        f"inference_batch_size={args.inference_batch_size}"
    )
    print(
        f"wall_s={wall_seconds:.3f} process_cpu_s={process_cpu_seconds:.3f} "
        f"cpu_core_equivalents={process_cpu_seconds / wall_seconds:.3f}"
    )
    print(
        f"games_s={stats.games_per_second:.3f} "
        f"inference_calls={stats.inference_calls} "
        f"inference_mean_batch={stats.mean_inference_batch:.2f} "
        f"inference_max_batch={stats.max_inference_batch}"
    )
    print(f"torch_num_threads={torch.get_num_threads()}")
    if device.type == "cuda":
        print(
            f"cuda_peak_allocated_mb="
            f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f}"
        )
    print(f"python_profile={python_text}")
    print(f"torch_cpu_profile={torch_cpu_text}")
    if torch_cuda_text is not None:
        print(f"torch_cuda_profile={torch_cuda_text}")
    print(f"torch_trace={trace_path}")
    print("===== TOP PYTHON CUMULATIVE =====")
    print(stream.getvalue(), end="")
    print("===== TOP TORCH CPU =====")
    print(torch_cpu_text.read_text(encoding="utf-8"), end="")
    if torch_cuda_text is not None:
        print("===== TOP TORCH CUDA =====")
        print(torch_cuda_text.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
