"""Update population-research rollout batching at an idle iteration boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.training.population_runtime import with_runtime_batching_at_idle
from app.training.torch_population_research import (
    load_torch_population_research_state,
    save_torch_population_research_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parallel-games", type=int, required=True)
    parser.add_argument("--inference-batch-size", type=int, required=True)
    args = parser.parse_args()

    state_path = args.run_dir / "population.run.json"
    if not state_path.is_file():
        parser.error(f"missing population run-state: {state_path}")
    try:
        state = load_torch_population_research_state(state_path)
        updated = with_runtime_batching_at_idle(
            state,
            max_parallel_games=args.parallel_games,
            max_inference_batch_size=args.inference_batch_size,
        )
        save_torch_population_research_state(updated, state_path)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runtime_batching_updated completed_iterations={updated.completed_iterations} "
        f"phase={updated.phase.value} parallel_games={updated.config.max_parallel_games} "
        f"inference_batch_size={updated.config.max_inference_batch_size}"
    )


if __name__ == "__main__":
    main()
