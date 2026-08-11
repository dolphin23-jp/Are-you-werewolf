"""Idle-boundary runtime batching updates for population research."""

from __future__ import annotations

from dataclasses import replace

from app.training.torch_population_research import (
    TorchPopulationResearchState,
    TorchPopulationRunPhase,
)


def with_runtime_batching_at_idle(
    state: TorchPopulationResearchState,
    *,
    max_parallel_games: int,
    max_inference_batch_size: int,
) -> TorchPopulationResearchState:
    """Return an idle state with runtime-only batching limits updated.

    These settings affect rollout scheduling and inference batching only. They are
    deliberately mutable only before the next iteration freezes its population.
    """

    if state.phase is not TorchPopulationRunPhase.IDLE:
        raise ValueError("runtime batching can only change at an idle iteration boundary")
    if max_parallel_games <= 0:
        raise ValueError("max_parallel_games must be positive")
    if max_inference_batch_size <= 0:
        raise ValueError("max_inference_batch_size must be positive")
    return replace(
        state,
        config=replace(
            state.config,
            max_parallel_games=max_parallel_games,
            max_inference_batch_size=max_inference_batch_size,
        ),
    )
