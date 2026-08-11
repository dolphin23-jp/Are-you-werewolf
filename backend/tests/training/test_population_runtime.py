import pytest

pytest.importorskip("torch")

from app.training.population_runtime import with_runtime_batching_at_idle
from app.training.torch_population_research import (
    TorchPopulationResearchConfig,
    TorchPopulationResearchState,
    TorchPopulationRunPhase,
)


def test_runtime_batching_updates_only_runtime_limits_at_idle():
    state = TorchPopulationResearchState(
        config=TorchPopulationResearchConfig(
            recent_policies=5,
            games_per_profile=5,
            extra_games=32,
            max_parallel_games=16,
            max_inference_batch_size=64,
        ),
        completed_iterations=8,
    )

    updated = with_runtime_batching_at_idle(
        state,
        max_parallel_games=64,
        max_inference_batch_size=256,
    )

    assert updated.completed_iterations == 8
    assert updated.phase is TorchPopulationRunPhase.IDLE
    assert updated.config.max_parallel_games == 64
    assert updated.config.max_inference_batch_size == 256
    assert updated.config.recent_policies == state.config.recent_policies
    assert updated.config.games_per_profile == state.config.games_per_profile
    assert updated.config.extra_games == state.config.extra_games
    assert updated.config.ppo_config == state.config.ppo_config


def test_runtime_batching_rejects_active_iteration():
    state = TorchPopulationResearchState(
        config=TorchPopulationResearchConfig(),
        phase=TorchPopulationRunPhase.MEASURE,
        village_policy_ids=("g0",),
        werewolf_policy_ids=("g1",),
        fox_policy_ids=("g2",),
        iteration_pool_generation=3,
    )

    with pytest.raises(ValueError, match="idle iteration boundary"):
        with_runtime_batching_at_idle(
            state,
            max_parallel_games=64,
            max_inference_batch_size=256,
        )


def test_runtime_batching_requires_positive_limits():
    state = TorchPopulationResearchState(config=TorchPopulationResearchConfig())

    with pytest.raises(ValueError, match="max_parallel_games must be positive"):
        with_runtime_batching_at_idle(
            state,
            max_parallel_games=0,
            max_inference_batch_size=256,
        )
    with pytest.raises(ValueError, match="max_inference_batch_size must be positive"):
        with_runtime_batching_at_idle(
            state,
            max_parallel_games=64,
            max_inference_batch_size=0,
        )
