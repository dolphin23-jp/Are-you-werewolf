"""Empirical three-faction payoff evaluation for Transformer populations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffResult,
    PopulationPayoffTable,
    ProfilePayoff,
)
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_vectorized import TorchVectorizedEpisodeCollector


@dataclass(frozen=True)
class TorchProfileEvaluationRequest:
    """One immutable population profile and the game seeds still needing measurement."""

    profile: PolicyProfile
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("profile evaluation request requires at least one seed")


@dataclass(frozen=True)
class TorchPopulationEvaluationStats:
    """Operational metrics for vectorized empirical payoff measurement."""

    games: int
    rollout_chunks: int
    rollout_seconds: float
    checkpoint_loads: int
    inference_calls: int
    inference_observations: int
    max_pending_inference_requests: int
    max_inference_batch: int

    @property
    def games_per_second(self) -> float:
        if self.rollout_seconds <= 0:
            return 0.0
        return self.games / self.rollout_seconds

    @property
    def mean_inference_batch(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return self.inference_observations / self.inference_calls


@dataclass(frozen=True)
class _EvaluationGame:
    profile: PolicyProfile
    seed: int


def evaluate_torch_policy_profiles(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    table: PopulationPayoffTable,
    requests: tuple[TorchProfileEvaluationRequest, ...],
    *,
    max_discussion_ticks: int = 8,
    max_parallel_games: int = 8,
    max_inference_batch_size: int | None = None,
    temperature: float = 1.0,
) -> TorchPopulationEvaluationStats:
    """Measure many profile games with cross-game, mixed-model inference batching.

    Models are cached only for one active rollout chunk, bounding accelerator
    residency by ``max_parallel_games`` rather than by the total policy cube.
    Games that happen to share a seed are placed in different chunks because the
    vectorized collector requires unique seeds within one call; their original
    seeds are never changed.
    """

    if not requests:
        raise ValueError("population evaluation requires at least one request")
    if max_parallel_games <= 0:
        raise ValueError("max_parallel_games must be positive")
    if max_inference_batch_size is not None and max_inference_batch_size <= 0:
        raise ValueError("max_inference_batch_size must be positive")

    games = tuple(
        _EvaluationGame(request.profile, seed)
        for request in requests
        for seed in request.seeds
    )
    rollout_chunks = 0
    checkpoint_loads = 0
    inference_calls = 0
    inference_observations = 0
    max_pending = 0
    max_inference_batch = 0
    rollout_started = perf_counter()

    for chunk in _chunk_unique_seeds(games, max_parallel_games=max_parallel_games):
        rollout_chunks += 1
        policy_ids = sorted(
            {
                policy_id
                for game in chunk
                for policy_id in (
                    game.profile.village,
                    game.profile.werewolf,
                    game.profile.fox,
                )
            }
        )
        model_cache = {policy_id: pool.load(policy_id).eval() for policy_id in policy_ids}
        checkpoint_loads += len(model_cache)
        default_model = model_cache[chunk[0].profile.village]
        team_models = tuple(
            {
                Team.VILLAGE: model_cache[game.profile.village],
                Team.WEREWOLF: model_cache[game.profile.werewolf],
                Team.FOX: model_cache[game.profile.fox],
            }
            for game in chunk
        )
        collector = TorchVectorizedEpisodeCollector(
            player_specs,
            default_model,
            max_discussion_ticks=max_discussion_ticks,
            max_inference_batch_size=max_inference_batch_size,
            temperature=temperature,
        )
        results = collector.collect(
            tuple(game.seed for game in chunk),
            team_models=team_models,
        )
        table.record_results(
            PopulationPayoffResult(
                profile=game.profile,
                winner=result.winner,
                is_draw=result.is_draw,
                days=result.days,
            )
            for game, result in zip(chunk, results, strict=True)
        )

        stats = collector.inference_stats
        inference_calls += stats.inference_calls
        inference_observations += stats.inference_observations
        max_pending = max(max_pending, stats.max_pending_requests)
        max_inference_batch = max(max_inference_batch, stats.max_inference_batch)

        # Release the active model set before loading the next chunk. This keeps
        # GPU residency proportional to rollout concurrency rather than pool size.
        del collector, team_models, default_model, model_cache

    return TorchPopulationEvaluationStats(
        games=len(games),
        rollout_chunks=rollout_chunks,
        rollout_seconds=perf_counter() - rollout_started,
        checkpoint_loads=checkpoint_loads,
        inference_calls=inference_calls,
        inference_observations=inference_observations,
        max_pending_inference_requests=max_pending,
        max_inference_batch=max_inference_batch,
    )


def evaluate_torch_policy_profile(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    table: PopulationPayoffTable,
    profile: PolicyProfile,
    *,
    seeds: tuple[int, ...],
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
    max_parallel_games: int = 1,
    max_inference_batch_size: int | None = None,
) -> ProfilePayoff:
    """Compatibility wrapper for measuring one immutable Transformer profile."""

    evaluate_torch_policy_profiles(
        player_specs,
        pool,
        table,
        (TorchProfileEvaluationRequest(profile, seeds),),
        max_discussion_ticks=max_discussion_ticks,
        max_parallel_games=max_parallel_games,
        max_inference_batch_size=max_inference_batch_size,
        temperature=temperature,
    )
    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile evaluation did not create a payoff record")
    return record


def _chunk_unique_seeds(
    games: tuple[_EvaluationGame, ...],
    *,
    max_parallel_games: int,
) -> tuple[tuple[_EvaluationGame, ...], ...]:
    """Pack games greedily while keeping collector seeds unique per chunk."""

    remaining = list(games)
    chunks: list[tuple[_EvaluationGame, ...]] = []
    while remaining:
        chunk: list[_EvaluationGame] = []
        deferred: list[_EvaluationGame] = []
        seen_seeds: set[int] = set()
        for game in remaining:
            if len(chunk) < max_parallel_games and game.seed not in seen_seeds:
                chunk.append(game)
                seen_seeds.add(game.seed)
            else:
                deferred.append(game)
        if not chunk:
            raise RuntimeError("population evaluation could not schedule a rollout chunk")
        chunks.append(tuple(chunk))
        remaining = deferred
    return tuple(chunks)
