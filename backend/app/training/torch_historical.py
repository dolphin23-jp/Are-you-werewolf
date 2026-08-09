"""Historical-opponent self-play for the Transformer policy population."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.historical_train import HistoricalBatchStats
from app.training.learned_runner import LearnedEpisodeResult
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.torch_policy import TorchTransformerPolicy
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import (
    TorchPPOConfig,
    TorchPPOTrainer,
    TorchPPOUpdateStats,
)
from app.training.torch_vectorized import TorchVectorizedEpisodeCollector


@dataclass(frozen=True)
class TorchHistoricalBatchStats(HistoricalBatchStats):
    """Historical outcomes plus rollout/cache metrics for research training."""

    update: TorchPPOUpdateStats
    rollout_seconds: float
    learner_seconds: float
    inference_calls: int
    inference_observations: int
    max_pending_inference_requests: int
    max_inference_batch: int
    opponent_checkpoint_loads: int

    @property
    def rollout_episodes_per_second(self) -> float:
        if self.rollout_seconds <= 0:
            return 0.0
        return self.episodes / self.rollout_seconds

    @property
    def rollout_decisions_per_second(self) -> float:
        if self.rollout_seconds <= 0:
            return 0.0
        return (self.mean_decisions * self.episodes) / self.rollout_seconds

    @property
    def learner_decisions_per_second(self) -> float:
        if self.learner_seconds <= 0:
            return 0.0
        return self.update.decisions / self.learner_seconds

    @property
    def mean_inference_batch(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return self.inference_observations / self.inference_calls


class TorchHistoricalTrainingLoop:
    """Train one Transformer faction against immutable past generations."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: TorchTransformerPolicy,
        pool: TorchPolicyPool,
        *,
        opponent_strategy: PopulationMetaStrategy | None = None,
        opponent_seed: int = 0,
        ppo_config: TorchPPOConfig | None = None,
        max_discussion_ticks: int = 8,
        max_parallel_games: int = 8,
        max_inference_batch_size: int | None = None,
        temperature: float = 1.0,
        trainer_seed: int = 0,
    ) -> None:
        if max_parallel_games <= 0:
            raise ValueError("max_parallel_games must be positive")
        if max_inference_batch_size is not None and max_inference_batch_size <= 0:
            raise ValueError("max_inference_batch_size must be positive")
        if temperature != 1.0:
            raise ValueError(
                "Torch PPO currently requires temperature=1 because traces do not store it"
            )
        self.player_specs = player_specs
        self.model = model
        self.pool = pool
        self.opponent_strategy = opponent_strategy
        self.optimizer = TorchPPOTrainer(model, ppo_config, seed=trainer_seed)
        self.max_discussion_ticks = max_discussion_ticks
        self.max_parallel_games = max_parallel_games
        self.max_inference_batch_size = max_inference_batch_size
        self.temperature = temperature
        self._rng = random.Random(opponent_seed)
        if opponent_strategy is not None:
            self._validate_strategy_membership(opponent_strategy)

    def train_batch(
        self,
        *,
        learner_team: Team,
        start_seed: int,
        episodes: int,
    ) -> TorchHistoricalBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if not self.pool.entries:
            raise ValueError("historical self-play requires a non-empty policy pool")

        opponent_ids: list[str] = []
        sampled_matchups: list[dict[Team, str]] = []
        for _ in range(episodes):
            matchup: dict[Team, str] = {}
            for team in Team:
                if team is learner_team:
                    continue
                policy_id = self._sample_opponent(team)
                matchup[team] = policy_id
                opponent_ids.append(f"{team.value}:{policy_id}")
            sampled_matchups.append(matchup)

        seeds = tuple(start_seed + offset for offset in range(episodes))
        collector = TorchVectorizedEpisodeCollector(
            self.player_specs,
            self.model,
            max_discussion_ticks=self.max_discussion_ticks,
            max_inference_batch_size=self.max_inference_batch_size,
            temperature=self.temperature,
        )

        results: list[LearnedEpisodeResult] = []
        opponent_checkpoint_loads = 0
        rollout_started = perf_counter()
        for start in range(0, episodes, self.max_parallel_games):
            stop = min(start + self.max_parallel_games, episodes)
            chunk_matchups = sampled_matchups[start:stop]
            unique_opponent_ids = sorted(
                {
                    policy_id
                    for matchup in chunk_matchups
                    for policy_id in matchup.values()
                }
            )
            opponent_models = {
                policy_id: self.pool.load(policy_id).eval()
                for policy_id in unique_opponent_ids
            }
            opponent_checkpoint_loads += len(opponent_models)
            chunk_team_models = tuple(
                {
                    learner_team: self.model,
                    **{
                        team: opponent_models[policy_id]
                        for team, policy_id in matchup.items()
                    },
                }
                for matchup in chunk_matchups
            )
            results.extend(
                collector.collect(
                    seeds[start:stop],
                    team_models=chunk_team_models,
                )
            )
        rollout_seconds = perf_counter() - rollout_started

        trajectories = []
        wins = 0
        draws = 0
        total_days = 0
        for result in results:
            learner_ids = {
                player_id
                for player_id, team in result.teams.items()
                if team is learner_team
            }
            trajectories.append(result.trajectory.for_players(learner_ids))
            wins += int(result.winner is learner_team)
            draws += int(result.is_draw)
            total_days += result.days

        learner_started = perf_counter()
        update = self.optimizer.update(trajectories)
        learner_seconds = perf_counter() - learner_started
        losses = episodes - wins - draws
        inference = collector.inference_stats
        return TorchHistoricalBatchStats(
            learner_team=learner_team,
            episodes=episodes,
            wins=wins,
            losses=losses,
            draws=draws,
            mean_days=total_days / episodes,
            mean_decisions=sum(len(trajectory.decisions) for trajectory in trajectories)
            / episodes,
            opponent_policy_ids=tuple(opponent_ids),
            update=update,
            rollout_seconds=rollout_seconds,
            learner_seconds=learner_seconds,
            inference_calls=inference.inference_calls,
            inference_observations=inference.inference_observations,
            max_pending_inference_requests=inference.max_pending_requests,
            max_inference_batch=inference.max_inference_batch,
            opponent_checkpoint_loads=opponent_checkpoint_loads,
        )

    def checkpoint_opponent_rng_state(self) -> tuple[Any, ...]:
        """Return the historical-opponent RNG state at a batch boundary."""

        return self._rng.getstate()

    def restore_opponent_rng_state(self, state: tuple[Any, ...]) -> None:
        """Restore the exact historical-opponent sampling stream."""

        self._rng.setstate(state)

    def _sample_opponent(self, team: Team) -> str:
        if self.opponent_strategy is not None:
            return self.opponent_strategy.sample(team, self._rng)
        eligible = self.pool.entries_for_team(team)
        if not eligible:
            raise ValueError(f"policy pool has no opponent policy for {team}")
        return self._rng.choice(eligible).policy_id

    def _validate_strategy_membership(self, strategy: PopulationMetaStrategy) -> None:
        for team in Team:
            eligible = {
                entry.policy_id for entry in self.pool.entries_for_team(team)
            }
            missing = {
                item.policy_id
                for item in strategy.weights(team)
                if item.policy_id not in eligible
            }
            if missing:
                joined = ", ".join(sorted(missing))
                raise ValueError(
                    f"meta-strategy references policies not eligible for {team}: {joined}"
                )
