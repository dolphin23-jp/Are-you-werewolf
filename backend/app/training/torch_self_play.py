"""Full self-play training loop for the PyTorch Transformer policy."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeResult
from app.training.self_play_train import SelfPlayBatchStats
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_trainer import (
    TorchPPOConfig,
    TorchPPOTrainer,
    TorchPPOUpdateStats,
)
from app.training.torch_vectorized import TorchVectorizedEpisodeCollector


@dataclass(frozen=True)
class TorchSelfPlayBatchStats(SelfPlayBatchStats):
    """Self-play outcomes plus runtime metrics kept outside policy observations."""

    update: TorchPPOUpdateStats
    rollout_seconds: float
    learner_seconds: float
    inference_calls: int
    inference_observations: int
    max_pending_inference_requests: int
    max_inference_batch: int

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


class TorchSelfPlayTrainingLoop:
    """Generate batched 17-seat episodes, then PPO-update one shared model."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        *,
        model: TorchTransformerPolicy | None = None,
        model_config: TransformerPolicyConfig | None = None,
        ppo_config: TorchPPOConfig | None = None,
        max_discussion_ticks: int = 8,
        max_parallel_games: int = 8,
        max_inference_batch_size: int | None = None,
        temperature: float = 1.0,
        trainer_seed: int = 0,
    ) -> None:
        if model is not None and model_config is not None:
            raise ValueError("provide either model or model_config, not both")
        if max_parallel_games <= 0:
            raise ValueError("max_parallel_games must be positive")
        if max_inference_batch_size is not None and max_inference_batch_size <= 0:
            raise ValueError("max_inference_batch_size must be positive")
        if temperature != 1.0:
            raise ValueError(
                "Torch PPO currently requires temperature=1 because traces do not store it"
            )
        self.player_specs = player_specs
        self.model = model or TorchTransformerPolicy(model_config)
        self.optimizer = TorchPPOTrainer(
            self.model,
            ppo_config,
            seed=trainer_seed,
        )
        self.max_discussion_ticks = max_discussion_ticks
        self.max_parallel_games = max_parallel_games
        self.max_inference_batch_size = max_inference_batch_size
        self.temperature = temperature

    def train_batch(self, *, start_seed: int, episodes: int) -> TorchSelfPlayBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        seeds = tuple(start_seed + offset for offset in range(episodes))
        collector = TorchVectorizedEpisodeCollector(
            self.player_specs,
            self.model,
            max_discussion_ticks=self.max_discussion_ticks,
            max_inference_batch_size=self.max_inference_batch_size,
            temperature=self.temperature,
        )
        results: list[LearnedEpisodeResult] = []
        rollout_started = perf_counter()
        for start in range(0, len(seeds), self.max_parallel_games):
            results.extend(
                collector.collect(seeds[start : start + self.max_parallel_games])
            )
        rollout_seconds = perf_counter() - rollout_started
        trajectories = [result.trajectory for result in results]

        learner_started = perf_counter()
        update = self.optimizer.update(trajectories)
        learner_seconds = perf_counter() - learner_started
        inference = collector.inference_stats
        return TorchSelfPlayBatchStats(
            episodes=episodes,
            village_wins=sum(result.winner is Team.VILLAGE for result in results),
            werewolf_wins=sum(result.winner is Team.WEREWOLF for result in results),
            fox_wins=sum(result.winner is Team.FOX for result in results),
            draws=sum(result.is_draw for result in results),
            mean_days=sum(result.days for result in results) / episodes,
            mean_decisions=(
                sum(len(result.trajectory.decisions) for result in results) / episodes
            ),
            update=update,
            rollout_seconds=rollout_seconds,
            learner_seconds=learner_seconds,
            inference_calls=inference.inference_calls,
            inference_observations=inference.inference_observations,
            max_pending_inference_requests=inference.max_pending_requests,
            max_inference_batch=inference.max_inference_batch,
        )
