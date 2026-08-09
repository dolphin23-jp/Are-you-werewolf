"""Batch self-play loop for the lightweight NumPy policy baseline."""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import NumpyPPOTrainer, PPOConfig, PPOUpdateStats


@dataclass(frozen=True)
class SelfPlayBatchStats:
    episodes: int
    village_wins: int
    werewolf_wins: int
    fox_wins: int
    draws: int
    mean_days: float
    mean_decisions: float
    update: PPOUpdateStats


class NumpySelfPlayTrainingLoop:
    """Generate on-policy games, then update the one shared role-conditioned model."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        *,
        model: NumpyMLPPolicy | None = None,
        hidden_size: int = 64,
        model_seed: int = 0,
        ppo_config: PPOConfig | None = None,
        max_discussion_ticks: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.player_specs = player_specs
        self.model = model or NumpyMLPPolicy(hidden_size=hidden_size, seed=model_seed)
        self.optimizer = NumpyPPOTrainer(self.model, ppo_config)
        self.max_discussion_ticks = max_discussion_ticks
        self.temperature = temperature

    def train_batch(self, *, start_seed: int, episodes: int) -> SelfPlayBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        results = []
        trajectories = []
        for offset in range(episodes):
            runner = LearnedEpisodeRunner(
                self.player_specs,
                self.model,
                max_discussion_ticks=self.max_discussion_ticks,
                temperature=self.temperature,
            )
            result = runner.run(start_seed + offset)
            results.append(result)
            trajectories.append(result.trajectory)

        update = self.optimizer.update(trajectories)
        return SelfPlayBatchStats(
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
        )
