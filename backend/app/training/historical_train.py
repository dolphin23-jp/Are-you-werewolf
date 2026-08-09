"""Historical-opponent self-play for the lightweight NumPy policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import NumpyPPOTrainer, PPOConfig, PPOUpdateStats
from app.training.policy_contract import LearnedPolicyModel
from app.training.policy_pool import NumpyPolicyPool


@dataclass(frozen=True)
class HistoricalBatchStats:
    learner_team: Team
    episodes: int
    wins: int
    losses: int
    draws: int
    mean_days: float
    mean_decisions: float
    opponent_policy_ids: tuple[str, ...]
    update: PPOUpdateStats


class HistoricalNumpyTrainingLoop:
    """Train one faction at a time against immutable past policy generations."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: NumpyMLPPolicy,
        pool: NumpyPolicyPool,
        *,
        opponent_strategy: PopulationMetaStrategy | None = None,
        opponent_seed: int = 0,
        ppo_config: PPOConfig | None = None,
        max_discussion_ticks: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.player_specs = player_specs
        self.model = model
        self.pool = pool
        self.opponent_strategy = opponent_strategy
        self.optimizer = NumpyPPOTrainer(model, ppo_config)
        self.max_discussion_ticks = max_discussion_ticks
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
    ) -> HistoricalBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if not self.pool.entries:
            raise ValueError("historical self-play requires a non-empty policy pool")

        trajectories = []
        wins = 0
        draws = 0
        total_days = 0
        opponent_ids: list[str] = []

        for offset in range(episodes):
            team_models: dict[Team, LearnedPolicyModel] = {learner_team: self.model}
            for team in Team:
                if team is learner_team:
                    continue
                policy_id = self._sample_opponent(team)
                team_models[team] = self.pool.load(policy_id)
                opponent_ids.append(f"{team.value}:{policy_id}")

            result = LearnedEpisodeRunner(
                self.player_specs,
                self.model,
                team_models=team_models,
                max_discussion_ticks=self.max_discussion_ticks,
                temperature=self.temperature,
            ).run(start_seed + offset)
            learner_ids = {
                player_id
                for player_id, team in result.teams.items()
                if team is learner_team
            }
            trajectories.append(result.trajectory.for_players(learner_ids))
            wins += int(result.winner is learner_team)
            draws += int(result.is_draw)
            total_days += result.days

        update = self.optimizer.update(trajectories)
        losses = episodes - wins - draws
        return HistoricalBatchStats(
            learner_team=learner_team,
            episodes=episodes,
            wins=wins,
            losses=losses,
            draws=draws,
            mean_days=total_days / episodes,
            mean_decisions=sum(len(t.decisions) for t in trajectories) / episodes,
            opponent_policy_ids=tuple(opponent_ids),
            update=update,
        )

    def _sample_opponent(self, team: Team) -> str:
        if self.opponent_strategy is not None:
            return self.opponent_strategy.sample(team, self._rng)
        return self.pool.sample(self._rng).policy_id

    def _validate_strategy_membership(self, strategy: PopulationMetaStrategy) -> None:
        known = {entry.policy_id for entry in self.pool.entries}
        for team in Team:
            missing = {
                item.policy_id for item in strategy.weights(team) if item.policy_id not in known
            }
            if missing:
                joined = ", ".join(sorted(missing))
                raise ValueError(f"meta-strategy references unknown pool policies: {joined}")
