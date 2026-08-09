"""Historical-opponent self-play for the Transformer policy population."""

from __future__ import annotations

import random

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.historical_train import HistoricalBatchStats
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.policy_contract import LearnedPolicyModel
from app.training.torch_policy import TorchTransformerPolicy
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import TorchPPOConfig, TorchPPOTrainer


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
        temperature: float = 1.0,
        trainer_seed: int = 0,
    ) -> None:
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
        self.model.eval()

        for offset in range(episodes):
            team_models: dict[Team, LearnedPolicyModel] = {learner_team: self.model}
            for team in Team:
                if team is learner_team:
                    continue
                policy_id = self._sample_opponent(team)
                opponent = self.pool.load(policy_id).eval()
                team_models[team] = opponent
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
            mean_decisions=sum(len(trajectory.decisions) for trajectory in trajectories)
            / episodes,
            opponent_policy_ids=tuple(opponent_ids),
            update=update,
        )

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
