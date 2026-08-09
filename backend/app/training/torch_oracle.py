"""Approximate faction best-response oracle for Transformer populations."""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.numpy_trainer import PPOUpdateStats
from app.training.policy_pool import PolicyPoolEntry
from app.training.psro_oracle import dominant_policy_id
from app.training.torch_historical import TorchHistoricalTrainingLoop
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import TorchPPOConfig


@dataclass(frozen=True)
class TorchOracleTrainingStats:
    team: Team
    parent_policy_id: str
    oracle_entry: PolicyPoolEntry
    episodes: int
    wins: int
    losses: int
    draws: int
    mean_days: float
    mean_decisions: float
    update: PPOUpdateStats


def train_torch_population_oracle(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    strategy: PopulationMetaStrategy,
    *,
    team: Team,
    episodes: int,
    start_seed: int,
    opponent_seed: int = 0,
    parent_policy_id: str | None = None,
    ppo_config: TorchPPOConfig | None = None,
    max_discussion_ticks: int = 8,
    trainer_seed: int = 0,
) -> TorchOracleTrainingStats:
    """Train and persist one approximate response to the current opponent mix."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    parent_id = parent_policy_id or dominant_policy_id(strategy, team)
    eligible = {entry.policy_id for entry in pool.entries_for_team(team)}
    if parent_id not in eligible:
        raise ValueError(f"oracle parent {parent_id} is not eligible for {team}")

    model = pool.load(parent_id).eval()
    loop = TorchHistoricalTrainingLoop(
        player_specs,
        model,
        pool,
        opponent_strategy=strategy,
        opponent_seed=opponent_seed,
        ppo_config=ppo_config,
        max_discussion_ticks=max_discussion_ticks,
        trainer_seed=trainer_seed,
    )
    batch = loop.train_batch(
        learner_team=team,
        start_seed=start_seed,
        episodes=episodes,
    )
    entry = pool.add(
        model,
        parent_id=parent_id,
        specialized_team=team,
    )
    return TorchOracleTrainingStats(
        team=team,
        parent_policy_id=parent_id,
        oracle_entry=entry,
        episodes=batch.episodes,
        wins=batch.wins,
        losses=batch.losses,
        draws=batch.draws,
        mean_days=batch.mean_days,
        mean_decisions=batch.mean_decisions,
        update=batch.update,
    )
