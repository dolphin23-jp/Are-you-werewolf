"""Approximate best-response oracle training for one population faction.

The oracle is intentionally lightweight: it clones an existing eligible policy,
trains only the selected faction's decisions against the other factions' current
meta-strategy mixture, then stores the result as a faction-specialized immutable
policy generation. This is a PSRO-style population-expansion step, not a proof
that the learned policy is an exact best response.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.historical_train import HistoricalNumpyTrainingLoop
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.numpy_trainer import PPOConfig, PPOUpdateStats
from app.training.policy_pool import NumpyPolicyPool, PolicyPoolEntry


@dataclass(frozen=True)
class OracleTrainingStats:
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


def dominant_policy_id(strategy: PopulationMetaStrategy, team: Team) -> str:
    """Choose the highest-probability policy as a deterministic oracle initializer."""
    weights = strategy.weights(team)
    if not weights:
        raise ValueError(f"meta-strategy has no policies for {team}")
    return max(weights, key=lambda item: (item.probability, item.policy_id)).policy_id


def train_population_oracle(
    player_specs: list[PlayerSpec],
    pool: NumpyPolicyPool,
    strategy: PopulationMetaStrategy,
    *,
    team: Team,
    episodes: int,
    start_seed: int,
    opponent_seed: int = 0,
    parent_policy_id: str | None = None,
    ppo_config: PPOConfig | None = None,
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
) -> OracleTrainingStats:
    """Train and persist one approximate response to the current opponent mix."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    parent_id = parent_policy_id or dominant_policy_id(strategy, team)
    eligible = {entry.policy_id for entry in pool.entries_for_team(team)}
    if parent_id not in eligible:
        raise ValueError(f"oracle parent {parent_id} is not eligible for {team}")

    model = pool.load(parent_id)
    loop = HistoricalNumpyTrainingLoop(
        player_specs,
        model,
        pool,
        opponent_strategy=strategy,
        opponent_seed=opponent_seed,
        ppo_config=ppo_config,
        max_discussion_ticks=max_discussion_ticks,
        temperature=temperature,
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
    return OracleTrainingStats(
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
