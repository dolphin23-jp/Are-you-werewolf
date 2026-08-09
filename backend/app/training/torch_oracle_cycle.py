"""Resumable multi-faction PSRO oracle-cycle orchestration for Transformers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.policy_pool import PolicyPoolEntry
from app.training.psro_oracle import dominant_policy_id
from app.training.torch_historical import (
    TorchHistoricalBatchStats,
    TorchHistoricalTrainingLoop,
)
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_trainer import TorchPPOConfig


@dataclass(frozen=True)
class TorchOracleRunProgress:
    """Committed progress for one fixed-strategy multi-faction oracle cycle."""

    teams: tuple[Team, ...]
    team_index: int
    completed_episodes: int
    episodes_per_oracle: int
    oracle_batch_size: int
    base_seed: int
    opponent_seed: int
    trainer_seed: int
    next_pool_generation: int
    active_parent_policy_id: str | None
    completed_policy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.teams:
            raise ValueError("oracle teams cannot be empty")
        if len(set(self.teams)) != len(self.teams):
            raise ValueError("oracle teams cannot contain duplicates")
        if not 0 <= self.team_index <= len(self.teams):
            raise ValueError("oracle team_index is out of range")
        if self.episodes_per_oracle <= 0:
            raise ValueError("episodes_per_oracle must be positive")
        if self.oracle_batch_size <= 0:
            raise ValueError("oracle_batch_size must be positive")
        if not 0 <= self.completed_episodes <= self.episodes_per_oracle:
            raise ValueError("completed_episodes is out of range")
        if self.next_pool_generation < 0:
            raise ValueError("next_pool_generation cannot be negative")
        if len(self.completed_policy_ids) != self.team_index:
            raise ValueError("completed policy count must equal team_index")
        if self.is_complete:
            if self.completed_episodes != 0:
                raise ValueError("completed cycle cannot retain active episodes")
            if self.active_parent_policy_id is not None:
                raise ValueError("completed cycle cannot retain an active parent")
        elif self.active_parent_policy_id is None:
            raise ValueError("active oracle requires a parent policy")

    @property
    def is_complete(self) -> bool:
        return self.team_index == len(self.teams)

    @property
    def active_team(self) -> Team | None:
        if self.is_complete:
            return None
        return self.teams[self.team_index]

    @property
    def active_team_base_seed(self) -> int:
        if self.is_complete:
            raise ValueError("completed oracle cycle has no active seed")
        return self.base_seed + self.team_index * self.episodes_per_oracle

    @property
    def next_start_seed(self) -> int:
        return self.active_team_base_seed + self.completed_episodes

    @property
    def next_batch_episodes(self) -> int:
        if self.is_complete:
            return 0
        remaining = self.episodes_per_oracle - self.completed_episodes
        return min(self.oracle_batch_size, remaining)

    @property
    def completed_batches(self) -> int:
        if self.completed_episodes == 0:
            return 0
        return (
            self.completed_episodes + self.oracle_batch_size - 1
        ) // self.oracle_batch_size


def start_torch_oracle_cycle(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    strategy: PopulationMetaStrategy,
    *,
    teams: tuple[Team, ...] = tuple(Team),
    episodes_per_oracle: int,
    oracle_batch_size: int,
    base_seed: int,
    opponent_seed: int = 0,
    trainer_seed: int = 0,
    ppo_config: TorchPPOConfig | None = None,
    max_discussion_ticks: int = 8,
    max_parallel_games: int = 8,
    max_inference_batch_size: int | None = None,
) -> tuple[TorchHistoricalTrainingLoop, TorchOracleRunProgress]:
    """Create the first faction learner without mutating the immutable pool."""

    if not pool.entries:
        raise ValueError("oracle cycle requires a non-empty policy pool")
    if not teams:
        raise ValueError("oracle teams cannot be empty")
    first_team = teams[0]
    parent_id = _eligible_parent(pool, strategy, first_team)
    loop = _new_team_loop(
        player_specs,
        pool,
        strategy,
        team=first_team,
        parent_id=parent_id,
        opponent_seed=opponent_seed,
        trainer_seed=trainer_seed,
        ppo_config=ppo_config,
        max_discussion_ticks=max_discussion_ticks,
        max_parallel_games=max_parallel_games,
        max_inference_batch_size=max_inference_batch_size,
    )
    progress = TorchOracleRunProgress(
        teams=teams,
        team_index=0,
        completed_episodes=0,
        episodes_per_oracle=episodes_per_oracle,
        oracle_batch_size=oracle_batch_size,
        base_seed=base_seed,
        opponent_seed=opponent_seed,
        trainer_seed=trainer_seed,
        next_pool_generation=pool.next_generation,
        active_parent_policy_id=parent_id,
    )
    return loop, progress


def train_torch_oracle_subbatch(
    loop: TorchHistoricalTrainingLoop,
    progress: TorchOracleRunProgress,
) -> tuple[TorchHistoricalBatchStats, TorchOracleRunProgress]:
    """Train the next deterministic episode slice of the active oracle."""

    team = progress.active_team
    if team is None:
        raise ValueError("oracle cycle is already complete")
    episodes = progress.next_batch_episodes
    if episodes <= 0:
        raise ValueError("active oracle is ready for finalization")
    stats = loop.train_batch(
        learner_team=team,
        start_seed=progress.next_start_seed,
        episodes=episodes,
    )
    return stats, replace(
        progress,
        completed_episodes=progress.completed_episodes + episodes,
    )


def finalize_torch_oracle(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    loop: TorchHistoricalTrainingLoop,
    progress: TorchOracleRunProgress,
) -> tuple[
    TorchHistoricalTrainingLoop | None,
    TorchOracleRunProgress,
    PolicyPoolEntry,
]:
    """Persist one finished specialist and initialize the next fixed-strategy oracle."""

    team = progress.active_team
    if team is None:
        raise ValueError("oracle cycle is already complete")
    if progress.completed_episodes != progress.episodes_per_oracle:
        raise ValueError("active oracle has unfinished episodes")
    parent_id = progress.active_parent_policy_id
    if parent_id is None:
        raise ValueError("active oracle is missing its parent policy")
    strategy = loop.opponent_strategy
    if strategy is None:
        raise ValueError("PSRO oracle cycle requires a fixed opponent strategy")

    entry = pool.ensure_generation(
        loop.model,
        generation=progress.next_pool_generation,
        parent_id=parent_id,
        specialized_team=team,
    )
    completed_ids = (*progress.completed_policy_ids, entry.policy_id)
    next_index = progress.team_index + 1
    if next_index == len(progress.teams):
        return (
            None,
            replace(
                progress,
                team_index=next_index,
                completed_episodes=0,
                next_pool_generation=progress.next_pool_generation + 1,
                active_parent_policy_id=None,
                completed_policy_ids=completed_ids,
            ),
            entry,
        )

    next_team = progress.teams[next_index]
    next_parent = _eligible_parent(pool, strategy, next_team)
    next_loop = _new_team_loop(
        player_specs,
        pool,
        strategy,
        team=next_team,
        parent_id=next_parent,
        opponent_seed=progress.opponent_seed + next_index,
        trainer_seed=progress.trainer_seed + next_index,
        ppo_config=loop.optimizer.config,
        max_discussion_ticks=loop.max_discussion_ticks,
        max_parallel_games=loop.max_parallel_games,
        max_inference_batch_size=loop.max_inference_batch_size,
    )
    return (
        next_loop,
        replace(
            progress,
            team_index=next_index,
            completed_episodes=0,
            next_pool_generation=progress.next_pool_generation + 1,
            active_parent_policy_id=next_parent,
            completed_policy_ids=completed_ids,
        ),
        entry,
    )


def validate_torch_oracle_pool_boundary(
    pool: TorchPolicyPool,
    progress: TorchOracleRunProgress,
) -> None:
    """Reject pool mutations outside the single supported crash window."""

    expected = progress.next_pool_generation
    actual = pool.next_generation
    if progress.is_complete:
        if actual != expected:
            raise ValueError("policy pool does not match completed oracle cycle")
    elif progress.completed_episodes == progress.episodes_per_oracle:
        if actual not in (expected, expected + 1):
            raise ValueError(
                "policy pool does not match the completed active oracle boundary"
            )
    elif actual != expected:
        raise ValueError("policy pool changed during an active oracle")

    first_completed_generation = expected - len(progress.completed_policy_ids)
    if first_completed_generation < 0:
        raise ValueError("oracle completed-policy lineage is invalid")
    for offset, (policy_id, team) in enumerate(
        zip(progress.completed_policy_ids, progress.teams, strict=False)
    ):
        try:
            entry = pool.get(policy_id)
        except KeyError as exc:
            raise ValueError(f"completed oracle policy {policy_id} is missing") from exc
        if entry.generation != first_completed_generation + offset:
            raise ValueError("completed oracle generation lineage is invalid")
        if entry.specialized_team is not team:
            raise ValueError("completed oracle specialized faction is invalid")


def _eligible_parent(
    pool: TorchPolicyPool,
    strategy: PopulationMetaStrategy,
    team: Team,
) -> str:
    parent_id = dominant_policy_id(strategy, team)
    eligible = {entry.policy_id for entry in pool.entries_for_team(team)}
    if parent_id not in eligible:
        raise ValueError(f"oracle parent {parent_id} is not eligible for {team}")
    return parent_id


def _new_team_loop(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    strategy: PopulationMetaStrategy,
    *,
    team: Team,
    parent_id: str,
    opponent_seed: int,
    trainer_seed: int,
    ppo_config: TorchPPOConfig | None,
    max_discussion_ticks: int,
    max_parallel_games: int,
    max_inference_batch_size: int | None,
) -> TorchHistoricalTrainingLoop:
    if parent_id not in {
        entry.policy_id for entry in pool.entries_for_team(team)
    }:
        raise ValueError(f"oracle parent {parent_id} is not eligible for {team}")
    return TorchHistoricalTrainingLoop(
        player_specs,
        pool.load(parent_id).eval(),
        pool,
        opponent_strategy=strategy,
        opponent_seed=opponent_seed,
        ppo_config=ppo_config,
        max_discussion_ticks=max_discussion_ticks,
        max_parallel_games=max_parallel_games,
        max_inference_batch_size=max_inference_batch_size,
        trainer_seed=trainer_seed,
    )
