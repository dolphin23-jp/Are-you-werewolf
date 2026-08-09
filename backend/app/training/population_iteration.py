"""One empirical population-training iteration for the NumPy self-play baseline.

The iteration intentionally stays simple and inspectable:

1. measure a complete three-faction payoff cube for recent immutable policies,
2. solve a payoff-driven mixture over that measured population,
3. train the current shared policy once as Village, Werewolf, and Fox against
   opponents sampled from that mixture,
4. save the resulting shared policy as the next immutable generation.

This is a practical bridge toward PSRO/JPSRO-style training, not a formal
population-game equilibrium solver.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.historical_train import HistoricalBatchStats, HistoricalNumpyTrainingLoop
from app.training.meta_strategy import (
    PopulationMetaDiagnostics,
    PopulationMetaStrategy,
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.numpy_policy import NumpyMLPPolicy
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool, PolicyPoolEntry
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffTable,
    evaluate_policy_profile,
)


@dataclass(frozen=True)
class PopulationIterationStats:
    measured_policy_ids: tuple[str, ...]
    measured_profiles: int
    new_profile_games: int
    meta_strategy: PopulationMetaStrategy
    diagnostics: PopulationMetaDiagnostics
    training: tuple[HistoricalBatchStats, ...]
    saved_entry: PolicyPoolEntry


def run_population_iteration(
    player_specs: list[PlayerSpec],
    pool: NumpyPolicyPool,
    table: PopulationPayoffTable,
    *,
    recent_policies: int = 3,
    games_per_profile: int = 3,
    evaluation_seed: int = 30000,
    training_seed: int = 40000,
    opponent_seed: int = 1,
    episodes_per_team: int = 2,
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
    meta_temperature: float = 0.25,
    meta_iterations: int = 100,
    meta_damping: float = 0.5,
    ppo_config: PPOConfig | None = None,
) -> PopulationIterationStats:
    """Measure, solve, train all factions, and checkpoint one new generation."""
    if recent_policies <= 0:
        raise ValueError("recent_policies must be positive")
    if games_per_profile <= 0:
        raise ValueError("games_per_profile must be positive")
    if episodes_per_team <= 0:
        raise ValueError("episodes_per_team must be positive")
    latest = pool.latest()
    if latest is None:
        raise ValueError("population iteration requires a non-empty policy pool")

    selected = tuple(entry.policy_id for entry in pool.entries[-recent_policies:])
    new_profile_games = 0
    profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(selected, repeat=3)
    )
    for profile in profiles:
        existing = table.get(profile)
        existing_games = existing.games if existing is not None else 0
        missing = max(0, games_per_profile - existing_games)
        if missing == 0:
            continue
        start = _profile_seed(evaluation_seed, profile) + existing_games
        evaluate_policy_profile(
            player_specs,
            pool,
            table,
            profile,
            seeds=tuple(range(start, start + missing)),
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        )
        new_profile_games += missing

    meta_strategy = solve_logit_response_mixture(
        table,
        village=selected,
        werewolf=selected,
        fox=selected,
        temperature=meta_temperature,
        iterations=meta_iterations,
        damping=meta_damping,
    )
    diagnostics = diagnose_meta_strategy(table, meta_strategy)

    model = pool.load(latest.policy_id)
    trainer = HistoricalNumpyTrainingLoop(
        player_specs,
        model,
        pool,
        opponent_seed=opponent_seed,
        meta_strategy=meta_strategy,
        ppo_config=ppo_config,
        max_discussion_ticks=max_discussion_ticks,
        temperature=temperature,
    )
    training: list[HistoricalBatchStats] = []
    for team_index, team in enumerate(Team):
        training.append(
            trainer.train_batch(
                learner_team=team,
                start_seed=training_seed + team_index * episodes_per_team,
                episodes=episodes_per_team,
            )
        )

    saved_entry = pool.add(model, parent_id=latest.policy_id)
    return PopulationIterationStats(
        measured_policy_ids=selected,
        measured_profiles=len(profiles),
        new_profile_games=new_profile_games,
        meta_strategy=meta_strategy,
        diagnostics=diagnostics,
        training=tuple(training),
        saved_entry=saved_entry,
    )


def _profile_seed(base_seed: int, profile: PolicyProfile) -> int:
    """Stable profile-local seed independent of enumeration order."""
    payload = f"{profile.village}|{profile.werewolf}|{profile.fox}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return base_seed + int.from_bytes(digest, "big") % 1_000_000_000
