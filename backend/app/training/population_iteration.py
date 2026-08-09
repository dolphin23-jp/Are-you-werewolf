"""One PSRO-style empirical population iteration for the NumPy baseline.

The iteration composes the existing population-training pieces:

1. choose recent faction-eligible immutable policies,
2. complete their three-faction empirical payoff cube,
3. optionally spend extra evaluation games on the most uncertain profiles,
4. solve the current empirical meta-strategy,
5. train one independent approximate response oracle for each faction,
6. append those three faction-specialized policies to the immutable pool.

This is still an approximate empirical population loop. The logit-response meta
solver is not a Nash/JPSRO solver, and PPO oracles are not exact best responses.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import (
    PopulationMetaDiagnostics,
    PopulationMetaStrategy,
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.numpy_trainer import PPOConfig
from app.training.policy_pool import NumpyPolicyPool
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffTable,
    ProfilePayoff,
    evaluate_policy_profile,
)
from app.training.psro_oracle import OracleTrainingStats, train_population_oracle


@dataclass(frozen=True)
class PopulationIterationStats:
    village_policy_ids: tuple[str, ...]
    werewolf_policy_ids: tuple[str, ...]
    fox_policy_ids: tuple[str, ...]
    measured_profiles: int
    new_profile_games: int
    adaptive_games: int
    meta_strategy: PopulationMetaStrategy
    diagnostics: PopulationMetaDiagnostics
    oracles: tuple[OracleTrainingStats, ...]


def run_population_iteration(
    player_specs: list[PlayerSpec],
    pool: NumpyPolicyPool,
    table: PopulationPayoffTable,
    *,
    recent_policies: int = 3,
    games_per_profile: int = 3,
    extra_games: int = 0,
    uncertainty_prior: float = 0.5,
    evaluation_seed: int = 30000,
    oracle_seed: int = 40000,
    opponent_seed: int = 1,
    oracle_episodes: int = 2,
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
    meta_temperature: float = 0.25,
    meta_iterations: int = 100,
    meta_damping: float = 0.5,
    ppo_config: PPOConfig | None = None,
) -> PopulationIterationStats:
    """Measure the restricted game, solve it, and add three response oracles."""
    if recent_policies <= 0:
        raise ValueError("recent_policies must be positive")
    if games_per_profile <= 0:
        raise ValueError("games_per_profile must be positive")
    if extra_games < 0:
        raise ValueError("extra_games cannot be negative")
    if uncertainty_prior <= 0:
        raise ValueError("uncertainty_prior must be positive")
    if oracle_episodes <= 0:
        raise ValueError("oracle_episodes must be positive")
    if not pool.entries:
        raise ValueError("population iteration requires a non-empty policy pool")

    village_ids = pool.policy_ids_for_team(Team.VILLAGE, last=recent_policies)
    werewolf_ids = pool.policy_ids_for_team(Team.WEREWOLF, last=recent_policies)
    fox_ids = pool.policy_ids_for_team(Team.FOX, last=recent_policies)
    if not village_ids or not werewolf_ids or not fox_ids:
        raise ValueError("each faction must have at least one eligible policy")

    profiles = tuple(
        PolicyProfile(village, werewolf, fox)
        for village, werewolf, fox in itertools.product(
            village_ids,
            werewolf_ids,
            fox_ids,
        )
    )
    new_profile_games = 0
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

    for _ in range(extra_games):
        selected = max(
            profiles,
            key=lambda profile: (
                _require_record(table, profile).max_posterior_payoff_std(
                    prior=uncertainty_prior
                ),
                -_require_record(table, profile).games,
                profile,
            ),
        )
        before = _require_record(table, selected)
        seed = _profile_seed(evaluation_seed, selected) + before.games
        evaluate_policy_profile(
            player_specs,
            pool,
            table,
            selected,
            seeds=(seed,),
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        )

    meta_strategy = solve_logit_response_mixture(
        table,
        village=village_ids,
        werewolf=werewolf_ids,
        fox=fox_ids,
        temperature=meta_temperature,
        iterations=meta_iterations,
        damping=meta_damping,
    )
    diagnostics = diagnose_meta_strategy(table, meta_strategy)

    oracles = tuple(
        train_population_oracle(
            player_specs,
            pool,
            meta_strategy,
            team=team,
            episodes=oracle_episodes,
            start_seed=oracle_seed + team_index * oracle_episodes,
            opponent_seed=opponent_seed + team_index,
            ppo_config=ppo_config,
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        )
        for team_index, team in enumerate(Team)
    )
    return PopulationIterationStats(
        village_policy_ids=village_ids,
        werewolf_policy_ids=werewolf_ids,
        fox_policy_ids=fox_ids,
        measured_profiles=len(profiles),
        new_profile_games=new_profile_games,
        adaptive_games=extra_games,
        meta_strategy=meta_strategy,
        diagnostics=diagnostics,
        oracles=oracles,
    )


def _require_record(
    table: PopulationPayoffTable,
    profile: PolicyProfile,
) -> ProfilePayoff:
    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile measurement did not produce a payoff record")
    return record


def _profile_seed(base_seed: int, profile: PolicyProfile) -> int:
    """Stable profile-local seed independent of enumeration order."""
    payload = f"{profile.village}|{profile.werewolf}|{profile.fox}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return base_seed + int.from_bytes(digest, "big") % 1_000_000_000
