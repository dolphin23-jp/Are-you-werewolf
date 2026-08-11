"""Payoff-only compression of audited policy populations.

Strategic retention is deliberately downstream of game learning. It uses only
empirical terminal-payoff measurements to decide which already-trained policies
remain in a bounded restricted population. It does not inspect semantic actions,
roles claimed in chat, policy age, or any human-authored strategy labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from app.engine.roles import Team
from app.training.meta_strategy import solve_logit_response_mixture
from app.training.population_payoff import PopulationPayoffTable
from app.training.retention_audit import TeamRetentionDiagnostic, diagnose_team_retention


@dataclass(frozen=True)
class RetentionSubsetCandidate:
    """One candidate compression and its held-out exploitability diagnostics."""

    selected_policy_ids: tuple[str, ...]
    held_out_policy_ids: tuple[str, ...]
    max_held_out_gain: float
    max_held_out_ci95_high: float
    sum_positive_held_out_gain: float
    held_out_diagnostics: tuple[TeamRetentionDiagnostic, ...]


@dataclass(frozen=True)
class StrategicRetentionSelection:
    """Best fixed-size subset among a fully audited same-faction candidate set."""

    team: Team
    candidate_policy_ids: tuple[str, ...]
    keep: int
    selected_policy_ids: tuple[str, ...]
    held_out_policy_ids: tuple[str, ...]
    max_held_out_gain: float
    max_held_out_ci95_high: float
    sum_positive_held_out_gain: float
    candidates: tuple[RetentionSubsetCandidate, ...]


def retention_triggered_by_fixed_audit(
    *,
    saved_ci95_low: float,
    fixed_ci95_low: float,
    threshold: float = 0.0,
) -> bool:
    """Require a positive challenger lower bound under both saved and fixed meta."""

    return saved_ci95_low > threshold and fixed_ci95_low > threshold


def select_team_population_subset(
    table: PopulationPayoffTable,
    *,
    current_population: dict[Team, tuple[str, ...]],
    team: Team,
    candidate_policy_ids: tuple[str, ...],
    keep: int,
    temperature: float,
    iterations: int,
    damping: float,
) -> StrategicRetentionSelection:
    """Choose a bounded same-team subset that best suppresses held-out deviations.

    All candidate policies must already have unilateral payoff slices against the
    supplied opponent populations. For every subset, the empirical meta-strategy
    is re-solved using the exact same solver settings as the source population.
    Policies left out of that subset are then evaluated as external deviations.

    The primary objective minimizes the largest held-out mean deviation gain.
    Ties are broken by the largest held-out 95% CI upper bound, then the sum of
    positive held-out gains, then lexicographically for deterministic replay.
    """

    _validate_inputs(
        current_population=current_population,
        team=team,
        candidate_policy_ids=candidate_policy_ids,
        keep=keep,
        temperature=temperature,
        iterations=iterations,
        damping=damping,
    )

    results: list[RetentionSubsetCandidate] = []
    for selected in combinations(candidate_policy_ids, keep):
        selected_set = set(selected)
        held_out = tuple(
            policy_id
            for policy_id in candidate_policy_ids
            if policy_id not in selected_set
        )
        populations = dict(current_population)
        populations[team] = selected
        strategy = solve_logit_response_mixture(
            table,
            village=populations[Team.VILLAGE],
            werewolf=populations[Team.WEREWOLF],
            fox=populations[Team.FOX],
            temperature=temperature,
            iterations=iterations,
            damping=damping,
        )
        diagnostics = tuple(
            diagnose_team_retention(table, strategy, team, policy_id)
            for policy_id in held_out
        )
        gains = tuple(item.challenger_gain_vs_mixture for item in diagnostics)
        ci_highs = tuple(item.challenger_gain_ci95_high for item in diagnostics)
        results.append(
            RetentionSubsetCandidate(
                selected_policy_ids=selected,
                held_out_policy_ids=held_out,
                max_held_out_gain=max(gains),
                max_held_out_ci95_high=max(ci_highs),
                sum_positive_held_out_gain=sum(max(0.0, gain) for gain in gains),
                held_out_diagnostics=diagnostics,
            )
        )

    best = min(
        results,
        key=lambda item: (
            item.max_held_out_gain,
            item.max_held_out_ci95_high,
            item.sum_positive_held_out_gain,
            item.selected_policy_ids,
        ),
    )
    return StrategicRetentionSelection(
        team=team,
        candidate_policy_ids=candidate_policy_ids,
        keep=keep,
        selected_policy_ids=best.selected_policy_ids,
        held_out_policy_ids=best.held_out_policy_ids,
        max_held_out_gain=best.max_held_out_gain,
        max_held_out_ci95_high=best.max_held_out_ci95_high,
        sum_positive_held_out_gain=best.sum_positive_held_out_gain,
        candidates=tuple(results),
    )


def _validate_inputs(
    *,
    current_population: dict[Team, tuple[str, ...]],
    team: Team,
    candidate_policy_ids: tuple[str, ...],
    keep: int,
    temperature: float,
    iterations: int,
    damping: float,
) -> None:
    if set(current_population) != set(Team):
        raise ValueError("current_population must contain exactly the three factions")
    if any(not current_population[current_team] for current_team in Team):
        raise ValueError("each current faction population must be non-empty")
    if len(candidate_policy_ids) != len(set(candidate_policy_ids)):
        raise ValueError("candidate_policy_ids must be unique")
    if not candidate_policy_ids:
        raise ValueError("candidate_policy_ids must be non-empty")
    if keep <= 0 or keep >= len(candidate_policy_ids):
        raise ValueError("keep must leave at least one candidate held out")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
