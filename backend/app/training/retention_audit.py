"""Fixed-evaluation diagnostics for policies dropped from a restricted population.

This module is diagnostic only. It never changes rewards, observations, action
masks, training state, or policy checkpoints. A dropped challenger is evaluated
against the opponent mixtures of a completed population iteration so strategic
forgetting can be measured directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable, ProfilePayoff


@dataclass(frozen=True)
class PayoffEstimate:
    mean: float
    standard_error: float


@dataclass(frozen=True)
class TeamRetentionDiagnostic:
    team: Team
    challenger_policy_id: str
    mixture: PayoffEstimate
    best_current_policy_id: str
    best_current_policy_payoff: float
    challenger: PayoffEstimate
    challenger_gain_vs_mixture: float
    challenger_gain_standard_error: float
    challenger_gain_ci95_low: float
    challenger_gain_ci95_high: float
    challenger_gain_vs_best_current: float


@dataclass(frozen=True)
class RetentionDiagnostics:
    village: TeamRetentionDiagnostic
    werewolf: TeamRetentionDiagnostic
    fox: TeamRetentionDiagnostic

    def for_team(self, team: Team) -> TeamRetentionDiagnostic:
        if team is Team.VILLAGE:
            return self.village
        if team is Team.WEREWOLF:
            return self.werewolf
        if team is Team.FOX:
            return self.fox
        raise ValueError(f"unsupported team {team}")


def build_retention_profiles(
    strategy: PopulationMetaStrategy,
    challengers: dict[Team, str],
) -> tuple[PolicyProfile, ...]:
    """Return the current cube plus one unilateral challenger slice per faction."""

    _validate_challengers(strategy, challengers)
    village = tuple(item.policy_id for item in strategy.village)
    werewolf = tuple(item.policy_id for item in strategy.werewolf)
    fox = tuple(item.policy_id for item in strategy.fox)

    profiles = {
        PolicyProfile(village_id, werewolf_id, fox_id)
        for village_id in village
        for werewolf_id in werewolf
        for fox_id in fox
    }
    profiles.update(
        PolicyProfile(challengers[Team.VILLAGE], werewolf_id, fox_id)
        for werewolf_id in werewolf
        for fox_id in fox
    )
    profiles.update(
        PolicyProfile(village_id, challengers[Team.WEREWOLF], fox_id)
        for village_id in village
        for fox_id in fox
    )
    profiles.update(
        PolicyProfile(village_id, werewolf_id, challengers[Team.FOX])
        for village_id in village
        for werewolf_id in werewolf
    )
    return tuple(sorted(profiles))


def diagnose_retention(
    table: PopulationPayoffTable,
    strategy: PopulationMetaStrategy,
    challengers: dict[Team, str],
) -> RetentionDiagnostics:
    """Compare dropped challengers with a frozen current-population mixture."""

    _validate_challengers(strategy, challengers)
    diagnostics = {
        team: diagnose_team_retention(table, strategy, team, challengers[team])
        for team in Team
    }
    return RetentionDiagnostics(
        village=diagnostics[Team.VILLAGE],
        werewolf=diagnostics[Team.WEREWOLF],
        fox=diagnostics[Team.FOX],
    )


def diagnose_team_retention(
    table: PopulationPayoffTable,
    strategy: PopulationMetaStrategy,
    team: Team,
    challenger_policy_id: str,
) -> TeamRetentionDiagnostic:
    current_ids = tuple(item.policy_id for item in strategy.weights(team))
    if challenger_policy_id in current_ids:
        raise ValueError("retention challenger must be outside the current population")

    mixture = mixture_payoff_estimate(table, strategy, team)
    current_payoffs = {
        policy_id: candidate_payoff_estimate(table, strategy, team, policy_id).mean
        for policy_id in current_ids
    }
    best_policy_id, best_policy_payoff = max(
        current_payoffs.items(),
        key=lambda item: (item[1], item[0]),
    )
    challenger = candidate_payoff_estimate(
        table,
        strategy,
        team,
        challenger_policy_id,
    )
    gain = challenger.mean - mixture.mean
    gain_se = math.sqrt(
        challenger.standard_error**2 + mixture.standard_error**2
    )
    ci_radius = 1.96 * gain_se
    return TeamRetentionDiagnostic(
        team=team,
        challenger_policy_id=challenger_policy_id,
        mixture=mixture,
        best_current_policy_id=best_policy_id,
        best_current_policy_payoff=best_policy_payoff,
        challenger=challenger,
        challenger_gain_vs_mixture=gain,
        challenger_gain_standard_error=gain_se,
        challenger_gain_ci95_low=gain - ci_radius,
        challenger_gain_ci95_high=gain + ci_radius,
        challenger_gain_vs_best_current=challenger.mean - best_policy_payoff,
    )


def mixture_payoff_estimate(
    table: PopulationPayoffTable,
    strategy: PopulationMetaStrategy,
    team: Team,
) -> PayoffEstimate:
    weights = {
        current_team: {
            item.policy_id: item.probability
            for item in strategy.weights(current_team)
        }
        for current_team in Team
    }
    terms: list[tuple[float, ProfilePayoff]] = []
    for village_id, village_weight in weights[Team.VILLAGE].items():
        for werewolf_id, werewolf_weight in weights[Team.WEREWOLF].items():
            for fox_id, fox_weight in weights[Team.FOX].items():
                profile = PolicyProfile(village_id, werewolf_id, fox_id)
                record = table.get(profile)
                if record is None:
                    raise ValueError(f"missing fixed-evaluation profile {profile}")
                terms.append(
                    (
                        village_weight * werewolf_weight * fox_weight,
                        record,
                    )
                )
    return _weighted_estimate(terms, team)


def candidate_payoff_estimate(
    table: PopulationPayoffTable,
    strategy: PopulationMetaStrategy,
    team: Team,
    policy_id: str,
) -> PayoffEstimate:
    weights = {
        current_team: {
            item.policy_id: item.probability
            for item in strategy.weights(current_team)
        }
        for current_team in Team
    }
    terms: list[tuple[float, ProfilePayoff]] = []
    if team is Team.VILLAGE:
        for werewolf_id, werewolf_weight in weights[Team.WEREWOLF].items():
            for fox_id, fox_weight in weights[Team.FOX].items():
                profile = PolicyProfile(policy_id, werewolf_id, fox_id)
                record = table.get(profile)
                if record is None:
                    raise ValueError(f"missing village challenger profile {profile}")
                terms.append((werewolf_weight * fox_weight, record))
    elif team is Team.WEREWOLF:
        for village_id, village_weight in weights[Team.VILLAGE].items():
            for fox_id, fox_weight in weights[Team.FOX].items():
                profile = PolicyProfile(village_id, policy_id, fox_id)
                record = table.get(profile)
                if record is None:
                    raise ValueError(f"missing werewolf challenger profile {profile}")
                terms.append((village_weight * fox_weight, record))
    elif team is Team.FOX:
        for village_id, village_weight in weights[Team.VILLAGE].items():
            for werewolf_id, werewolf_weight in weights[Team.WEREWOLF].items():
                profile = PolicyProfile(village_id, werewolf_id, policy_id)
                record = table.get(profile)
                if record is None:
                    raise ValueError(f"missing fox challenger profile {profile}")
                terms.append((village_weight * werewolf_weight, record))
    else:
        raise ValueError(f"unsupported team {team}")
    return _weighted_estimate(terms, team)


def _weighted_estimate(
    terms: list[tuple[float, ProfilePayoff]],
    team: Team,
) -> PayoffEstimate:
    if not terms:
        raise ValueError("payoff estimate requires at least one measured profile")
    mean = sum(weight * record.mean_payoff(team) for weight, record in terms)
    variance = sum(
        weight * weight * _record_mean_variance(record, team)
        for weight, record in terms
    )
    return PayoffEstimate(mean=mean, standard_error=math.sqrt(max(0.0, variance)))


def _record_mean_variance(record: ProfilePayoff, team: Team) -> float:
    if record.games <= 1:
        return 0.0
    mean = record.mean_payoff(team)
    rewards_squared = record.games - record.draws
    sample_variance = (rewards_squared - record.games * mean * mean) / (
        record.games - 1
    )
    return max(0.0, sample_variance / record.games)


def _validate_challengers(
    strategy: PopulationMetaStrategy,
    challengers: dict[Team, str],
) -> None:
    if set(challengers) != set(Team):
        raise ValueError("retention audit requires exactly one challenger per faction")
    for team in Team:
        current_ids = {item.policy_id for item in strategy.weights(team)}
        challenger = challengers[team]
        if challenger in current_ids:
            raise ValueError(
                f"{team.value} challenger {challenger} is already in the current population"
            )
