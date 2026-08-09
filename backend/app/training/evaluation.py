"""Evaluation harness for measuring one candidate faction against fixed opponents."""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeRunner
from app.training.policy_contract import LearnedPolicyModel


@dataclass(frozen=True)
class FactionEvaluationStats:
    team: Team
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    mean_days: float


def evaluate_faction(
    player_specs: list[PlayerSpec],
    candidate: LearnedPolicyModel,
    opponent: LearnedPolicyModel,
    *,
    team: Team,
    seeds: tuple[int, ...],
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
) -> FactionEvaluationStats:
    """Give only ``team`` to the candidate and hold all other factions fixed."""
    if not seeds:
        raise ValueError("evaluation requires at least one seed")

    wins = 0
    draws = 0
    total_days = 0
    for seed in seeds:
        result = LearnedEpisodeRunner(
            player_specs,
            opponent,
            team_models={team: candidate},
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        ).run(seed)
        wins += int(result.winner is team)
        draws += int(result.is_draw)
        total_days += result.days

    games = len(seeds)
    losses = games - wins - draws
    return FactionEvaluationStats(
        team=team,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games,
        mean_days=total_days / games,
    )
