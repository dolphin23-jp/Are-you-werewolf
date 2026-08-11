from pathlib import Path

import pytest

from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.strategic_retention import (
    retention_triggered_by_fixed_audit,
    select_team_population_subset,
)


def _record(
    table: PopulationPayoffTable,
    profile: PolicyProfile,
    *,
    winner: Team | None,
    is_draw: bool = False,
    games: int = 4,
) -> None:
    for _ in range(games):
        table.record_result(
            profile,
            winner=winner,
            is_draw=is_draw,
            days=3,
        )


def test_retention_trigger_requires_both_fixed_lower_bounds_positive():
    assert retention_triggered_by_fixed_audit(
        saved_ci95_low=0.04,
        fixed_ci95_low=0.02,
    )
    assert not retention_triggered_by_fixed_audit(
        saved_ci95_low=-0.01,
        fixed_ci95_low=0.02,
    )
    assert not retention_triggered_by_fixed_audit(
        saved_ci95_low=0.04,
        fixed_ci95_low=0.0,
    )


def test_strategic_retention_keeps_nonreplaceable_payoff_response(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    current = {
        Team.VILLAGE: ("v0",),
        Team.WEREWOLF: ("w0",),
        Team.FOX: ("f0", "f1"),
    }
    _record(
        table,
        PolicyProfile("v0", "w0", "f0"),
        winner=Team.VILLAGE,
    )
    _record(
        table,
        PolicyProfile("v0", "w0", "f1"),
        winner=None,
        is_draw=True,
    )
    _record(
        table,
        PolicyProfile("v0", "w0", "fx"),
        winner=Team.FOX,
    )

    selection = select_team_population_subset(
        table,
        current_population=current,
        team=Team.FOX,
        candidate_policy_ids=("f0", "f1", "fx"),
        keep=2,
        temperature=0.25,
        iterations=20,
        damping=0.5,
    )

    assert selection.selected_policy_ids == ("f1", "fx")
    assert selection.held_out_policy_ids == ("f0",)
    assert selection.max_held_out_gain < 0.0
    assert len(selection.candidates) == 3


def test_strategic_retention_requires_complete_candidate_slices(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    current = {
        Team.VILLAGE: ("v0",),
        Team.WEREWOLF: ("w0",),
        Team.FOX: ("f0", "f1"),
    }
    _record(
        table,
        PolicyProfile("v0", "w0", "f0"),
        winner=Team.VILLAGE,
    )
    _record(
        table,
        PolicyProfile("v0", "w0", "f1"),
        winner=Team.FOX,
    )

    with pytest.raises(ValueError, match="complete measured payoff cube|missing fox"):
        select_team_population_subset(
            table,
            current_population=current,
            team=Team.FOX,
            candidate_policy_ids=("f0", "f1", "fx"),
            keep=2,
            temperature=0.25,
            iterations=10,
            damping=0.5,
        )
