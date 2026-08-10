from pathlib import Path

import pytest

from app.engine.roles import Team
from app.training.meta_strategy import PolicyWeight, PopulationMetaStrategy
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable
from app.training.retention_audit import (
    build_retention_profiles,
    diagnose_team_retention,
)


def _strategy() -> PopulationMetaStrategy:
    return PopulationMetaStrategy(
        village=(PolicyWeight("v0", 0.5), PolicyWeight("v1", 0.5)),
        werewolf=(PolicyWeight("w0", 0.5), PolicyWeight("w1", 0.5)),
        fox=(PolicyWeight("f0", 0.5), PolicyWeight("f1", 0.5)),
    )


def test_build_retention_profiles_adds_only_unilateral_slices():
    strategy = _strategy()
    profiles = build_retention_profiles(
        strategy,
        {
            Team.VILLAGE: "vx",
            Team.WEREWOLF: "wx",
            Team.FOX: "fx",
        },
    )

    assert len(profiles) == 20
    assert PolicyProfile("vx", "w0", "f0") in profiles
    assert PolicyProfile("v0", "wx", "f0") in profiles
    assert PolicyProfile("v0", "w0", "fx") in profiles
    assert PolicyProfile("vx", "wx", "f0") not in profiles
    assert PolicyProfile("vx", "w0", "fx") not in profiles


def test_team_retention_compares_dropped_policy_with_frozen_mixture(
    tmp_path: Path,
):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    strategy = PopulationMetaStrategy(
        village=(PolicyWeight("v0", 0.5), PolicyWeight("v1", 0.5)),
        werewolf=(PolicyWeight("w0", 1.0),),
        fox=(PolicyWeight("f0", 1.0),),
    )
    for _ in range(4):
        table.record_result(
            PolicyProfile("v0", "w0", "f0"),
            winner=Team.WEREWOLF,
            is_draw=False,
            days=3,
        )
        table.record_result(
            PolicyProfile("v1", "w0", "f0"),
            winner=Team.VILLAGE,
            is_draw=False,
            days=3,
        )
        table.record_result(
            PolicyProfile("vx", "w0", "f0"),
            winner=Team.VILLAGE,
            is_draw=False,
            days=3,
        )

    result = diagnose_team_retention(table, strategy, Team.VILLAGE, "vx")

    assert result.mixture.mean == 0.0
    assert result.mixture.standard_error == 0.0
    assert result.best_current_policy_id == "v1"
    assert result.best_current_policy_payoff == 1.0
    assert result.challenger.mean == 1.0
    assert result.challenger_gain_vs_mixture == 1.0
    assert result.challenger_gain_vs_best_current == 0.0
    assert result.challenger_gain_ci95_low == 1.0
    assert result.challenger_gain_ci95_high == 1.0


def test_team_retention_reports_sampling_uncertainty(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    strategy = PopulationMetaStrategy(
        village=(PolicyWeight("v0", 1.0),),
        werewolf=(PolicyWeight("w0", 1.0),),
        fox=(PolicyWeight("f0", 1.0),),
    )
    current = PolicyProfile("v0", "w0", "f0")
    challenger = PolicyProfile("vx", "w0", "f0")
    for winner in (Team.VILLAGE, Team.WEREWOLF, Team.VILLAGE, Team.WEREWOLF):
        table.record_result(current, winner=winner, is_draw=False, days=3)
        table.record_result(challenger, winner=winner, is_draw=False, days=3)

    result = diagnose_team_retention(table, strategy, Team.VILLAGE, "vx")

    assert result.mixture.mean == 0.0
    assert result.challenger.mean == 0.0
    assert result.mixture.standard_error > 0.0
    assert result.challenger.standard_error > 0.0
    assert result.challenger_gain_ci95_low < 0.0
    assert result.challenger_gain_ci95_high > 0.0


def test_retention_challenger_must_be_dropped():
    with pytest.raises(ValueError, match="outside the current population"):
        diagnose_team_retention(
            PopulationPayoffTable(Path("unused.json")),
            PopulationMetaStrategy(
                village=(PolicyWeight("v0", 1.0),),
                werewolf=(PolicyWeight("w0", 1.0),),
                fox=(PolicyWeight("f0", 1.0),),
            ),
            Team.VILLAGE,
            "v0",
        )
