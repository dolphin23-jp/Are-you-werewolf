"""The viewpoint is the only gate between the true assignment and a deduction."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.reasoning import perspectives as perspectives_module
from app.ai.reasoning.perspectives import (
    CommonPublicPerspective,
    GenericVillagerPerspective,
    PerspectiveLeakError,
    PlayerPrivatePerspective,
    TrueWorldPerspective,
    perspective_for,
    require_in_game,
)
from app.engine.roles import RoleName
from tests.ai.reasoning.solver.conftest import Board


def test_the_public_view_knows_nobodys_card(board: Board):
    assert CommonPublicPerspective().known_roles(board.observations) == {}


def test_a_generic_villager_assumes_only_their_own_seat(board: Board):
    seat = board.one(RoleName.WEREWOLF)

    # Deliberately built for a wolf seat: this perspective answers "what could an
    # ordinary villager here conclude", so it must not consult the real card.
    known = GenericVillagerPerspective(seat).known_roles(board.observations)

    assert known == {seat: RoleName.VILLAGER}


def test_a_private_view_reveals_exactly_the_seats_dealt_knowledge(board: Board):
    wolves = board.seats(RoleName.WEREWOLF)
    freemasons = board.seats(RoleName.FREEMASON)
    madman = board.one(RoleName.MADMAN)
    fox = board.one(RoleName.FOX)
    observations = board.observations

    assert PlayerPrivatePerspective(wolves[0]).known_roles(observations) == {
        wolf: RoleName.WEREWOLF for wolf in wolves
    }
    assert PlayerPrivatePerspective(freemasons[0]).known_roles(observations) == {
        mason: RoleName.FREEMASON for mason in freemasons
    }
    assert PlayerPrivatePerspective(madman).known_roles(observations) == {
        madman: RoleName.MADMAN
    }
    assert PlayerPrivatePerspective(fox).known_roles(observations) == {fox: RoleName.FOX}


def test_the_omniscient_view_is_flagged_and_unreachable_from_normal_code(board: Board):
    debug = TrueWorldPerspective()

    assert debug.known_roles(board.observations) == dict(board.observations.true_roles)
    assert debug.debug_only is True
    # The factory AI code calls can only ever hand back an in-game viewpoint.
    assert perspective_for(None).debug_only is False
    assert perspective_for("p3").debug_only is False
    assert not isinstance(perspective_for("p3"), TrueWorldPerspective)


def test_an_omniscient_view_is_rejected_where_a_player_decision_would_use_it():
    with pytest.raises(PerspectiveLeakError):
        require_in_game(TrueWorldPerspective())

    for perspective in (
        CommonPublicPerspective(),
        GenericVillagerPerspective("p3"),
        PlayerPrivatePerspective("p3"),
    ):
        assert require_in_game(perspective) is perspective


def test_perspective_ids_distinguish_viewpoints():
    ids = {
        CommonPublicPerspective().perspective_id,
        GenericVillagerPerspective("p3").perspective_id,
        PlayerPrivatePerspective("p3").perspective_id,
        PlayerPrivatePerspective("p4").perspective_id,
        TrueWorldPerspective().perspective_id,
    }

    # Cache keys are built from these, so two viewpoints sharing an id would
    # serve one seat's deductions to another.
    assert len(ids) == 5


def test_the_solver_never_reaches_the_llm_layer():
    solver_sources = list(
        (Path(perspectives_module.__file__).parent / "solver").rglob("*.py")
    )
    assert solver_sources

    for source in solver_sources:
        text = source.read_text(encoding="utf-8")
        for forbidden in ("app.ai.provider", "LLMProvider", "generate_structured", "openai"):
            assert forbidden not in text, f"{source.name} reaches the LLM layer"
