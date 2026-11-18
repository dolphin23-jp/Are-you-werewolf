"""Saying a verdict again is not a new verdict.

The engine keeps every version of a published result and only drops a same-day
repeat, so a seer who mentions yesterday's white again today -- in passing,
without naming the night -- leaves a standing version dated today. Read
naively that version is a result for *last* night: the curse arithmetic loses
the night it was about, the memo moves it to the wrong row, and the timeline
check reports two divinations on one night that nobody made.

`ObservationSet` therefore dates a plain restatement by the statement it
repeats (docs/real_game_log_reasoning_plan.md, principle 3). A correction, a
re-publication after a retraction, or a restatement that names a night of its
own are new statements and keep their own date.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective
from app.ai.reasoning.solver import build_solver, has_role
from app.ai.reasoning.timeline import find_timeline_conflicts
from app.engine.roles import RoleName
from app.engine.state import GameState
from tests.ai.reasoning import game1_boards as g1
from tests.ai.reasoning.solver import boards


def _day_three_with_restatement(**restatement: object) -> GameState:
    """Day 3: ダイキ publishes カイト for night 2 and mentions シオン again."""
    state = g1.day_two_morning()
    boards.execute(state, g1.REN, day=2)
    boards.verdict(state, g1.DAIKI, "seer", g1.KAITO, False, day=3, referenced_day=2)
    boards.verdict(state, g1.DAIKI, "seer", g1.SHION, False, day=3, **restatement)  # type: ignore[arg-type]
    state.day = 3
    return state


def _shion(observations: ObservationSet):  # type: ignore[no-untyped-def]
    return next(v for v in observations.verdicts_by(g1.DAIKI) if v.target_id == g1.SHION)


def test_a_restatement_keeps_the_day_and_night_it_repeats():
    verdict = _shion(ObservationSet.from_state(_day_three_with_restatement()))

    assert (verdict.day, verdict.referenced_day, verdict.source_night) == (2, 1, 1)


def test_a_restatement_does_not_invent_a_second_divination_for_last_night():
    observations = ObservationSet.from_state(_day_three_with_restatement())

    assert [c.kind for c in find_timeline_conflicts(observations)] == []


def test_a_restatement_does_not_undo_the_curse_arithmetic():
    observations = ObservationSet.from_state(_day_three_with_restatement())

    public = build_solver(observations, CommonPublicPerspective())

    assert public.is_possible(has_role(g1.DAIKI, RoleName.SEER)) is False


def test_a_restatement_that_names_its_own_night_keeps_it():
    verdict = _shion(ObservationSet.from_state(_day_three_with_restatement(referenced_day=2)))

    # The claimant now says night 2. That is a new account of the timing, and
    # the timeline check is the place to notice it -- not this projection.
    assert (verdict.day, verdict.referenced_day) == (3, 2)


def test_a_correction_is_a_new_statement():
    state = g1.day_two_morning()
    boards.correct_verdict(state, g1.DAIKI, "seer", g1.SHION, True, day=3)
    state.day = 3

    verdict = _shion(ObservationSet.from_state(state))

    assert (verdict.day, verdict.referenced_day, verdict.is_werewolf) == (3, None, True)


def test_a_verdict_published_again_after_a_retraction_is_a_new_statement():
    state = g1.day_two_morning()
    boards.retract_verdict(state, g1.DAIKI, "seer", g1.SHION, day=3)
    boards.verdict(state, g1.DAIKI, "seer", g1.SHION, False, day=4)
    state.day = 4

    verdict = _shion(ObservationSet.from_state(state))

    assert (verdict.day, verdict.referenced_day) == (4, None)


def test_an_unrestated_verdict_is_untouched():
    verdict = _shion(ObservationSet.from_state(g1.day_two_morning()))

    assert (verdict.day, verdict.referenced_day) == (2, 1)
