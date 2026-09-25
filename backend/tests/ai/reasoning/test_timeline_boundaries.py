"""The boundary cases of *when*: who could act, who could be targeted, who could say it.

The timeline helpers distinguish an execution (which precedes that day's night)
from an overnight death (simultaneous with that night's submitted actions). They
had no direct tests at all, and one case was wrong: the first victim is not in
`night_deaths`, so they read as alive for every night after night 0. That let
"I divined the first victim on night 2" pass as a legal look.

Also here: the bluffer's story checked against both assumptions at once, and
evidence coming back when a withdrawn claim is made again.
"""

from __future__ import annotations

from app.ai.reasoning.belief import BeliefEngine
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    CommonPublicPerspective,
    PlayerPrivatePerspective,
)
from app.ai.reasoning.solver import (
    AccurateTimeline,
    CompleteResultDisclosure,
    HonestResults,
    build_solver,
)
from app.ai.reasoning.timeline import (
    ConflictKind,
    can_act_on_night,
    can_be_targeted_on_night,
    disclosable_result_nights,
    find_timeline_conflicts,
)
from app.engine.roles import RoleName
from tests.ai.reasoning.solver import boards


def _board(day: int = 3):  # type: ignore[no-untyped-def]
    return boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )


def _obs(state) -> ObservationSet:  # type: ignore[no-untyped-def]
    return ObservationSet.from_state(state)


def _kinds(state) -> set[str]:  # type: ignore[no-untyped-def]
    return {c.kind for c in find_timeline_conflicts(_obs(state))}


# -- who could act, and who could be targeted --


def test_a_seer_attacked_tonight_still_acted_tonight():
    state = _board()
    boards.die_by_attack(state, "p4", night=2)

    assert can_act_on_night(_obs(state), "p4", 2)
    assert not can_act_on_night(_obs(state), "p4", 3)


def test_a_seer_executed_today_cannot_act_tonight():
    """Execution precedes that day's night."""
    state = _board()
    boards.execute(state, "p4", day=2)

    assert can_act_on_night(_obs(state), "p4", 1)
    assert not can_act_on_night(_obs(state), "p4", 2)


def test_a_target_attacked_tonight_can_be_divined_tonight():
    state = _board()
    boards.die_by_attack(state, "p9", night=2)

    assert can_be_targeted_on_night(_obs(state), "p9", 2)
    assert not can_be_targeted_on_night(_obs(state), "p9", 3)


def test_the_fox_cursed_tonight_was_divined_tonight():
    """The look is what killed it."""
    state = _board()
    boards.die_by_curse(state, "p6", night=2)

    assert can_be_targeted_on_night(_obs(state), "p6", 2)


def test_a_target_executed_today_cannot_be_divined_tonight():
    state = _board()
    boards.execute(state, "p9", day=2)

    assert not can_be_targeted_on_night(_obs(state), "p9", 2)


def test_the_first_victim_can_never_be_divined():
    """Excluded on night 0 by the game's rules, and dead for every night after."""
    state = _board()
    boards.kill_first_victim(state, "p13")

    for night in range(0, 3):
        assert not can_be_targeted_on_night(_obs(state), "p13", night)


def test_claiming_to_have_divined_the_first_victim_is_a_conflict():
    state = _board()
    boards.kill_first_victim(state, "p13")
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p13", False, day=3, referenced_day=2)

    assert ConflictKind.TARGET_ALREADY_DEAD in _kinds(state)


# -- the result against the day it was spoken --


def test_publishing_tonights_result_before_tonight_is_a_conflict():
    state = _board(day=2)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=2, referenced_day=2)

    assert ConflictKind.FUTURE_NIGHT in _kinds(state)


def test_an_impossible_early_publication_stays_impossible_days_later():
    """Compared to the day it was *said*. Against today's date it would quietly
    turn legal once enough days had passed."""
    state = _board(day=5)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.verdict(state, "p2", "seer", "p9", True, day=2, referenced_day=2)

    assert ConflictKind.FUTURE_NIGHT in _kinds(state)


def test_publishing_an_old_result_late_is_legal():
    state = _board(day=3)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", False, day=3, referenced_day=1)

    assert ConflictKind.FUTURE_NIGHT not in _kinds(state)


# -- disclosure: what happened versus what could be said --


def test_a_seer_killed_overnight_never_had_that_nights_result_to_disclose():
    state = _board(day=3)
    boards.die_by_attack(state, "p4", night=2)

    assert disclosable_result_nights(_obs(state), "p4", 3) == frozenset({0, 1})


def test_an_unpublished_final_night_is_not_a_disclosure_failure():
    state = _board(day=3)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", False, day=1, referenced_day=0)
    boards.verdict(state, "p4", "seer", "p11", False, day=2, referenced_day=1)
    boards.die_by_attack(state, "p4", night=2)

    solver = build_solver(
        _obs(state), CommonPublicPerspective(), assumptions=(CompleteResultDisclosure("p4"),)
    )

    assert not solver.explain_contradiction().is_contradictory


def test_a_living_seer_missing_a_night_does_fail_disclosure():
    state = _board(day=3)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", False, day=1, referenced_day=0)
    boards.verdict(state, "p4", "seer", "p11", False, day=2, referenced_day=1)

    solver = build_solver(
        _obs(state), CommonPublicPerspective(), assumptions=(CompleteResultDisclosure("p4"),)
    )

    assert solver.explain_contradiction().is_contradictory


# -- the bluffer's story needs both assumptions at once --


def _story_contradicts(state, *assumptions) -> bool:  # type: ignore[no-untyped-def]
    solver = build_solver(
        _obs(state), ClaimedStoryPerspective("p1", RoleName.SEER), assumptions=assumptions
    )
    return solver.explain_contradiction().is_contradictory


def _too_many_blacks(state) -> None:  # type: ignore[no-untyped-def]
    """Four blacks on four legal nights: the calendar is fine, the colours are not
    (there are only three wolves)."""
    boards.claim(state, "p1", RoleName.SEER, day=1)
    for night, target in enumerate(("p9", "p11", "p12", "p13")):
        boards.verdict(state, "p1", "seer", target, True, day=night + 1, referenced_day=night)


def _two_looks_one_night(state) -> None:  # type: ignore[no-untyped-def]
    """Colours that could all be true, on a calendar that cannot."""
    boards.claim(state, "p1", RoleName.SEER, day=1)
    boards.verdict(state, "p1", "seer", "p9", False, day=2, referenced_day=1)
    boards.verdict(state, "p1", "seer", "p11", False, day=2, referenced_day=1)


def test_a_legal_calendar_with_impossible_colours_collapses():
    state = _board(day=4)
    _too_many_blacks(state)

    assert not _story_contradicts(state, AccurateTimeline("p1"))  # timing alone: fine
    assert _story_contradicts(state, HonestResults("p1"), AccurateTimeline("p1"))


def test_possible_colours_on_an_impossible_calendar_collapse():
    state = _board(day=2)
    _two_looks_one_night(state)

    assert not _story_contradicts(state, HonestResults("p1"))  # colours alone: fine
    assert _story_contradicts(state, HonestResults("p1"), AccurateTimeline("p1"))


def test_the_production_story_checks_both_at_once():
    """Either assumption alone misses one of the two broken stories above; the
    story a bluffer actually refreshes against must catch both."""
    from app.ai.reasoning.belief import StoryStatus, deception_state_for, refresh_story

    for day, build in ((4, _too_many_blacks), (2, _two_looks_one_night)):
        state = _board(day=day)
        build(state)
        observations = _obs(state)
        story = refresh_story(
            deception_state_for("p1", observations, claimed_role=RoleName.SEER), observations
        )
        assert story.status is StoryStatus.COLLAPSED, build.__name__


def test_a_dead_seers_unpublished_final_night_does_not_collapse_a_story():
    state = _board(day=3)
    boards.claim(state, "p1", RoleName.SEER, day=1)
    boards.verdict(state, "p1", "seer", "p9", False, day=1, referenced_day=0)
    boards.verdict(state, "p1", "seer", "p11", False, day=2, referenced_day=1)
    boards.die_by_attack(state, "p1", night=2)

    assert not _story_contradicts(state, HonestResults("p1"), AccurateTimeline("p1"))


# -- evidence comes back when a withdrawn claim is made again --


def _engine(state):  # type: ignore[no-untyped-def]
    engine = BeliefEngine("p7", PlayerPrivatePerspective("p7"))
    _observe(engine, state)
    return engine


def _observe(engine, state) -> None:  # type: ignore[no-untyped-def]
    observations = _obs(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective("p7")),
        observations,
    )


def _active(engine, category: str, subject: str) -> bool:  # type: ignore[no-untyped-def]
    return any(
        r.category == category and r.subject_id == subject for r in engine.active_evidence()
    )


def test_a_black_retracted_then_claimed_again_is_a_reason_again():
    state = _board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    engine = _engine(state)
    assert _active(engine, "published_black", "p9")

    boards.retract_verdict(state, "p4", "seer", "p9", day=2)
    _observe(engine, state)
    assert not _active(engine, "published_black", "p9")

    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    _observe(engine, state)
    assert _active(engine, "published_black", "p9")


def test_a_co_withdrawn_then_made_again_restores_its_reason():
    state = _board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    engine = _engine(state)
    assert _active(engine, "contested_claim", "p2")

    boards.retract_claim(state, "p2", day=2)
    _observe(engine, state)
    assert not _active(engine, "contested_claim", "p2")

    boards.claim(state, "p2", RoleName.SEER, day=2)
    _observe(engine, state)
    assert _active(engine, "contested_claim", "p2")


def test_correct_then_retract_then_reclaim_lands_on_the_original_colour():
    state = _board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    engine = _engine(state)

    boards.correct_verdict(state, "p4", "seer", "p9", False, day=2, referenced_day=1)
    _observe(engine, state)
    assert _active(engine, "published_white", "p9")
    assert not _active(engine, "published_black", "p9")

    boards.retract_verdict(state, "p4", "seer", "p9", day=2)
    _observe(engine, state)
    assert not _active(engine, "published_white", "p9")

    boards.verdict(state, "p4", "seer", "p9", True, day=2, referenced_day=1)
    _observe(engine, state)
    assert _active(engine, "published_black", "p9")
    assert not _active(engine, "published_white", "p9")


def test_a_contest_broken_then_restored_marks_both_claimants_again():
    state = _board(day=2)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    engine = _engine(state)

    boards.retract_claim(state, "p2", day=2)
    _observe(engine, state)
    assert not _active(engine, "contested_claim", "p4")

    boards.claim(state, "p2", RoleName.SEER, day=2)
    _observe(engine, state)
    assert _active(engine, "contested_claim", "p4")
    assert _active(engine, "contested_claim", "p2")
