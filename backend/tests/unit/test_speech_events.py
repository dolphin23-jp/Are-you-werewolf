"""Claim and result history, derived from the speech-event log.

The point of the log is that these questions have answers at all: who claimed
what, in which message, what they claimed before that, and which verdict is the
correction rather than the original.
"""

from __future__ import annotations

import pytest

from app.engine.game import GameError
from app.engine.roles import RoleName
from app.engine.speech_events import (
    CLAIM_CONFIDENCE_THRESHOLD,
    ResultStatus,
    RoleClaimStatus,
    SpeechEventType,
    current_role_claim,
    events_for_message,
    result_versions,
    role_claim_history,
)
from tests.conftest import make_controller


def _started(seed: int = 4, day: int = 1):  # type: ignore[no-untyped-def]
    controller = make_controller(seed=seed)
    controller.state.day = day
    return controller


# -- role claims --


@pytest.mark.parametrize(
    "role",
    [RoleName.SEER, RoleName.MEDIUM, RoleName.HUNTER, RoleName.FREEMASON],
)
def test_each_public_co_is_recorded_with_its_source_message(role: RoleName):
    controller = _started()

    controller.co("p1", role.value, source_message_id="m7")

    claim = current_role_claim(controller.state.speech_events, "p1")
    assert (claim.role, claim.status) == (role, RoleClaimStatus.CLAIMED)
    assert claim.source_message_id == "m7"
    assert controller.state.co_declarations[0].source_message_id == "m7"


def test_a_freemason_co_and_partner_reveal_are_two_events_on_one_message():
    controller = _started()

    controller.co("p1", RoleName.FREEMASON.value, source_message_id="m3")
    controller.claim_freemason_partner("p1", "p2", source_message_id="m3")

    events = events_for_message(controller.state.speech_events, "m3")
    assert [event.event_type for event in events] == [
        SpeechEventType.ROLE_CLAIM,
        SpeechEventType.PARTNER_CLAIM,
    ]
    relation = controller.state.freemason_partner_claims[0]
    assert (relation.partner_id, relation.confirmed) == ("p2", False)

    controller.claim_freemason_partner("p2", "p1", source_message_id="m4")

    assert controller.state.freemason_partner_claims[0].confirmed is True


def test_a_retracted_co_leaves_the_current_composition_but_stays_in_history():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")

    controller.retract_co("p1", source_message_id="m5")

    assert current_role_claim(controller.state.speech_events, "p1") is None
    assert controller.state.co_declarations == ()
    history = role_claim_history(controller.state.speech_events, "p1")
    assert [state.status for state in history] == [
        RoleClaimStatus.CLAIMED,
        RoleClaimStatus.RETRACTED,
    ]
    assert history[0].role == RoleName.SEER


def test_a_slide_from_seer_to_freemason_records_the_role_it_came_from():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")

    controller.state.day = 2
    controller.co("p1", RoleName.FREEMASON.value, source_message_id="m9")

    claim = current_role_claim(controller.state.speech_events, "p1")
    assert (claim.role, claim.status) == (RoleName.FREEMASON, RoleClaimStatus.SWITCHED)
    assert claim.previous_role == RoleName.SEER
    assert claim.source_message_id == "m9"
    # The old API answers "who is claiming what now", so the seer claim is gone.
    assert [(c.player_id, c.claimed_role) for c in controller.state.co_declarations] == [
        ("p1", RoleName.FREEMASON)
    ]


def test_restating_the_same_co_is_a_reaffirmation_not_a_second_claim():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")

    controller.state.day = 2
    controller.co("p1", RoleName.SEER.value, source_message_id="m8")

    assert len(controller.state.co_declarations) == 1
    history = role_claim_history(controller.state.speech_events, "p1")
    assert [state.reaffirmed for state in history] == [False, True]
    assert history[-1].source_message_id == "m8"


def test_an_unknown_claimed_role_is_rejected():
    controller = _started()

    with pytest.raises(GameError):
        controller.co("p1", "wizard")


# -- ability results --


def test_a_co_and_a_result_can_share_one_message():
    controller = _started()

    controller.co("p1", RoleName.SEER.value, source_message_id="m2")
    controller.public_result("p1", "seer", "p5", True, source_message_id="m2")

    events = events_for_message(controller.state.speech_events, "m2")
    assert [event.event_type for event in events] == [
        SpeechEventType.ROLE_CLAIM,
        SpeechEventType.ABILITY_RESULT,
    ]
    assert controller.state.public_result_claims[0].source_message_id == "m2"


def test_two_results_in_one_message_are_two_separate_records():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m2")

    controller.public_result("p1", "seer", "p5", True, source_message_id="m2")
    controller.public_result("p1", "seer", "p6", False, source_message_id="m2")

    published = {
        (claim.target_id, claim.is_werewolf) for claim in controller.state.public_result_claims
    }
    assert published == {("p5", True), ("p6", False)}
    assert len(events_for_message(controller.state.speech_events, "m2")) == 3


def test_a_correction_supersedes_the_original_without_erasing_it():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")
    controller.public_result("p1", "seer", "p5", True, source_message_id="m1")

    controller.state.day = 2
    controller.correct_public_result(
        "p1", "seer", "p5", False, source_message_id="m9", referenced_day=1
    )

    versions = result_versions(controller.state.speech_events, target_id="p5")
    assert [(v.version, v.is_werewolf, v.status) for v in versions] == [
        (1, True, ResultStatus.SUPERSEDED),
        (2, False, ResultStatus.ACTIVE),
    ]
    assert versions[1].corrected is True
    assert versions[1].referenced_day == 1
    # The compatibility view reports what stands now, and only that.
    assert [(c.target_id, c.is_werewolf) for c in controller.state.public_result_claims] == [
        ("p5", False)
    ]


def test_a_retracted_result_disappears_from_the_board_but_not_from_the_log():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")
    controller.public_result("p1", "seer", "p5", True, source_message_id="m1")

    controller.retract_public_result("p1", "seer", "p5", source_message_id="m6")

    assert controller.state.public_result_claims == ()
    versions = result_versions(controller.state.speech_events, target_id="p5")
    assert [(v.version, v.status) for v in versions] == [(1, ResultStatus.RETRACTED)]


def test_restating_the_same_verdict_the_same_day_adds_no_version():
    controller = _started()
    controller.co("p1", RoleName.SEER.value, source_message_id="m1")

    controller.public_result("p1", "seer", "p5", True, source_message_id="m1")
    controller.public_result("p1", "seer", "p5", True, source_message_id="m2")

    assert len(result_versions(controller.state.speech_events, target_id="p5")) == 1


def test_seer_and_medium_results_about_the_same_player_are_separate_subjects():
    controller = _started()
    controller.co("p1", RoleName.SEER.value)
    controller.co("p2", RoleName.MEDIUM.value)

    controller.public_result("p1", "seer", "p5", False, source_message_id="m1")
    controller.public_result("p2", "medium", "p5", True, source_message_id="m2")

    assert len(result_versions(controller.state.speech_events, target_id="p5")) == 2
    assert len(controller.state.public_result_claims) == 2


def test_a_result_needs_a_known_target_and_a_real_ability():
    controller = _started()

    with pytest.raises(GameError):
        controller.public_result("p1", "seer", "p99", True)
    with pytest.raises(GameError):
        controller.public_result("p1", "hunter", "p5", True)


# -- confidence --


def test_a_low_confidence_claim_is_recorded_but_never_promoted():
    controller = _started()

    controller.record_speech_event(
        "p1",
        SpeechEventType.ROLE_CLAIM,
        role=RoleName.SEER,
        source_message_id="m1",
        confidence=0.4,
    )

    assert controller.state.co_declarations == ()
    assert current_role_claim(controller.state.speech_events, "p1") is None
    # Still on the record: something claim-shaped was said.
    event = controller.state.speech_events[0]
    assert event.confidence < CLAIM_CONFIDENCE_THRESHOLD
    assert event.is_binding is False


def test_a_confident_claim_after_an_ambiguous_one_still_registers():
    controller = _started()
    controller.record_speech_event(
        "p1", SpeechEventType.ROLE_CLAIM, role=RoleName.SEER, confidence=0.4
    )

    controller.co("p1", RoleName.SEER.value, source_message_id="m2")

    claim = current_role_claim(controller.state.speech_events, "p1")
    assert (claim.role, claim.reaffirmed) == (RoleName.SEER, False)
