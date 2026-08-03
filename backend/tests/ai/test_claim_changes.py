"""Withdrawing and amending what you already said, through the real path.

Sliding from seer to medium, taking a verdict back, correcting black to white --
these are moves players make, and until now the only way into the record was an
original claim. Everything else had to be inferred from prose, which meant the
ledger and the table could disagree about whether a retraction had happened at
all.

These run through `AICoordinator._speak` and the chat route rather than calling
the parser directly: a conversion that works in isolation and never fires in a
game is not a feature.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.claims import build_claim_drafts, register_claim_drafts
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.schemas import (
    ClaimAction,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PublicResultClaim,
    ResultAction,
    SummaryOutput,
    VoteOutput,
)
from app.engine.game import GameError
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from tests.conftest import make_controller

AI_IDS = ["p1", "p2"]


class ScriptedProvider:
    """Returns one prepared discussion output, then ordinary filler."""

    def __init__(self, output: DiscussionOutput) -> None:
        self.output = output

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            return self.output
        if response_schema is VoteOutput:
            return VoteOutput(vote_target="p2")
        if response_schema is NightActionOutput:
            return NightActionOutput(target="p2")
        return SummaryOutput(summary="要約")


def _controller():  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 2
    return controller


def _speak(controller, output: DiscussionOutput, speaker: str = "p1"):  # type: ignore[no-untyped-def]
    coordinator = AICoordinator(
        controller.state, AI_IDS, ScriptedProvider(output), seed=4
    )
    asyncio.run(
        coordinator._speak(controller, controller.state, speaker, "initial_view")
    )
    return coordinator


# -- role claims --


def test_a_declared_retraction_removes_the_standing_co():
    controller = _controller()
    controller.co("p1", "seer")
    assert PublicFactLedger(controller.state).claimed_role_of("p1") is RoleName.SEER

    _speak(
        controller,
        DiscussionOutput(
            public_message="すみません、先ほどの話は取り下げます。",
            claim_action=ClaimAction(action="retract"),
        ),
    )

    assert PublicFactLedger(controller.state).claimed_role_of("p1") is None


def test_a_declared_slide_replaces_the_role_and_keeps_the_history():
    controller = _controller()
    controller.co("p1", "seer")

    _speak(
        controller,
        DiscussionOutput(
            public_message="実は違います。",
            claim_action=ClaimAction(action="switch", role="medium"),
        ),
    )

    ledger = PublicFactLedger(controller.state)
    assert ledger.claimed_role_of("p1") is RoleName.MEDIUM
    # The seer claim is still in the log: "used to claim seer" stays answerable.
    assert any(
        event.role is RoleName.SEER and event.event_type is SpeechEventType.ROLE_CLAIM
        for event in controller.state.speech_events
    )


def test_retracting_a_co_nobody_made_records_nothing():
    controller = _controller()

    _speak(
        controller,
        DiscussionOutput(
            public_message="撤回します。", claim_action=ClaimAction(action="retract")
        ),
    )

    assert not [
        event
        for event in controller.state.speech_events
        if event.event_type is SpeechEventType.ROLE_RETRACTION
    ]


def test_the_canonical_sentence_is_written_into_the_message():
    """The visible text and the ledger have to agree, or the table is arguing
    about a retraction nobody said out loud."""
    controller = _controller()
    controller.co("p1", "seer")
    output = DiscussionOutput(
        public_message="考え直しました。",
        claim_action=ClaimAction(action="retract"),
    )

    _speak(controller, output)

    assert "撤回" in output.public_message


# -- results --


def test_a_declared_result_retraction_removes_the_live_verdict():
    controller = _controller()
    controller.co("p1", "seer")
    controller.public_result("p1", "seer", "p2", True)
    assert PublicFactLedger(controller.state).find_result("p1", "seer", "p2")

    _speak(
        controller,
        DiscussionOutput(
            public_message="判定を見直します。",
            result_actions=[
                ResultAction(action="retract", result_type="seer", target_id="p2")
            ],
        ),
    )

    assert PublicFactLedger(controller.state).find_result("p1", "seer", "p2") is None


def test_a_declared_correction_replaces_the_colour():
    controller = _controller()
    controller.co("p1", "seer")
    controller.public_result("p1", "seer", "p2", True)

    _speak(
        controller,
        DiscussionOutput(
            public_message="訂正があります。",
            result_actions=[
                ResultAction(
                    action="correct",
                    result_type="seer",
                    target_id="p2",
                    is_werewolf=False,
                )
            ],
        ),
    )

    live = PublicFactLedger(controller.state).find_result("p1", "seer", "p2")
    assert live is not None
    assert live.is_werewolf is False


def test_correcting_a_result_that_was_never_published_records_nothing():
    controller = _controller()
    controller.co("p1", "seer")

    _speak(
        controller,
        DiscussionOutput(
            public_message="訂正します。",
            result_actions=[
                ResultAction(
                    action="correct",
                    result_type="seer",
                    target_id="p2",
                    is_werewolf=False,
                )
            ],
        ),
    )

    assert PublicFactLedger(controller.state).find_result("p1", "seer", "p2") is None


def test_a_declared_night_is_kept_on_the_published_result():
    controller = _controller()
    controller.co("p1", "seer")

    _speak(
        controller,
        DiscussionOutput(
            public_message="結果を伝えます。",
            public_results=[
                PublicResultClaim(
                    result_type="seer",
                    target_id="p2",
                    is_werewolf=True,
                    referenced_day=0,
                )
            ],
        ),
    )

    result = PublicFactLedger(controller.state).find_result("p1", "seer", "p2")
    assert result is not None
    assert result.referenced_day == 0
    assert result.source_night == 0


# -- the human path --


def test_a_human_can_withdraw_a_co_in_plain_words():
    controller = _controller()
    controller.co("p0", "seer")
    ledger = PublicFactLedger(controller.state)

    drafts = build_claim_drafts(
        DiscussionOutput(public_message="占いCOを撤回します。すみません。"),
        ledger,
        speaker_id="p0",
    )
    register_claim_drafts(controller, "p0", drafts, "m1")

    assert PublicFactLedger(controller.state).claimed_role_of("p0") is None


def test_the_same_words_with_nothing_to_withdraw_change_nothing():
    controller = _controller()
    ledger = PublicFactLedger(controller.state)

    drafts = build_claim_drafts(
        DiscussionOutput(public_message="COを撤回します。"), ledger, speaker_id="p0"
    )

    assert not [
        draft
        for draft in drafts
        if draft.event_type is SpeechEventType.ROLE_RETRACTION
    ]


def test_an_ambiguous_second_thought_does_not_delete_a_claim():
    """Narrow on purpose. A hedge is not a retraction, and treating it as one
    silently removes a fact the whole table is still reasoning from."""
    controller = _controller()
    controller.co("p0", "seer")
    ledger = PublicFactLedger(controller.state)

    drafts = build_claim_drafts(
        DiscussionOutput(public_message="さっきの話、間違っていたかもしれません。"),
        ledger,
        speaker_id="p0",
    )

    assert not [
        draft
        for draft in drafts
        if draft.event_type is SpeechEventType.ROLE_RETRACTION
    ]


def test_withdrawing_the_result_when_two_are_standing_needs_a_name():
    controller = _controller()
    controller.co("p0", "seer")
    controller.public_result("p0", "seer", "p1", True)
    controller.public_result("p0", "seer", "p2", False)
    ledger = PublicFactLedger(controller.state)

    ambiguous = build_claim_drafts(
        DiscussionOutput(public_message="占い結果を撤回します。"),
        ledger,
        speaker_id="p0",
    )
    named = build_claim_drafts(
        DiscussionOutput(public_message="p2への占い結果を撤回します。"),
        ledger,
        speaker_id="p0",
    )

    assert ambiguous == []
    assert [draft.target_id for draft in named] == ["p2"]


# -- failures are not swallowed --


def test_an_unexpected_error_is_not_hidden_behind_a_running_game():
    """`except Exception: continue` used to make a real bug look like a claim
    the player never made."""

    class Exploding:
        def record_speech_event(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    drafts = build_claim_drafts(
        DiscussionOutput(public_message="占いCOします。"),
        PublicFactLedger(_controller().state),
        speaker_id="p1",
    )

    with pytest.raises(RuntimeError):
        register_claim_drafts(Exploding(), "p1", drafts, "m1")


def test_an_illegal_move_is_tolerated_and_leaves_the_log_clean():
    class Rejecting:
        def record_speech_event(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise GameError("wrong phase")

    drafts = build_claim_drafts(
        DiscussionOutput(public_message="占いCOします。"),
        PublicFactLedger(_controller().state),
        speaker_id="p1",
    )

    assert register_claim_drafts(Rejecting(), "p1", drafts, "m1") == []
