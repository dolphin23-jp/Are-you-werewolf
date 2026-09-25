"""The candidate the table sees, the target recorded as stated, and the ballot agree.

The old enforcement rewrote the reasoning memo and stripped sentences containing
the literal "第一候補". The memo is not what the table reads, and "第一処刑候補"
does not contain that substring, so a model insisting on its own candidate got
through. These tests check the three things that must match -- the final text,
the recorded stated target, the ballot -- against a provider that always names
someone the code did not choose.

Memo-only checks would pass even with the old bug in place, so none are used.
"""

from __future__ import annotations

import asyncio

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.rendering import displayed_execution_target
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
)
from app.engine.phases import Phase
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class InsistentProvider:
    """Always argues for `rogue`, in whichever phrasing it is given."""

    def __init__(self, rogue: str, template: str) -> None:
        self.rogue = rogue
        self.template = template

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            output = DiscussionOutput(public_message=self.template.format(rogue=self.rogue))
            output.reasoning_memo.execution_target = self.rogue
            return output
        if response_schema is VoteOutput:
            return VoteOutput(vote_target=self.rogue)
        if response_schema is NightActionOutput:
            return NightActionOutput(target=self.rogue)
        return SummaryOutput(summary="要約")


def _setup(template: str):  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    runtime = ReasoningRuntime(state, AI_IDS, seed=4)
    runtime.refresh(state)
    speaker = "p1"
    decided = runtime.seats[speaker].belief.state.current_execution_target
    assert decided is not None
    rogue = next(pid for pid in state.alive_ids() if pid not in (speaker, decided))
    coordinator = AICoordinator(
        state, AI_IDS, InsistentProvider(rogue, template), seed=4, reasoning=runtime
    )
    return controller, state, runtime, coordinator, speaker, decided, rogue


def _last_public(state, speaker: str) -> str:  # type: ignore[no-untyped-def]
    return next(
        m.content
        for m in reversed(state.chat_log)
        if m.author_id == speaker and m.channel.value == "public"
    )


@pytest.mark.parametrize(
    "template",
    [
        "{rogue}を第一候補にします。",
        # Not a substring match for "第一候補" -- the phrasing the old filter missed.
        "第一処刑候補は{rogue}です。",
        "{rogue}を吊りたい！",
        "{rogue}に投票します。",
    ],
)
def test_the_displayed_candidate_is_the_decided_one(template: str):
    controller, state, runtime, coordinator, speaker, decided, rogue = _setup(template)

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    shown = _last_public(state, speaker)
    ledger = PublicFactLedger(state)
    assert displayed_execution_target(shown, ledger) == decided
    # No sentence declaring the rival survives anywhere in the displayed text.
    assert rogue not in ledger.mentioned_player_ids(shown)


def test_stated_target_is_read_from_the_displayed_text():
    controller, state, runtime, coordinator, speaker, decided, _ = _setup(
        "{rogue}を第一候補にします。"
    )

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    shown = _last_public(state, speaker)
    assert runtime.stated_target(speaker) == decided
    assert runtime.stated_target(speaker) == displayed_execution_target(
        shown, PublicFactLedger(state)
    )


def test_the_ballot_matches_what_was_said():
    controller, state, runtime, coordinator, speaker, decided, rogue = _setup(
        "{rogue}を第一候補にします。"
    )

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    state.phase = Phase.VOTING
    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    assert state.pending_votes[speaker] == decided != rogue
    assert coordinator.validation.vote_plan_mismatches == []


def test_nothing_is_recorded_as_said_when_the_message_is_refused():
    """The stated target used to be recorded before `chat`, so a refused
    message still left the seat 'having said' something nobody saw."""
    controller, state, runtime, coordinator, speaker, _, _ = _setup(
        "{rogue}を第一候補にします。"
    )
    real_chat = controller.chat

    def refusing_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        from app.engine.game import GameError

        raise GameError("speaker is no longer alive")

    controller.chat = refusing_chat  # type: ignore[method-assign]
    try:
        result = asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    finally:
        controller.chat = real_chat  # type: ignore[method-assign]

    assert result is None
    assert runtime.stated_target(speaker) is None


def test_an_unexpected_error_is_not_swallowed():
    controller, state, _, coordinator, speaker, _, _ = _setup("{rogue}を第一候補にします。")

    def exploding_chat(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    controller.chat = exploding_chat  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
