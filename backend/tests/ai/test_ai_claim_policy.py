"""What an AI seat may put on the board as a CO.

A declared field is the model's decision; a claim read only from its prose is
a matcher's inference and registers only for the seat's own role or planned
fake. A freemason claim from anyone but a freemason is removed outright: the
engine allows it (a human may), but for an AI it has almost no upside.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.reasoning.claims import SpeechEventDraft
from app.ai.schemas import DiscussionOutput
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class _Says:
    def __init__(self, message: str, claim: str | None = None) -> None:
        self.message = message
        self.claim = claim

    async def generate_structured(  # type: ignore[no-untyped-def]
        self, *, system, messages, response_schema, **kwargs
    ):
        del system, messages, kwargs
        if response_schema is DiscussionOutput:
            return DiscussionOutput(public_message=self.message, public_claim_role=self.claim)
        return None


def _speak_as(role: RoleName, provider: _Says):  # type: ignore[no-untyped-def]
    """Speak once as a living AI seat of `role` that is free to claim."""
    for seed in range(1, 80):
        controller = make_controller(seed=seed)
        coordinator = AICoordinator(
            controller.state, AI_IDS, provider, seed=seed, pacing_scale=0.0
        )
        state = controller.state
        controller.start_game()
        controller.resolve_night()
        controller.start_discussion()
        speaker = next(
            (
                pid
                for pid in AI_IDS
                if state.players[pid].role is role
                and state.players[pid].alive
                and not coordinator._freemason_must_hide(state, pid)
                and coordinator._freemason_opening(state, pid) is None
            ),
            None,
        )
        if speaker is None:
            continue
        asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
        return coordinator, state, speaker
    raise AssertionError(f"no seed gave a free AI {role.value}")


def _claims_of(state, speaker):  # type: ignore[no-untyped-def]
    return [c.claimed_role for c in state.co_declarations if c.player_id == speaker]


def test_a_prose_only_claim_of_the_seats_own_role_registers():
    """The model wrote the CO but left the field empty: still its claim."""
    _, state, speaker = _speak_as(RoleName.SEER, _Says("占い師COです。灰を見ていきます。"))
    assert _claims_of(state, speaker) == [RoleName.SEER]


def test_a_prose_only_claim_of_another_role_does_not_register():
    """Only a matcher read this as a claim; the seat never meant to make it."""
    coordinator, state, speaker = _speak_as(
        RoleName.VILLAGER, _Says("霊媒CO、まだ結果はありません。")
    )
    assert _claims_of(state, speaker) == []
    assert "unintended_prose_claim_dropped" in coordinator.validation.codes()


def test_a_declared_claim_of_another_role_stands():
    """A structured field is a decision (a bluff is allowed), not an inference."""
    _, state, speaker = _speak_as(RoleName.VILLAGER, _Says("占い師です。", claim="seer"))
    assert _claims_of(state, speaker) == [RoleName.SEER]


def test_a_non_freemason_ai_cannot_claim_freemason_in_either_channel():
    coordinator, state, speaker = _speak_as(
        RoleName.WEREWOLF,
        _Says("共有者COします。相方はユイです。灰を比較します。", claim="freemason"),
    )
    assert _claims_of(state, speaker) == []
    assert not state.freemason_partner_claims
    said = state.chat_log[-1].content
    assert "共有" not in said and "相方" not in said
    assert "灰を比較します" in said
    assert "fake_freemason_claim_removed" in coordinator.validation.codes()


def test_a_real_freemason_still_claims():
    _, state, speaker = _speak_as(RoleName.FREEMASON, _Says("共有者COします。", claim="freemason"))
    assert _claims_of(state, speaker) == [RoleName.FREEMASON]


def test_a_planned_fake_role_counts_as_intended_even_from_prose():
    controller = make_controller(seed=5)
    coordinator = AICoordinator(controller.state, AI_IDS, _Says(""), seed=5, pacing_scale=0.0)
    state = controller.state
    wolf = next(pid for pid in AI_IDS if state.players[pid].role is RoleName.WEREWOLF)
    coordinator._wolf_deception.fake_role_by_player[wolf] = RoleName.SEER
    prose_claim = SpeechEventDraft(
        event_type=SpeechEventType.ROLE_CLAIM, role=RoleName.SEER, confidence=0.9
    )
    prose_other = SpeechEventDraft(
        event_type=SpeechEventType.ROLE_CLAIM, role=RoleName.MEDIUM, confidence=0.9
    )

    assert coordinator._ai_claim_policy(state, wolf, [prose_claim]) == [prose_claim]
    assert coordinator._ai_claim_policy(state, wolf, [prose_other]) == []
