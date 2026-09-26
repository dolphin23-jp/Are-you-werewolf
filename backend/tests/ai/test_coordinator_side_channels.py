"""What one AI hands the others must be what the table actually heard."""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator, _key_point_as_said, _question_as_said
from app.ai.provider.mock import MockProvider
from app.ai.schemas import DirectedQuestion, DiscussionOutput
from app.engine.roles import RoleName
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


def test_a_question_is_registered_only_as_it_was_asked_in_public():
    state = make_controller(seed=1).state
    target = state.players["p5"]
    message = f"{target.name}(p5)さん、昨日の投票理由は何ですか？ 私は灰を見ます。"

    asked = DirectedQuestion(target_id="p5", question="昨日の投票理由は何ですか？")
    hidden = DirectedQuestion(target_id="p5", question="仲間のp8を守るため票をずらせる？")
    unasked = DirectedQuestion(target_id="p9", question="あなたは狼？")

    assert _question_as_said(state, message, asked) == "昨日の投票理由は何ですか？"
    # Not in the message: the posted sentence that asks p5 stands in for it.
    stand_in = f"{target.name}(p5)さん、昨日の投票理由は何ですか？"
    assert _question_as_said(state, message, hidden) == stand_in
    # Nobody asked p9 anything in public.
    assert _question_as_said(state, message, unasked) == ""


def test_a_key_point_never_carries_what_the_message_did_not_say():
    message = "p3の投票先が昨日から変わっています。理由を聞きたいです。"

    assert _key_point_as_said(message, "p3の投票先が昨日から変わっています") == (
        "p3の投票先が昨日から変わっています"
    )
    assert _key_point_as_said(message, "仲間p8を守るためp16へ票を寄せる") == (
        "p3の投票先が昨日から変わっています。"
    )


class _ClaimsThroughTheStructuredField:
    async def generate_structured(  # type: ignore[no-untyped-def]
        self, *, system, messages, response_schema, **kwargs
    ):
        del system, messages, kwargs
        if response_schema is DiscussionOutput:
            return DiscussionOutput(
                public_message="灰の発言を比較していきます。", public_claim_role="freemason"
            )
        return None


def test_the_hidden_freemason_cannot_claim_through_the_structured_field():
    for seed in range(1, 60):
        controller = make_controller(seed=seed)
        coordinator = AICoordinator(
            controller.state, AI_IDS, _ClaimsThroughTheStructuredField(), seed=seed,
            pacing_scale=0.0,
        )
        plan = coordinator._freemason_public_plan
        state = controller.state
        controller.start_game()
        controller.resolve_night()
        if plan is None or not all(state.players[pid].alive for pid in plan[:2]):
            continue
        controller.start_discussion()
        partner = plan[1]
        if not coordinator._freemason_must_hide(state, partner):
            continue
        asyncio.run(coordinator._speak(controller, state, partner, "initial_view"))
        assert all(claim.player_id != partner for claim in state.co_declarations)
        assert "共有" not in state.chat_log[-1].content
        return
    raise AssertionError("no seed produced a hidden freemason partner")


def test_a_human_wolf_is_not_given_a_part_in_the_ai_wolves_plan():
    for seed in range(1, 200):
        controller = make_controller(seed=seed)
        if controller.state.players["p0"].role is not RoleName.WEREWOLF:
            continue
        coordinator = AICoordinator(
            controller.state, AI_IDS, MockProvider(seed=seed), seed=seed, pacing_scale=0.0
        )
        plan = coordinator._wolf_deception
        assert "p0" not in plan.fake_role_by_player
        assert "p0" not in plan.lurking_player_ids


def test_a_refused_message_is_not_recorded_as_said():
    """The memo and the transcript line used to be written before the engine
    accepted the message, so a turn nobody heard showed up as spoken."""
    from app.engine.game import GameError
    from app.eval.transcript import TranscriptRecorder

    controller = make_controller(seed=3)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        controller.state, AI_IDS, MockProvider(seed=3), seed=3, recorder=recorder,
        pacing_scale=0.0,
    )
    state = controller.state
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    speaker = next(pid for pid in AI_IDS if state.players[pid].alive)

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise GameError("the speaker died mid-round")

    controller.chat = refuse  # type: ignore[method-assign]
    result = asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    assert result is None
    assert not [u for u in recorder.transcript.utterances if u.player_id == speaker]
    assert coordinator._context.get_reasoning_memo(speaker) in (None, {})
