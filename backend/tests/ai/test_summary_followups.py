"""The close after the summary is a turn of its own, told once and truthfully.

The summariser's 本指定 was built as a second summary, whose text invited a
new question that nobody would answer: the round closes on that turn. A note
in the system prompt added that every named seat had been given its turn.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.coordinator import _FINAL_DESIGNATION_STAGE, AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.metrics import MetricsCollector
from app.ai.personalities import assign_personalities, chat_stage_max_chars
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DirectedQuestion, DiscussionOutput, MorningIntentOutput
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import GameState
from app.eval.transcript import TranscriptRecorder
from tests.ai.test_turn_structure import _spy, _StagedProvider, _summary_asking
from tests.conftest import make_controller

SUMMARY = "consensus_summary"


def _human_session(controller: Any, human_id: str = "p0") -> SimpleNamespace:
    """The attributes the API's session carries, with a living human at `human_id`."""
    return SimpleNamespace(
        controller=controller,
        discussion_lock=asyncio.Lock(),
        human_id=human_id,
        discussion_paused=False,
        discussion_pause_requested=False,
        discussion_step_budget=None,
    )


def _table(
    provider: _StagedProvider, ai_ids: tuple[str, ...] = ("p1", "p2", "p3")
) -> tuple[Any, AICoordinator, list[tuple[str, str]]]:
    """AI seats around a living human p0; p1 claims freemason and gives the summary."""
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.co("p1", "freemason")
    coordinator = AICoordinator(controller.state, list(ai_ids), provider, seed=1)
    return controller, coordinator, _spy(coordinator, provider)


async def _press(coordinator: AICoordinator, session: SimpleNamespace, mode: str) -> None:
    """One press of the human's control, the way /pass-turn and /discussion-control do it."""
    if mode == "step":
        session.discussion_paused = False
        session.discussion_pause_requested = False
        session.discussion_step_budget = 1
    coordinator.resume_after_human(session)
    await coordinator.advance_discussion(session)


async def _drive(coordinator: AICoordinator, session: SimpleNamespace, mode: str) -> None:
    for _ in range(60):
        round_state = getattr(session, "discussion_round", None)
        if round_state is not None and round_state.complete:
            return
        await _press(coordinator, session, mode)
    raise AssertionError("the round never closed")


def _after_summary(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return turns[[stage for _seat, stage in turns].index(SUMMARY) :]


# -- (4) the designation is a stage of its own, told once and truthfully --


def _builder(state: GameState, engine: str) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine=engine,
    )


def _scene(content: str, marker: str) -> str:
    return next(line for line in content.splitlines() if line.startswith(marker))


def test_the_chat_designation_names_one_candidate_without_a_question_under_its_own_cap():
    state = make_controller(seed=4).state
    state.day = 2

    _system, messages = _builder(state, "v3").build_discussion_context(
        state, "p1", _FINAL_DESIGNATION_STAGE, board_memo="【盤面メモ】\n- なし"
    )
    scene = _scene(messages[0].content, "【いまの場面】")

    assert scene.startswith("【いまの場面】本指定。")
    assert chat_stage_max_chars(_FINAL_DESIGNATION_STAGE) == 60
    assert "60字以内" in scene
    assert "質問はしない" in scene
    assert "聞いてよい" not in scene
    # Truthful whoever answered: it does not claim every named seat had a turn.
    assert "番を回した" not in scene


def test_the_structured_designation_has_its_own_stage_text():
    state = make_controller(seed=4).state
    state.day = 2

    _system, messages = _builder(state, "legacy").build_discussion_context(
        state, "p1", _FINAL_DESIGNATION_STAGE
    )
    scene = _scene(messages[0].content, "【議論段階】")

    assert scene.startswith("【議論段階】本指定。")
    assert "新しい質問" in scene
    assert "第二候補" not in scene


def _capture(provider: _StagedProvider) -> list[tuple[str, str, str]]:
    """Every discussion request as (stage, system, user text)."""
    requests: list[tuple[str, str, str]] = []
    original = provider.generate_structured

    async def capture(**kwargs: Any) -> Any:
        if kwargs["response_schema"] is not MorningIntentOutput:
            user = "\n".join(message.content for message in kwargs["messages"])
            requests.append((provider.stage, kwargs["system"], user))
        return await original(**kwargs)

    provider.generate_structured = capture  # type: ignore[method-assign]
    return requests


def _v3_table(provider: _StagedProvider) -> tuple[Any, AICoordinator, list[tuple[str, str]]]:
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    controller.co("p1", "freemason")
    ai_ids = [f"p{i}" for i in range(1, 17)]
    coordinator = AICoordinator(
        state,
        ai_ids,
        provider,
        seed=4,
        reasoning=ReasoningRuntime(state, ai_ids, seed=4),
        model_decides=True,
        recorder=TranscriptRecorder(),
    )
    return controller, coordinator, _spy(coordinator, provider)


def test_the_v3_designation_turn_is_told_once_by_its_own_scene_and_capped_at_60():
    provider = _StagedProvider({SUMMARY: _summary_asking("p2")})
    requests = _capture(provider)
    controller, coordinator, turns = _v3_table(provider)

    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    assert turns[-1] == ("p1", _FINAL_DESIGNATION_STAGE)
    designation = [
        (system, user) for stage, system, user in requests if stage == _FINAL_DESIGNATION_STAGE
    ]
    assert len(designation) == 1
    system, user = designation[0]
    assert "【本指定】" not in system
    assert user.count("【いまの場面】") == 1
    assert "【いまの場面】本指定。" in user
    assert "聞いてよい" not in user
    # Nothing claims every seat the summary named was given a turn.
    assert "番を回した" not in system + user
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    closing = [u for u in utterances if u.stage == _FINAL_DESIGNATION_STAGE]
    assert [u.effective_length_limit for u in closing] == [60]


def test_a_designation_that_asks_anyway_still_closes_the_round_and_is_counted():
    question = DiscussionOutput(
        public_message="本指定はp3。p3、最後に何かある？",
        directed_questions=[DirectedQuestion(target_id="p3", question="最後に何かある？")],
    )
    provider = _StagedProvider({SUMMARY: _summary_asking("p2"), _FINAL_DESIGNATION_STAGE: question})
    metrics = MetricsCollector()
    provider._metrics = metrics  # type: ignore[attr-defined]
    controller, coordinator, turns = _table(provider)
    session = _human_session(controller)

    asyncio.run(asyncio.wait_for(_drive(coordinator, session, "pass"), timeout=10))

    assert _after_summary(turns) == [
        ("p1", SUMMARY),
        ("p2", "reaction"),
        ("p1", _FINAL_DESIGNATION_STAGE),
    ]
    assert session.discussion_round.complete is True
    assert metrics.events.get("designation_question_unanswered") == 1
