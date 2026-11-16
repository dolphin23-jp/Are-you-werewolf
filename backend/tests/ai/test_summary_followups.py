"""The close after the summary is a turn of its own, and every turn is counted.

The summariser's 本指定 was built as a second summary, whose text invited a
new question that nobody would answer: the round closes on that turn. A note
in the system prompt added that every named seat had been given its turn.
And the progress counter could read 31/29: owed replies and closing turns
were never counted in the total.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.coordinator import _FINAL_DESIGNATION_STAGE, AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.metrics import MetricsCollector
from app.ai.personalities import assign_personalities, chat_stage_max_chars
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DirectedQuestion, DiscussionOutput, MorningIntentOutput
from app.config import Settings
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import GameState
from app.eval.transcript import TranscriptRecorder
from app.main import app
from app.sessions.models import DiscussionRoundState
from app.sessions.store import get_session_store
from scripts import evaluate
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


# -- (3) the progress counter never reads past its total --


def test_answers_and_the_close_after_a_summary_at_the_budget_are_counted_in_the_total():
    provider = _StagedProvider({SUMMARY: _summary_asking("p2", "p3")})
    controller, coordinator, turns = _table(provider)
    session = _human_session(controller)
    # The opening fills the budget but one, so the summary is its last slot.
    session.discussion_round = DiscussionRoundState(
        day=controller.state.day, order=["p1", "p2", "p3"], stage="initial_view", max_total=4
    )

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=10))

    round_state = session.discussion_round
    assert _after_summary(turns) == [
        ("p1", SUMMARY),
        ("p2", "reaction"),
        ("p3", "reaction"),
        ("p1", _FINAL_DESIGNATION_STAGE),
    ]
    assert len(round_state.outputs) == 7
    assert round_state.max_total == 7


def test_an_owed_reply_is_counted_in_the_total():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    coordinator = AICoordinator(state, ["p1", "p2", "p3", "p4"], _StagedProvider(), seed=1)
    black = controller.chat("p1", "占いCO。p2は黒。")
    controller.public_result("p1", "seer", "p2", True, source_message_id=black)
    round_state = DiscussionRoundState(
        day=2,
        order=["p1", "p3"],
        cursor=2,
        stage="initial_view",
        outputs=[
            ("p1", DiscussionOutput(public_message="p2は黒。")),
            ("p3", DiscussionOutput(public_message="見解")),
        ],
        speech_counts={"p1": 1, "p3": 1},
        major_targets_ready=True,
        max_total=2,
    )

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) == ("p2", "rebuttal_or_reassessment")
    assert round_state.max_total == 3


def test_the_view_never_reads_more_spoken_than_total():
    client = TestClient(app)
    session_id = client.post("/api/games", json={"human_name": "Tester", "seed": 5}).json()[
        "session_id"
    ]
    session = get_session_store().get(session_id)
    assert session is not None
    # A round whose total missed a turn somewhere: the counter still reads sanely.
    session.discussion_round = DiscussionRoundState(
        day=session.controller.state.day,
        order=[],
        outputs=[("p1", DiscussionOutput(public_message="見解"))] * 3,
        max_total=2,
    )

    progress = client.get(f"/api/games/{session_id}/view").json()["discussion_progress"]

    assert progress == {"spoken": 3, "total": 3}


def test_seed_3_scenario_game_never_shows_more_spoken_than_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's overflow.py: v3, scenario provider, seed 3 read 31/29 on day 1."""
    seen: list[tuple[int, int, int]] = []
    original = AICoordinator.run_discussion_round

    async def checked(self: AICoordinator, session: Any) -> None:
        await original(self, session)
        round_state = session.discussion_round
        seen.append((round_state.day, len(round_state.outputs), round_state.max_total))

    monkeypatch.setattr(AICoordinator, "run_discussion_round", checked)
    settings = Settings(werewolf_llm_provider="scenario", werewolf_reasoning_engine="v3")

    run = asyncio.run(evaluate.play_game(3, settings, MetricsCollector()))

    assert run.finished
    assert seen and seen[0][0] == 1
    assert all(spoken <= total for _day, spoken, total in seen), seen


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
