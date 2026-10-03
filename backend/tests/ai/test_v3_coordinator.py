"""v3: code keeps the facts, the logic and the obligations; the model decides.

The checks mirror the v2 ones in `test_displayed_target.py` with the sign
flipped. Under v2 a model naming its own candidate was overruled and its
sentence rewritten; under v3 the candidate is the model's, the sentence is left
alone, and the ballot is compared against that candidate instead.
"""

from __future__ import annotations

import asyncio

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.board_memo import MEMO_HEADING
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.eval.transcript import TranscriptRecorder
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class InsistentProvider:
    """Argues for `rogue` everywhere, and keeps every prompt it was given."""

    def __init__(self, rogue: str, *, vote: str | None = None, message: str | None = None) -> None:
        self.rogue = rogue
        self.vote = vote or rogue
        self.message = message or "{name}が怪しいと思う。昨日の票の理由を聞きたい。"
        self.prompts: list[tuple[type, str, str]] = []

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del kwargs
        self.prompts.append((response_schema, system, "\n".join(m.content for m in messages)))
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema is DiscussionOutput:
            output = DiscussionOutput(public_message=self.message)
            output.reasoning_memo.execution_target = self.rogue
            output.reasoning_memo.suspects = [self.rogue]
            return output
        if response_schema is VoteOutput:
            return VoteOutput(vote_target=self.vote, reason="気が変わった。")
        if response_schema is NightActionOutput:
            return NightActionOutput(target=self.rogue)
        if response_schema is WolfChatOutput:
            return WolfChatOutput(message="今夜は様子見でいこう。")
        return SummaryOutput(summary="要約")


def _setup(*, vote: str | None = None):  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    runtime = ReasoningRuntime(state, AI_IDS, seed=4)
    runtime.refresh(state)
    speaker = "p1"
    coded = runtime.seats[speaker].belief.state.current_execution_target
    rogue = next(pid for pid in state.alive_ids() if pid not in (speaker, coded, "p0"))
    provider = InsistentProvider(
        rogue, vote=vote, message=f"{state.players[rogue].name}が怪しいと思う。票の理由を聞きたい。"
    )
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        provider,
        seed=4,
        reasoning=runtime,
        model_decides=True,
        recorder=recorder,
    )
    return controller, state, runtime, coordinator, provider, speaker, coded, rogue


def _last_public(state, speaker: str) -> str:  # type: ignore[no-untyped-def]
    return next(
        m.content
        for m in reversed(state.chat_log)
        if m.author_id == speaker and m.channel.value == "public"
    )


def test_the_models_candidate_stands_and_no_sentence_is_injected():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    shown = _last_public(state, speaker)
    assert shown == provider.message
    assert "現時点の第一処刑候補は" not in shown
    assert runtime.stated_target(speaker) == rogue
    assert coordinator.model_decides is True
    audit = coordinator._recorder.transcript.decision_audits[-1]  # type: ignore[union-attr]
    assert audit.decision_target == rogue
    assert audit.displayed_target == rogue
    assert audit.model_target_was_overridden is False


def test_the_prompt_carries_the_board_memo_and_not_the_brief():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    schema, system, user = next(p for p in provider.prompts if p[0] is DiscussionOutput)
    assert MEMO_HEADING in user
    assert "上の結論・根拠・数値は変更しないでください" not in user
    assert "第一処刑候補" not in user
    assert "【名簿(JSONのplayer_id用)】" in user
    assert "IDは絶対に書かない" in system
    memo = user.split(MEMO_HEADING, 1)[1].split("【名簿", 1)[0]
    assert "(p" not in memo


def test_an_illegal_candidate_is_still_repaired_by_validation():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()
    dead = next(pid for pid in state.alive_ids() if pid not in (speaker, rogue, "p0"))
    state.players[dead].alive = False
    provider.rogue = dead

    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))

    assert runtime.stated_target(speaker) != dead
    assert any(issue.code.startswith("memo_") for issue in coordinator.validation.issues)


def test_the_ballot_is_the_models_and_a_change_of_mind_is_recorded():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    other = next(pid for pid in state.alive_ids() if pid not in (speaker, rogue, "p0"))
    provider.vote = other
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    assert state.pending_votes[speaker] == other
    mismatch = coordinator.validation.vote_plan_mismatches[-1]
    assert (mismatch.stated_target, mismatch.actual_target) == (rogue, other)
    audit = coordinator._recorder.transcript.decision_audits[-1]  # type: ignore[union-attr]
    assert audit.vote_target == other
    assert audit.target_change_classification == "unexplained"
    assert audit.target_change_reason == "気が変わった。"


def test_a_consistent_ballot_records_no_change():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    assert state.pending_votes[speaker] == rogue
    assert coordinator.validation.vote_plan_mismatches == []


def test_the_vote_prompt_tells_the_model_what_it_said():
    controller, state, runtime, coordinator, provider, speaker, coded, rogue = _setup()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    _schema, _system, user = next(p for p in provider.prompts if p[0] is VoteOutput)
    assert f"【昼にあなたが推した吊り先】{state.players[rogue].name}" in user
    assert MEMO_HEADING in user


def _night_game() -> tuple[GameController, str]:
    for seed in range(1, 60):
        controller = make_controller(seed=seed)
        players = controller.state.players.values()
        seer = next(p.player_id for p in players if p.role is RoleName.SEER)
        if seer != "p0":
            controller.start_game()
            return controller, seer
    raise AssertionError("no seed dealt an AI seer")


def test_the_night_target_is_the_models_not_the_belief_engines():
    controller, seer = _night_game()
    state = controller.state
    rogue = next(pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id))
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    provider = InsistentProvider(rogue)
    coordinator = AICoordinator(
        state, AI_IDS, provider, seed=1, reasoning=runtime, model_decides=True
    )

    asyncio.run(coordinator._cast_divine(controller, state, seer))

    assert state.pending_divine is not None
    assert state.pending_divine[1] == rogue
    schema, _system, user = next(p for p in provider.prompts if p[0] is NightActionOutput)
    assert MEMO_HEADING in user


def test_every_wolf_speaks_in_the_wolf_chat_instead_of_one_plan():
    controller, _seer = _night_game()
    state = controller.state
    wolves = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF) if p.player_id != "p0"]
    runtime = ReasoningRuntime(state, AI_IDS, seed=1)
    provider = InsistentProvider(wolves[0])
    coordinator = AICoordinator(
        state, AI_IDS, provider, seed=1, reasoning=runtime, model_decides=True
    )

    asyncio.run(coordinator._run_wolf_chat_round(controller, state))

    wolf_lines = [m for m in state.chat_log if m.channel.value == "wolf"]
    assert len(wolf_lines) == len(wolves)
    assert all(m.content == "今夜は様子見でいこう。" for m in wolf_lines)
    assert not any(m.content.startswith("了解") for m in wolf_lines)


def test_model_decides_without_a_runtime_is_plain_legacy():
    state = make_controller(seed=4).state
    coordinator = AICoordinator(state, AI_IDS, InsistentProvider("p2"), seed=4, model_decides=True)

    assert coordinator.model_decides is False
    assert coordinator.reasoning is None


def test_v2_still_overrules_the_model():
    """The flip is opt-in: the same provider under v2 keeps v2's guarantees."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    runtime = ReasoningRuntime(state, AI_IDS, seed=4)
    runtime.refresh(state)
    coded = runtime.seats["p1"].belief.state.current_execution_target
    rogue = next(pid for pid in state.alive_ids() if pid not in ("p1", coded, "p0"))
    coordinator = AICoordinator(
        state, AI_IDS, InsistentProvider(rogue), seed=4, reasoning=runtime, model_decides=False
    )

    asyncio.run(coordinator._speak(controller, state, "p1", "initial_view"))

    assert runtime.stated_target("p1") == coded
    assert "現時点の第一処刑候補は" in _last_public(state, "p1")
    assert PublicFactLedger(state).mentioned_player_ids(_last_public(state, "p1"))
