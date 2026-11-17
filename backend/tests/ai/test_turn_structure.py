"""Who gets the floor before the vote: nobody is hanged for a silence code caused.

The first Claude game ended five of six days on the freemason leader's closing
summary, each time with a question the code never let anyone answer -- the
round closed on that summary before its questions were queued. A villager
blacked mid-morning on day 4 never spoke again and was hanged; the pressured
candidates had been ranked once, before the black existed. On day 1 seven
gray seats never got a turn and the votes went to "the one who stayed quiet".
These tests pin the turn structure that makes each of those impossible.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.ai.coordinator import _FINAL_DESIGNATION_STAGE, AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    ChatDiscussionOutput,
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
)
from app.engine.phases import Phase
from app.engine.state import ChatChannel, GameState
from app.sessions.models import DiscussionRoundState
from tests.ai import first_claude_game as game1
from tests.conftest import make_controller


class _StagedProvider:
    """Answers each turn by the stage the scheduler gave it.

    The spy below sets `stage` before every turn, so the script does not
    depend on how any prompt words its stage. `script` maps a stage to the
    turn to return; anything else is a plain line.
    """

    def __init__(self, script: dict[str, DiscussionOutput] | None = None) -> None:
        self.script = script or {}
        self.stage = ""
        self.speaker = ""
        # Seats whose every turn fails once the summary has been given.
        self.silent_after_summary: set[str] = set()
        self.summary_given = False

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if self.summary_given and self.speaker in self.silent_after_summary:
            return None
        output = self.script.get(self.stage, DiscussionOutput(public_message="見解を述べます。"))
        output = output.model_copy(deep=True)
        if response_schema is ChatDiscussionOutput:
            return ChatDiscussionOutput.from_discussion_output(output)
        return output


def _spy(coordinator: AICoordinator, provider: _StagedProvider) -> list[tuple[str, str]]:
    """Record every turn the scheduler hands out, in order, as (seat, stage)."""
    turns: list[tuple[str, str]] = []
    original = coordinator._speak

    async def speak(controller: Any, state: GameState, player_id: str, stage: str) -> Any:
        turns.append((player_id, stage))
        provider.stage = stage
        provider.speaker = player_id
        try:
            return await original(controller, state, player_id, stage)
        finally:
            provider.summary_given = provider.summary_given or stage == "consensus_summary"

    coordinator._speak = speak  # type: ignore[method-assign]
    return turns


def _session(controller: Any) -> SimpleNamespace:
    return SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())


def _summary_asking(*targets: str) -> DiscussionOutput:
    # The line names whom it asks: a structured question the line does not
    # put aloud books no answer (`AICoordinator._questions_said_aloud`).
    return DiscussionOutput(
        public_message=f"整理します。仮指定です。{'、'.join(targets)}、最後に反論はありますか？",
        directed_questions=[
            DirectedQuestion(target_id=target, question="最後に反論はありますか？")
            for target in targets
        ],
        ready_to_vote=True,
    )


# -- (a) the closing summary does not close the round on its own questions --


def _table(provider: _StagedProvider) -> tuple[Any, AICoordinator, list[tuple[str, str]]]:
    """Three AI seats and a human; p1 is the leader the summary falls to."""
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    # The summary goes to a living freemason claimant: p1, whatever it holds.
    controller.co("p1", "freemason")
    coordinator = AICoordinator(controller.state, ["p1", "p2", "p3"], provider, seed=1)
    return controller, coordinator, _spy(coordinator, provider)


def test_a_summary_question_is_answered_before_a_short_designation_closes_the_day():
    provider = _StagedProvider({"consensus_summary": _summary_asking("p2")})
    controller, coordinator, turns = _table(provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    summary = [stage for _seat, stage in turns].index("consensus_summary")
    # p2 is nobody's candidate here. Its answer used to be the 30-character
    # reaction; an answer to the summary is always a 60-character turn now
    # (seed-1 review, 1-A3).
    assert turns[summary:] == [
        ("p1", "consensus_summary"),
        ("p2", "summary_answer"),
        ("p1", _FINAL_DESIGNATION_STAGE),
    ]
    assert controller.state.chat_log[-1].author_id == "p1"
    # The question was put to p2 on its turn, so nothing is left unserved.
    assert controller.state.pending_questions["p2"]
    assert all(q.served for q in controller.state.pending_questions["p2"])


def test_the_pressured_candidate_answers_the_summary_with_a_rebuttal():
    candidate = _output("p2", "p2が怪しい。")
    provider = _StagedProvider(
        {"consensus_summary": _summary_asking("p2"), "initial_view": candidate}
    )
    controller, coordinator, turns = _table(provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    summary = [stage for _seat, stage in turns].index("consensus_summary")
    assert turns[summary + 1] == ("p2", "rebuttal_or_reassessment")
    assert turns[-1] == ("p1", _FINAL_DESIGNATION_STAGE)


def test_every_named_seat_answers_once_and_the_human_is_never_scheduled():
    provider = _StagedProvider({"consensus_summary": _summary_asking("p2", "p0", "p3", "p2")})
    controller, coordinator, turns = _table(provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    summary = [stage for _seat, stage in turns].index("consensus_summary")
    assert [seat for seat, _stage in turns[summary + 1 :]] == ["p2", "p3", "p1"]
    assert turns[-1] == ("p1", _FINAL_DESIGNATION_STAGE)


def test_the_designation_is_told_to_decide_without_asking_again():
    # The designation is a stage of its own: its scene line says what the turn
    # is for, once, in the user message -- no second note in the system prompt.
    requests: list[tuple[str, str, str]] = []
    provider = _StagedProvider({"consensus_summary": _summary_asking("p2")})
    original = provider.generate_structured

    async def capture(**kwargs: Any) -> Any:
        if kwargs["response_schema"] is not MorningIntentOutput:
            user = "\n".join(message.content for message in kwargs["messages"])
            requests.append((provider.stage, kwargs["system"], user))
        return await original(**kwargs)

    provider.generate_structured = capture  # type: ignore[method-assign]
    controller, coordinator, _turns = _table(provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    designation = [
        (system, user) for stage, system, user in requests if stage == _FINAL_DESIGNATION_STAGE
    ]
    assert len(designation) == 1
    system, user = designation[0]
    assert "【議論段階】本指定。" in user
    assert "新しい質問" in user
    assert "【本指定】" not in system
    others = [user for stage, _system, user in requests if stage != _FINAL_DESIGNATION_STAGE]
    assert not any("【議論段階】本指定。" in user for user in others)


def test_a_summary_naming_nobody_goes_on_to_a_provisional_designation():
    # This pinned that a summary with no question closed the round at once.
    # Seed 1's day 2 closed that way on 「皆さんは気になるグレーの方はいますか…？」,
    # a question to nobody by name, and the vote began with no candidate.
    # Now the facilitator gives a provisional designation; it names nobody
    # here either, so that closes the day (once a day, no designation).
    provider = _StagedProvider()
    controller, coordinator, turns = _table(provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    assert turns[-2:] == [("p1", "consensus_summary"), ("p1", "provisional_designation")]
    assert _FINAL_DESIGNATION_STAGE not in [stage for _seat, stage in turns]


def test_the_close_terminates_when_the_answering_seat_cannot_speak():
    provider = _StagedProvider({"consensus_summary": _summary_asking("p2")})
    provider.silent_after_summary = {"p2"}
    controller, coordinator, turns = _table(provider)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=10))

    summary = [stage for _seat, stage in turns].index("consensus_summary")
    assert turns[summary + 1 :] == [("p2", "summary_answer"), ("p1", _FINAL_DESIGNATION_STAGE)]
    assert session.discussion_round.complete is True


# -- (b) a black that lands after the ranking still gets its answer --


def _output(target: str | None, text: str = "見解") -> DiscussionOutput:
    output = DiscussionOutput(public_message=text)
    output.reasoning_memo.execution_target = target
    return output


def test_first_game_day_4_tsumugi_blacked_mid_morning_answers_before_the_summary():
    """Day 4: ツムギ spoke once, then アオイ (madman) published a black on him.

    The pressured list had already been ranked from the plan notes -- only
    ツムギ's own lean on カイト -- so カイト was queued twice and the summary
    closed the day. ツムギ's rebuttal ended up in a vote reason nobody saw.
    """
    ids = game1.name_to_id()
    tsumugi, kaito, aoi = ids["ツムギ"], ids["カイト"], ids["アオイ"]
    state = game1.state_at_vote(4)
    state.phase = Phase.DISCUSSION
    # The day as it stood right after アオイ's black (m61).
    state.chat_log = [
        m for m in state.chat_log if not (m.day == 4 and int(m.message_id[1:]) > 61)
    ]
    state.speech_events = [
        e for e in state.speech_events if not (e.day == 4 and int(e.source_message_id[1:]) > 61)
    ]
    day4 = [m for m in state.chat_log if m.day == 4 and m.channel is ChatChannel.PUBLIC]
    assert [m.author_id for m in day4][-2:] == [ids["マコト"], aoi]
    # One discussion utterance per public line, in the same order.
    turns = [
        utterance
        for utterance in game1.load_transcript()["utterances"]
        if utterance["day"] == 4 and utterance["kind"] == "discussion"
    ][: len(day4)]
    assert [turn["player_id"] for turn in turns] == [m.author_id for m in day4]
    outputs = [
        (turn["player_id"], _output((turn["reasoning_memo"] or {}).get("execution_target")))
        for turn in turns
    ]
    opening = [m.author_id for m in day4[:5]]
    coordinator = AICoordinator(state, game1.AI_IDS, MockProvider(seed=1), seed=1)
    round_state = DiscussionRoundState(
        day=4,
        order=opening,
        cursor=len(opening),
        stage="initial_view",
        immediate_count=1,
        outputs=outputs,
        speech_counts={seat: 1 for seat, _output in outputs},
        # Ranked before the black, from the plan notes then on the table.
        major_targets=[kaito],
        major_targets_ready=True,
        reply_queue=[kaito],
        queued={kaito},
        max_total=20,
    )

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) == (tsumugi, "rebuttal_or_reassessment")
    assert tsumugi in round_state.major_targets
    # One turn per accusation: the next pick is the queue, not ツムギ again.
    round_state.speech_counts[tsumugi] = 2
    assert coordinator._next_discussion_speaker(state, round_state)[0] == kaito


def test_a_black_on_a_seat_that_already_answered_it_owes_nothing():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    # Only the two seats that have spoken: from day 2 a seat with no line yet
    # gets a short turn before the summary, which is not what this pins.
    coordinator = AICoordinator(state, ["p1", "p2"], MockProvider(seed=1), seed=1)
    black = controller.chat("p1", "占いCO。p2は黒。")
    controller.public_result("p1", "seer", "p2", True, source_message_id=black)
    controller.chat("p2", "その黒は偽。対抗で占いCOする。")
    round_state = DiscussionRoundState(
        day=2,
        order=["p1", "p2"],
        cursor=2,
        outputs=[("p1", _output("p2")), ("p2", _output("p1"))],
        speech_counts={"p1": 1, "p2": 1},
        major_targets_ready=True,
        max_total=10,
    )

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert stage == "consensus_summary"
    assert "p2" not in round_state.major_targets


def test_a_new_co_reranks_the_pressured_candidates():
    """A counter-CO moved the table onto p5 after the ranking was frozen on p3/p6."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    ai_ids = [f"p{i}" for i in range(1, 9)]
    coordinator = AICoordinator(state, ai_ids, MockProvider(seed=1), seed=1)
    round_state = DiscussionRoundState(
        day=2,
        order=["p1", "p2", "p4"],
        cursor=3,
        outputs=[("p1", _output("p3")), ("p2", _output("p3")), ("p4", _output("p6"))],
        speech_counts={"p1": 1, "p2": 1, "p4": 1},
        max_total=30,
    )
    first, _stage = coordinator._next_discussion_speaker(state, round_state)
    assert round_state.major_targets == ["p3", "p6"]
    round_state.speech_counts[first] = round_state.speech_counts.get(first, 0) + 1
    # p4 moves to p5, then p7 counter-claims seer and also names p5.
    round_state.outputs.append(("p4", _output("p5")))
    controller.co("p7", "seer")
    round_state.outputs.append(("p7", _output("p5")))

    seen = []
    for _ in range(6):
        speaker, _stage = coordinator._next_discussion_speaker(state, round_state)
        if speaker is None:
            break
        seen.append(speaker)
        round_state.speech_counts[speaker] = round_state.speech_counts.get(speaker, 0) + 1

    assert "p5" in round_state.major_targets
    assert "p5" in seen


# -- (d) on day 1 every seat speaks before the vote --


def test_day_one_gives_every_silent_ai_seat_a_first_impression_before_the_close():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 1
    ai_ids = [f"p{i}" for i in range(1, 17)]
    provider = _StagedProvider()
    coordinator = AICoordinator(
        state,
        ai_ids,
        provider,
        seed=4,
        reasoning=ReasoningRuntime(state, ai_ids, seed=4),
        model_decides=True,
    )
    turns = _spy(coordinator, provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    spoken = {m.author_id for m in state.chat_log if m.channel is ChatChannel.PUBLIC}
    assert spoken >= set(ai_ids)
    # Each roll-call seat's first turn is a first impression, not a reaction.
    summary = [stage for _seat, stage in turns].index("consensus_summary")
    first_turns = {}
    for seat, stage in turns[:summary]:
        first_turns.setdefault(seat, stage)
    assert set(first_turns.values()) <= {"immediate", "initial_view", "freemason_confirmation"}


def test_later_days_open_short_and_hear_every_silent_seat_before_the_close():
    # This pinned that later days leave most seats silent. The owner's review
    # of seed 1 reversed that: 37 of 83 ballots on days 2-5 came from seats
    # with no turn that day. The opening stays short -- the duty speakers and
    # three seats -- and every seat still silent gets one short turn before
    # the summary (tests/ai/test_day_turn_schedule.py pins the details).
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    ai_ids = [f"p{i}" for i in range(1, 17)]
    provider = _StagedProvider()
    coordinator = AICoordinator(
        state,
        ai_ids,
        provider,
        seed=4,
        reasoning=ReasoningRuntime(state, ai_ids, seed=4),
        model_decides=True,
    )
    turns = _spy(coordinator, provider)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    opening = [stage for _seat, stage in turns].index("open_floor")
    assert opening <= 3 + sum(stage == "immediate" for _seat, stage in turns)
    spoken = {m.author_id for m in state.chat_log if m.channel is ChatChannel.PUBLIC}
    assert spoken == set(ai_ids)
