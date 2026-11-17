"""Who gets the floor during the day, after the owner's review of seed 1 (2026-11-17).

Each section pins one fix from `seed1-review-causes.md` §2.2, and every test
drives the round the way a game does -- `run_discussion_round` or
`advance_discussion` -- not only the scheduler's internal pick: the partner
confirmation's own unit test passed for months while the loop around it
handed the confirmation to the reply queue, so the forced turn never ran.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    ChatDiscussionOutput,
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
)
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.state import ChatChannel, GameState
from app.eval.transcript import TranscriptRecorder
from scripts import evaluate
from tests.conftest import make_controller

SUMMARY = "consensus_summary"
CONFIRMATION = "freemason_confirmation"


class _Table:
    """A staged provider and a spy: every turn as (seat, stage), every model call by seat.

    `script` answers a stage; `by_seat` answers one seat's turns at a stage,
    as (seat, stage) -> output, and wins over `script`. Anything else is a
    plain line with no candidate and no question.
    """

    def __init__(
        self,
        script: dict[str, DiscussionOutput] | None = None,
        by_seat: dict[tuple[str, str], DiscussionOutput] | None = None,
    ) -> None:
        self.script = script or {}
        self.by_seat = by_seat or {}
        self.stage = ""
        self.speaker = ""
        self.calls: list[tuple[str, str]] = []
        self.turns: list[tuple[str, str]] = []

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        self.calls.append((self.speaker, self.stage))
        output = self.by_seat.get(
            (self.speaker, self.stage),
            self.script.get(self.stage, DiscussionOutput(public_message="見解を述べます。")),
        ).model_copy(deep=True)
        if response_schema is ChatDiscussionOutput:
            return ChatDiscussionOutput.from_discussion_output(output)
        return output

    def watch(self, coordinator: AICoordinator) -> None:
        original = coordinator._speak

        async def speak(controller: Any, state: GameState, player_id: str, stage: str) -> Any:
            self.turns.append((player_id, stage))
            self.stage = stage
            self.speaker = player_id
            return await original(controller, state, player_id, stage)

        coordinator._speak = speak  # type: ignore[method-assign]


def _session(controller: Any) -> SimpleNamespace:
    return SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())


def _all_ai_morning(seed: int = 1) -> GameController:
    """The evaluation harness's all-AI table, at the start of day 1."""
    controller = GameController(
        session_id=f"schedule-{seed}", player_specs=evaluate._make_specs(), seed=seed
    )
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    return controller


def _v3(controller: GameController, table: _Table, seed: int = 1) -> AICoordinator:
    state = controller.state
    ids = list(state.players)
    coordinator = AICoordinator(
        state,
        ids,
        table,
        seed=seed,
        reasoning=ReasoningRuntime(state, ids, seed=seed),
        model_decides=True,
        recorder=TranscriptRecorder(),
    )
    table.watch(coordinator)
    return coordinator


def _public(state: GameState, day: int | None = None) -> list[Any]:
    return [
        m
        for m in state.chat_log
        if m.channel is ChatChannel.PUBLIC and (day is None or m.day == day)
    ]


def _line(text: str, target: str | None = None) -> DiscussionOutput:
    output = DiscussionOutput(public_message=text)
    output.reasoning_memo.execution_target = target
    return output


# -- 1-A1: the named partner confirms where the table expects it --


def test_a_partner_named_in_the_morning_confirms_at_the_end_of_the_morning_block():
    """Seed 1, day 1: マコト (p0) named ホノカ (p13) at m2 and she confirmed at m16.

    The loop moved the forced confirmation into the reply queue, behind a
    fifteen-seat roll call. Now she closes the morning block: after the duty
    speakers, before anyone's first view, replying to the line that named her.
    """
    controller = _all_ai_morning(seed=1)
    table = _Table()
    coordinator = _v3(controller, table)
    assert coordinator._freemason_public_plan == ("p0", "p13", True)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    state = controller.state
    lines = _public(state, day=1)
    named = next(m for m in lines if m.author_id == "p0")
    assert "ホノカ" in named.content
    morning = [seat for seat, stage in table.turns if stage == "immediate"]
    position = table.turns.index(("p13", CONFIRMATION))
    # Right behind the morning's duty speakers, before the first view.
    assert position == len(morning)
    assert all(stage == "immediate" for _seat, stage in table.turns[:position])
    confirmation = lines[position]
    assert confirmation.author_id == "p13"
    assert confirmation.reply_to == named.message_id
    assert state.freemason_partner_claims[0].confirmed is True
    # One turn for her before the close: the roll call does not call her again.
    summary = [stage for _seat, stage in table.turns].index(SUMMARY)
    assert [seat for seat, _stage in table.turns[:summary]].count("p13") == 1


def test_lines_code_writes_whole_cost_no_model_call():
    """The freemason CO and the partner confirmation are fixed sentences.

    The model's text for them was thrown away, so asking for it was one
    wasted request each.
    """
    controller = _all_ai_morning(seed=1)
    table = _Table()
    coordinator = _v3(controller, table)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    assert ("p0", "immediate") in table.turns
    assert ("p13", CONFIRMATION) in table.turns
    assert ("p0", "immediate") not in table.calls
    assert ("p13", CONFIRMATION) not in table.calls
    assert len(table.calls) == len(table.turns) - 2
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    coded_turns = {("p0", "immediate"), ("p13", CONFIRMATION)}
    coded = [u for u in utterances if (u.player_id, u.stage) in coded_turns]
    assert [u.used_fallback for u in coded] == [False, False]
    assert [u.model_text_chars for u in coded] == [0, 0]


def _small_table(table: _Table) -> tuple[GameController, AICoordinator]:
    """Three AI seats around a living human p0; p1 claims freemason and leads."""
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.co("p1", "freemason")
    coordinator = AICoordinator(controller.state, ["p1", "p2", "p3"], table, seed=1)
    table.watch(coordinator)
    return controller, coordinator


def test_a_partner_named_after_the_morning_takes_the_very_next_turn():
    naming = _line("共有者CO。相方はPlayer3(p3)です。")
    table = _Table(by_seat={("p2", "initial_view"): naming})
    controller, coordinator = _small_table(table)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    position = table.turns.index(("p2", "initial_view"))
    assert table.turns[position + 1] == ("p3", CONFIRMATION)
    lines = _public(controller.state)
    assert lines[position + 1].reply_to == lines[position].message_id
    # Its place later in the opening was given up for the confirmation.
    summary = [stage for _seat, stage in table.turns].index(SUMMARY)
    assert [seat for seat, _stage in table.turns[:summary]].count("p3") == 1


def test_a_partner_named_in_the_closing_summary_confirms_before_the_answers():
    """Decided: a confirmation is never deferred past a closing line.

    It is the next turn whatever line named the partner -- the summary's
    answers and the designation wait one turn for it.
    """
    summary = _line("相方はPlayer3(p3)です。p2、最後に何かある？")
    summary.directed_questions = [
        # Asked of p2 only; p3 is named as partner, not asked.
        DirectedQuestion(target_id="p2", question="最後に何かある？")
    ]
    table = _Table({SUMMARY: summary})
    controller, coordinator = _small_table(table)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    at = [stage for _seat, stage in table.turns].index(SUMMARY)
    assert table.turns[at + 1] == ("p3", CONFIRMATION)
    assert table.turns[-1] == ("p1", "final_designation")


def test_a_partner_named_in_the_final_designation_still_confirms_and_then_the_day_closes():
    summary = _line("今日はp2で。p2、最後に何かある？", target="p2")
    summary.directed_questions = [DirectedQuestion(target_id="p2", question="最後に何かある？")]
    designation = _line("本指定はp2。相方はPlayer3(p3)です。", target="p2")
    table = _Table({SUMMARY: summary, "final_designation": designation})
    controller, coordinator = _small_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=10))

    assert table.turns[-2:] == [("p1", "final_designation"), ("p3", CONFIRMATION)]
    assert session.discussion_round.complete is True
