"""Who gets the floor during the day, after the owner's review of seed 1 (2026-11-17).

Each section pins one fix from `seed1-review-causes.md` §2.2, and every test
drives the round the way a game does -- `run_discussion_round` or
`advance_discussion` -- not only the scheduler's internal pick: the partner
confirmation's own unit test passed for months while the loop around it
handed the confirmation to the reply queue, so the forced turn never ran.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.coordinator import AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    ChatDiscussionOutput,
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
)
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.state import ChatChannel, GameState, VoteRecord
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript, TranscriptRecorder
from app.sessions.models import DiscussionRoundState
from scripts import evaluate
from tests.conftest import make_controller

SUMMARY = "consensus_summary"
CONFIRMATION = "freemason_confirmation"


class _Table:
    """A staged provider and a spy: every turn as (seat, stage), every model call by seat.

    `script` answers a stage; `by_seat` answers one seat's turns at a stage,
    as (seat, stage) -> output, and wins over `script`. Anything else is a
    plain line with no candidate and no question. A (seat, stage) in
    `silent` fails every request of that turn. `queued` holds outputs a
    stage gives in turn, one per turn, ahead of everything else.
    """

    def __init__(
        self,
        script: dict[str, DiscussionOutput] | None = None,
        by_seat: dict[tuple[str, str], DiscussionOutput] | None = None,
        silent: set[tuple[str, str]] | None = None,
        queued: dict[str, list[DiscussionOutput]] | None = None,
    ) -> None:
        self.script = script or {}
        self.queued = queued or {}
        self.by_seat = by_seat or {}
        self.silent = silent or set()
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
        if (self.speaker, self.stage) in self.silent:
            return None
        if self.queued.get(self.stage):
            # One request per turn here: the scripted outputs always parse.
            output = self.queued[self.stage].pop(0).model_copy(deep=True)
            if response_schema is ChatDiscussionOutput:
                return ChatDiscussionOutput.from_discussion_output(output)
            return output
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


# -- 1-A4 / 2-A1 / 3-A1: the opening is the duty speakers plus three seats --


def _seer_morning(seed: int = 1) -> GameController:
    """The all-AI table at day 1 with the seer holding its night-0 result."""
    controller = GameController(
        session_id=f"opening-{seed}", player_specs=evaluate._make_specs(), seed=seed
    )
    controller.start_game()
    state = controller.state
    seer = next(p.player_id for p in state.players.values() if p.role.value == "seer")
    target = next(pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id))
    controller.submit_night_action(seer, "divine", target)
    controller.resolve_night()
    controller.start_discussion()
    return controller


def test_the_opening_is_every_duty_speaker_plus_three_seats_without_a_co():
    """Seed 1's mornings: duty seats filled the cap of five, so one gray seat opened.

    The cap counted the result holders and the planned COs too; from day 2
    on, one or two gray seats opened the day at most.
    """
    controller = _seer_morning(seed=1)
    state = controller.state
    coordinator = _v3(controller, _Table())
    alive = [pid for pid in state.players if state.players[pid].alive]

    round_state = coordinator._coded_discussion_round(state, alive)

    duty = round_state.order[: round_state.immediate_count]
    opening = round_state.order[round_state.immediate_count :]
    assert len(duty) >= 4  # the seer's result and the planned COs
    assert len(opening) == 3
    assert not set(opening) & set(duty)
    assert all(not [c for c in state.co_declarations if c.player_id == pid] for pid in opening)


def _tied_day_two() -> GameState:
    """Day 2 where no seat has more to say than another; p1, p10 and p11 spoke on day 1."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 1
    for pid in ("p1", "p10", "p11"):
        controller.chat(pid, "見解を述べます。")
    state.day = 2
    return state


def test_tied_seats_open_by_fewest_lines_then_a_seeded_draw_never_by_seat_id():
    """Eight of thirteen later days opened with the string-order first gray seats.

    p0, p10, p11 ... sort before p2, so ミオ (p5), レン (p6) and サクラ (p7)
    sat silent four days running in seed 1.
    """
    state = _tied_day_two()
    ai_ids = [f"p{i}" for i in range(1, 17)]
    picks = set()
    for seed in range(1, 7):
        runtime = ReasoningRuntime(state, ai_ids, seed=seed)
        runtime.refresh(state)
        assert len({c.value for c in runtime.speech_candidates(state)}) == 1

        chosen = runtime.select_opening_speakers(state)

        assert len(chosen) == 3
        # The three who already spoke yield to those who have not.
        assert not {"p1", "p10", "p11"} & set(chosen)
        again = ReasoningRuntime(state, ai_ids, seed=seed)
        again.refresh(state)
        assert again.select_opening_speakers(state) == chosen
        picks.add(tuple(sorted(chosen)))
    # Drawn per game, not read off the seat ids (p12, p13, p14 would be next).
    assert len(picks) > 1
    assert ("p12", "p13", "p14") not in picks


def test_a_tied_vote_has_no_majority_to_stand_against():
    """The opening's "against the majority" bonus read a tie as a majority for p3.

    `max(sorted(counts))` broke a 2-2 split by seat id, so every seat leaning
    to p9 was ranked as a dissenter and opened the next day ahead of others.
    """
    state = _tied_day_two()
    for voter, target in (("p1", "p9"), ("p2", "p9"), ("p4", "p3"), ("p5", "p3")):
        state.vote_records.append(VoteRecord(voter_id=voter, target_id=target, day=1, round=1))
    runtime = ReasoningRuntime(state, [f"p{i}" for i in range(1, 17)], seed=1)

    assert runtime._majority_target(PublicFactLedger(state)) is None

    state.vote_records.append(VoteRecord(voter_id="p6", target_id="p9", day=1, round=1))
    assert runtime._majority_target(PublicFactLedger(state)) == "p9"


def test_seats_answering_a_correction_are_queued_in_a_seeded_order_not_by_seat_id():
    """The runtime hands over the seats a correction moved sorted by id."""
    state = _tied_day_two()
    ai_ids = [f"p{i}" for i in range(1, 17)]
    moved = sorted(["p2", "p3", "p5", "p7", "p10", "p12"])

    def queued(seed: int) -> tuple[str, ...]:
        runtime = ReasoningRuntime(state, ai_ids, seed=seed)
        coordinator = AICoordinator(state, ai_ids, _Table(), seed=seed, reasoning=runtime)
        runtime.queue_reassessment_speakers(moved)
        round_state = DiscussionRoundState(day=2, order=[], stage="initial_view", max_total=30)
        speaker, _stage = coordinator._next_discussion_speaker(state, round_state)
        assert speaker is not None
        return (speaker, *round_state.reply_queue)

    orders = {queued(seed) for seed in range(1, 7)}

    assert all(sorted(order) == moved for order in orders)
    assert len(orders) > 1
    assert queued(3) == queued(3)


# -- 1-A4 (3) / 2-A2: from day 2, every silent seat gets one short turn --

OPEN_FLOOR = "open_floor"


def _day_two_table(
    table: _Table, *, observers: set[str] | None = None
) -> tuple[GameController, AICoordinator]:
    """Day 2 around a living human p0: sixteen AI seats, p1 leads as freemason."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    controller.co("p1", "freemason")
    ai_ids = [f"p{i}" for i in range(1, 17)]
    coordinator = AICoordinator(
        state,
        ai_ids,
        table,
        seed=4,
        reasoning=ReasoningRuntime(state, ai_ids, seed=4),
        model_decides=True,
        observer_player_ids=observers,
        recorder=TranscriptRecorder(),
    )
    table.watch(coordinator)
    return controller, coordinator


def test_later_days_give_every_silent_seat_one_short_turn_before_the_summary():
    """Seed 1, days 2-5: 37 of 83 ballots came from seats with no turn that day.

    On a later day only the morning's openers and the seats answering or
    answered spoke; everyone else voted without a word. Each living AI seat
    with no line yet now gets one short turn, under a stage that asks for
    nothing but what it wants to say.
    """
    table = _Table()
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), timeout=30))

    state = controller.state
    summary = [stage for _seat, stage in table.turns].index(SUMMARY)
    before = {seat for seat, _stage in table.turns[:summary]}
    assert before == {f"p{i}" for i in range(1, 17)}
    assert "p0" not in {seat for seat, _stage in table.turns}
    swept = [seat for seat, stage in table.turns if stage == OPEN_FLOOR]
    assert swept and len(swept) == len(set(swept))
    # Each had no line today when its turn came: one turn each, never a second.
    for seat in swept:
        first = next(i for i, (s, _stage) in enumerate(table.turns) if s == seat)
        assert table.turns[first] == (seat, OPEN_FLOOR)
    round_state = session.discussion_round
    assert len(round_state.outputs) <= round_state.max_total
    assert len(_public(state, day=2)) == len(round_state.outputs)
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    assert {u.effective_length_limit for u in utterances if u.stage == OPEN_FLOOR} == {60}


def test_the_silent_seat_turn_never_goes_to_an_observer_or_the_human():
    table = _Table()
    controller, coordinator = _day_two_table(table, observers={"p9"})

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    seats = {seat for seat, _stage in table.turns}
    assert "p9" not in seats and "p0" not in seats


def test_day_one_keeps_its_roll_call_and_has_no_silent_seat_turn():
    controller = _all_ai_morning(seed=1)
    table = _Table()
    coordinator = _v3(controller, table)

    asyncio.run(coordinator.run_discussion_round(_session(controller)))

    assert OPEN_FLOOR not in {stage for _seat, stage in table.turns}


def _builder(state: GameState, engine: str = "v3") -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players.values() if p.role.value == "werewolf"]
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


def _scene(state: GameState, stage: str, engine: str = "v3") -> str:
    marker = "【いまの場面】" if engine == "v3" else "【議論段階】"
    _system, messages = _builder(state, engine).build_discussion_context(
        state, "p1", stage, board_memo="【盤面メモ】" if engine == "v3" else None
    )
    return next(line for line in messages[0].content.splitlines() if line.startswith(marker))


def test_the_silent_seat_turn_asks_for_nothing_but_what_the_seat_wants_to_say():
    state = make_controller(seed=4).state
    state.day = 2

    scene = _scene(state, OPEN_FLOOR)

    assert scene.startswith(
        "【いまの場面】ひとこと。今日はまだ話していない。いま言いたいことを言う。60字以内。"
    )
    structured = _scene(state, OPEN_FLOOR, engine="legacy")
    assert structured.startswith("【議論段階】ひとこと。")
    assert "いま言いたいこと" in structured


# -- 3-A3: no seat is appointed to argue against the majority --


def test_a_table_piling_onto_one_seat_gets_no_appointed_reviewer():
    """The owner: a balancing voice should emerge, not be assigned by code."""
    piling = _line("p5が怪しい。", target="p5")
    table = _Table({"initial_view": piling, OPEN_FLOOR: piling})
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    stages = [stage for _seat, stage in table.turns]
    assert stages.count("initial_view") >= 3
    assert not [stage for stage in stages if stage.startswith("minority_review")]
    assert SUMMARY in stages


def test_old_records_still_print_the_removed_stage_by_name():
    """Seed 1 (2026-11-16) carries m68 `[minority_review:p2]`; the renderer prints it as is."""
    path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/live-v3-claude-sonnet55-seeds1-3-2026-11-16/seed1/transcript-seed1.json"
    )
    transcript = GameTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    rendered = render_transcript(transcript)

    assert "[minority_review:p2]" in rendered


# -- 1-A3: an answer to the summary is a 60-character turn --

ANSWER = "summary_answer"


def _asking(text: str, *targets: str, target: str | None = None) -> DiscussionOutput:
    output = _line(text, target=target)
    output.directed_questions = [
        DirectedQuestion(target_id=seat, question="どう思う？") for seat in targets
    ]
    return output


def test_a_seat_the_summary_names_answers_in_sixty_characters_not_thirty():
    """Seed 1: seats named by the summary answered in the 30-character reaction slot."""
    table = _Table({SUMMARY: _asking("p5とp7はどう思う？", "p5", "p7")})
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    at = [stage for _seat, stage in table.turns].index(SUMMARY)
    assert table.turns[at + 1 : at + 3] == [("p5", ANSWER), ("p7", ANSWER)]
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    assert {u.effective_length_limit for u in utterances if u.stage == ANSWER} == {60}


def test_the_answer_turn_says_what_it_is_for_in_both_registers():
    state = make_controller(seed=4).state
    state.day = 2

    scene = _scene(state, ANSWER)
    structured = _scene(state, ANSWER, engine="legacy")

    assert scene.startswith("【いまの場面】まとめへの回答。")
    assert "60字以内" in scene
    assert structured.startswith("【議論段階】まとめへの回答。")


# -- 1-A4 (4) / 計画不実行: a summary that names nobody does not end the day --

PROVISIONAL = "provisional_designation"
DESIGNATION = "final_designation"
DEFENCE = "rebuttal_or_reassessment"


def _after(turns: list[tuple[str, str]], stage: str = SUMMARY) -> list[tuple[str, str]]:
    return turns[[s for _seat, s in turns].index(stage) :]


def test_a_summary_naming_nobody_goes_on_to_a_provisional_designation_and_a_defence():
    """Seed 1, day 2, m38: 「皆さんは気になるグレーの方はいますか…？」 closed the day.

    The question was in the prose only -- no named seat, no candidate -- and
    the vote began with nobody designated and nobody heard in their defence.
    """
    table = _Table(
        {
            SUMMARY: _line("占いは今日は吊らず。皆さんは気になるグレーの方はいますか…？"),
            PROVISIONAL: _line("仮指定はp5。理由は発言が少ないこと。", target="p5"),
            DESIGNATION: _line("本指定はp5。", target="p5"),
        }
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [
        ("p1", SUMMARY),
        ("p1", PROVISIONAL),
        ("p5", DEFENCE),
        ("p1", DESIGNATION),
    ]
    assert session.discussion_round.complete is True
    round_state = session.discussion_round
    assert len(round_state.outputs) <= round_state.max_total


def test_after_a_summary_naming_nobody_any_seat_still_unheard_speaks_first():
    """Day 1: p3's turn failed, so it had no line when the summary came."""
    table = _Table(silent={("p3", "initial_view")})
    controller, coordinator = _small_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 10))

    assert _after(table.turns) == [("p1", SUMMARY), ("p3", OPEN_FLOOR), ("p1", PROVISIONAL)]


def test_a_provisional_designation_naming_nobody_closes_the_day_once():
    """Bounded: one provisional designation a day, and nobody named means no defence."""
    table = _Table()
    controller, coordinator = _small_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 10))

    assert _after(table.turns) == [("p1", SUMMARY), ("p1", PROVISIONAL)]
    assert session.discussion_round.complete is True


def test_the_provisional_designation_says_who_speaks_next_and_when_the_vote_starts():
    state = make_controller(seed=4).state
    state.day = 2

    scene = _scene(state, PROVISIONAL)
    structured = _scene(state, PROVISIONAL, engine="legacy")

    assert scene.startswith("【いまの場面】仮指定。")
    assert "60字以内" in scene
    assert "弁明" in scene and "投票" in scene
    assert structured.startswith("【議論段階】仮指定。")
    assert "弁明" in structured and "投票" in structured


# -- 1-A2 / 4-A1: a designated seat is heard before the vote --


def test_the_summary_candidate_defends_even_when_the_summary_does_not_ask():
    """Seed 1: a seat first named in the closing lines had no turn to answer.

    The facilitator doctrine says to hear the candidate before the final
    designation; the summary's own candidate now always gets that turn.
    """
    table = _Table(
        {
            SUMMARY: _line("今日はp5さんで考えています。", target="p5"),
            DESIGNATION: _line("本指定はp5さん。", target="p5"),
        }
    )
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    assert _after(table.turns) == [("p1", SUMMARY), ("p5", DEFENCE), ("p1", DESIGNATION)]
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    defence = [u for u in utterances if (u.player_id, u.stage) == ("p5", DEFENCE)]
    assert [u.effective_length_limit for u in defence] == [60]


def test_seed1_day4_ask_defend_designate_stays_three_lines():
    """m69 asks ハルト and names him, m70 he defends, m71 the designation: unchanged.

    The candidate was asked and is the candidate: one turn, not two. The
    designation names a seat that spoke after the summary, so no safety net.
    ハルト's seat here is Player2: the lines name him by the table's name, as
    a candidate is only one the line names (`_named_closing_target`).
    """
    summary = _asking(
        "今日はPlayer2さんに投票します。最後に言いたいことはありますか…？", "p2", target="p2"
    )
    table = _Table(
        {
            SUMMARY: summary,
            DEFENCE: _line("村人だと示せる決定的な根拠はない。", target="p5"),
            DESIGNATION: _line("今日はPlayer2さんに投票します。", target="p2"),
        }
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [("p1", SUMMARY), ("p2", DEFENCE), ("p1", DESIGNATION)]
    assert session.discussion_round.complete is True


def test_a_designation_on_a_seat_unheard_since_the_summary_gets_one_defence_and_a_redesignation():
    """Safety net, once a day: the final designation named someone who never answered."""
    table = _Table(
        {SUMMARY: _asking("p7さん、どう思う？", "p7")},
        queued={
            DESIGNATION: [
                _line("本指定はp5さん。", target="p5"),
                # Even a second unheard name does not reopen the floor.
                _line("本指定はp9さん。", target="p9"),
            ]
        },
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [
        ("p1", SUMMARY),
        ("p7", ANSWER),
        ("p1", DESIGNATION),
        ("p5", DEFENCE),
        ("p1", DESIGNATION),
    ]
    assert session.discussion_round.complete is True
    round_state = session.discussion_round
    assert len(round_state.outputs) <= round_state.max_total


def _human_table(table: _Table) -> tuple[GameController, AICoordinator, SimpleNamespace]:
    controller, coordinator = _small_table(table)
    session = SimpleNamespace(
        controller=controller,
        discussion_lock=asyncio.Lock(),
        human_id="p0",
        discussion_paused=False,
        discussion_pause_requested=False,
        discussion_step_budget=None,
    )
    return controller, coordinator, session


def _press_until_complete(
    coordinator: AICoordinator, session: SimpleNamespace, table: _Table
) -> list[tuple[list[tuple[str, str]], bool]]:
    """Pass-turn presses until the round closes: each press's turns, and whether it waits."""
    segments: list[tuple[list[tuple[str, str]], bool]] = []

    async def run() -> None:
        for _ in range(40):
            round_state = getattr(session, "discussion_round", None)
            if round_state is not None and round_state.complete:
                return
            before = len(table.turns)
            coordinator.resume_after_human(session)
            await coordinator.advance_discussion(session)
            segments.append((table.turns[before:], session.discussion_round.awaiting_human))
        raise AssertionError("the round never closed")

    asyncio.run(run())
    return segments


def test_a_summary_naming_the_living_human_as_candidate_gives_them_the_floor_first():
    """The human is never scheduled; the round stops for them, then designates."""
    table = _Table(
        {
            SUMMARY: _line("今日はPlayer0さんで考えています。", target="p0"),
            DESIGNATION: _line("本指定はPlayer0さん。", target="p0"),
        }
    )
    _controller, coordinator, session = _human_table(table)

    segments = _press_until_complete(coordinator, session, table)

    assert "p0" not in {seat for seat, _stage in table.turns}
    at = next(i for i, (given, _wait) in enumerate(segments) if ("p1", SUMMARY) in given)
    assert segments[at][0][-1] == ("p1", SUMMARY) and segments[at][1] is True
    assert segments[at + 1] == ([("p1", DESIGNATION)], False)
    assert session.discussion_round.complete is True


# -- the human's step and pause keep what a turn opened --


def _drive(coordinator: AICoordinator, session: SimpleNamespace, mode: str) -> None:
    """Press pass (or step, one turn a press) until the round closes, as the API does."""

    async def run() -> None:
        for _ in range(80):
            round_state = getattr(session, "discussion_round", None)
            if round_state is not None and round_state.complete:
                return
            if mode == "step":
                session.discussion_paused = False
                session.discussion_pause_requested = False
                session.discussion_step_budget = 1
            coordinator.resume_after_human(session)
            await coordinator.advance_discussion(session)
        raise AssertionError("the round never closed")

    asyncio.run(run())


def test_a_step_keeps_the_answer_a_question_was_owed():
    """Replies to a turn's questions were queued after the pause check.

    A step pauses after every turn, so in step mode no question ever got its
    answer turn that day: the human stepping through the table heard a
    different (shorter) discussion than one passing through it.
    """
    schedules = {}
    for mode in ("pass", "step"):
        table = _Table(by_seat={("p2", "initial_view"): _asking("p3さん、なぜ？", "p3")})
        _controller, coordinator, session = _human_table(table)

        _drive(coordinator, session, mode)

        schedules[mode] = table.turns
    assert schedules["step"] == schedules["pass"]
    asked = schedules["pass"].index(("p2", "initial_view"))
    assert "p3" in [seat for seat, _stage in schedules["pass"][asked + 1 :]]


# -- 計画不実行: the facilitator who asked hears the answer and acts on it --

FOLLOW_UP = "facilitator_followup"


def test_the_facilitator_gets_the_floor_back_once_its_question_is_answered():
    """Seed 1, m48 and m52: the facilitator asked, was answered, and had no next turn.

    m23's 「反論を聞いてから本指定」 could not be carried out for the same
    reason: nothing gave the floor back to the seat that asked.
    """
    table = _Table(by_seat={("p1", OPEN_FLOOR): _asking("p5さん、昨日の票の理由は？", "p5")})
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    asked = table.turns.index(("p1", OPEN_FLOOR))
    assert table.turns[asked + 1][0] == "p5"
    assert table.turns[asked + 2] == ("p1", FOLLOW_UP)
    assert [stage for _seat, stage in table.turns].count(FOLLOW_UP) == 1
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    assert {u.effective_length_limit for u in utterances if u.stage == FOLLOW_UP} == {60}


def test_the_facilitator_follow_up_comes_at_most_twice_a_day():
    table = _Table(
        by_seat={("p1", OPEN_FLOOR): _asking("p5さん、理由は？", "p5")},
        queued={
            FOLLOW_UP: [
                _asking("了解。p6さんはどう？", "p6"),
                _asking("了解。p7さんは？", "p7"),
                _asking("了解。p8さんは？", "p8"),
            ]
        },
    )
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert [stage for _seat, stage in table.turns].count(FOLLOW_UP) == 2
    round_state = session.discussion_round
    assert len(round_state.outputs) <= round_state.max_total


def test_a_seat_that_does_not_lead_gets_no_follow_up():
    table = _Table(by_seat={("p4", OPEN_FLOOR): _asking("p5さん、理由は？", "p5")})
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    assert FOLLOW_UP not in {stage for _seat, stage in table.turns}


def test_the_follow_up_turn_says_what_it_is_for():
    state = make_controller(seed=4).state
    state.day = 2

    scene = _scene(state, FOLLOW_UP)

    assert scene.startswith("【いまの場面】進行役の受け。")
    assert "60字以内" in scene
    assert _scene(state, FOLLOW_UP, engine="legacy").startswith("【議論段階】進行役の受け。")


# -- 1-B2 / 1-C2: a grey-random day closes without naming anyone --


def test_a_grey_random_summary_closes_the_day_without_a_designation():
    """Seed 1: the summary and designation texts demanded one name even on a grey-random day."""
    table = _Table({SUMMARY: _line("占いは吊らず、今日はグレランで。")})
    controller, coordinator = _day_two_table(table)
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    assert _after(table.turns) == [("p1", SUMMARY)]
    assert session.discussion_round.complete is True


def test_a_grey_random_summary_still_hears_whom_it_asked_and_defends_nobody():
    summary = _asking("今日はグレランにします。p7さん、何かある？", "p7", target="p5")
    table = _Table({SUMMARY: summary})
    controller, coordinator = _day_two_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    # p5 is only the facilitator's own lean on a grey-random day: no defence.
    assert _after(table.turns) == [("p1", SUMMARY), ("p7", ANSWER)]


def test_a_grey_random_provisional_designation_closes_the_day():
    table = _Table({PROVISIONAL: _line("それなら今日はグレランで。")})
    controller, coordinator = _small_table(table)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 10))

    assert _after(table.turns) == [("p1", SUMMARY), ("p1", PROVISIONAL)]


def test_a_summary_that_only_mentions_grey_random_is_not_one():
    """Turning grey-random down, or musing about it, does not close the day."""
    for text in (
        "グレランではなく指定にしたい。皆さんは？",
        "グレランでもいいけど、誰が気になる？",
        "グレランにしないなら誰？",
    ):
        table = _Table({SUMMARY: _line(text), PROVISIONAL: _line("仮指定はp3。", target="p3")})
        controller, coordinator = _small_table(table)

        asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 10))

        assert _after(table.turns)[:2] == [("p1", SUMMARY), ("p1", PROVISIONAL)], text


def test_the_closing_scenes_say_who_answers_next_when_the_vote_starts_and_allow_grey_random():
    state = make_controller(seed=4).state
    state.day = 2
    for engine in ("v3", "legacy"):
        summary = _scene(state, SUMMARY, engine)
        designation = _scene(state, DESIGNATION, engine)
        provisional = _scene(state, PROVISIONAL, engine)

        for scene in (summary, designation, provisional):
            assert "グレラン" in scene, (engine, scene)
            assert "投票に入" in scene, (engine, scene)
        # Who speaks between the summary and the vote.
        assert "答え" in summary and "本指定" in summary, engine
        assert "弁明" in provisional and "本指定" in provisional, engine
        assert "すぐ投票に入" in designation, engine
