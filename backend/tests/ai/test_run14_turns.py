"""Turn scheduling and prompt fixes from the owner's review of run 14 (2026-11-18).

Each section pins one item of that review:

- X3: with no living freemason claimant the day has no facilitator. Code
  used to appoint one (the last seat ready to vote, else the first in the
  day's order), and in all three games that made a wolf's fake claimant the
  summariser whose lines every ballot read as 【進行役の方針】.
- X6: a seat asked by name answers before the close, in a 60-character
  answer turn; no persona caps how much a seat speaks; 【聞き入れ方】 is the
  same for every role.
- N2: yesterday's named questions and what the asked seats said next reach
  the next day.
- X7: the morning's COs and results come in a drawn order, the real medium
  among them; a freemason's report closes the morning block.
- X8, N-7, X5: the hunter's doctrine keeps its condition; a fake claimant
  may give either colour; the fox's facts read the same for every side.
- X11: no private chat for a lone survivor; a code-written summary names no
  author.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.coordinator import AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.ai.personalities import assign_personalities
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.summaries import render_question_exchanges
from app.ai.schemas import DirectedQuestion, DiscussionOutput
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, GameState, PendingQuestion, PlayerState
from app.eval.reasoning_analyzer import _Record
from app.eval.transcript import TranscriptRecorder
from app.eval.transcript_render import render_transcript_full
from app.sessions.models import DiscussionRoundState
from tests.ai.test_day_turn_schedule import (
    _all_ai_morning,
    _line,
    _public,
    _seer_morning,
    _session,
    _Table,
    _v3,
)
from tests.conftest import make_controller

SUMMARY = "consensus_summary"
CLOSING = {SUMMARY, "provisional_designation", "final_designation", "facilitator_followup"}
ANSWER = "question_answer"
OPEN_FLOOR = "open_floor"
AI_IDS = [f"p{i}" for i in range(1, 17)]


def _day_two(table: _Table, *, lead: str | None = None) -> tuple[Any, AICoordinator]:
    """Day 2 around a living human p0; `lead` claims freemason, or nobody does."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    if lead is not None:
        controller.co(lead, "freemason")
    coordinator = AICoordinator(
        state,
        AI_IDS,
        table,
        seed=4,
        reasoning=ReasoningRuntime(state, AI_IDS, seed=4),
        model_decides=True,
        recorder=TranscriptRecorder(),
    )
    table.watch(coordinator)
    return controller, coordinator


def _builder(state: GameState, fake_roles: dict[str, RoleName] | None = None) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    fakes = fake_roles or {}
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="alpha" if fakes else "gamma",
            pattern_label="偽占い1+潜伏2" if fakes else "全潜伏",
            fake_role_by_player=fakes,
            lurking_player_ids=[pid for pid in wolf_ids if pid not in fakes],
        ),
        madman_fake_role=RoleName.MEDIUM,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine="v3",
    )


# -- X3: no facilitator without a living freemason claimant --


def test_a_day_with_no_freemason_claimant_closes_without_summary_or_designation():
    """Run 14: the fallback made ダイキ, レン and ケント (wolves) the day's summariser.

    Every seat here is ready to vote, so the old fallback would have handed
    the summary to the last of them. Now the floor still gets its turns --
    every living AI seat speaks -- and the vote follows with nobody summing
    up or designating.
    """
    ready = _line("見解を述べます。", target="p5")
    ready.ready_to_vote = True
    table = _Table(script={stage: ready for stage in ("initial_view", OPEN_FLOOR, "immediate")})
    controller, coordinator = _day_two(table)
    state = controller.state
    masons = [p.player_id for p in state.players_by_role(RoleName.FREEMASON)]
    for pid in masons:
        state.players[pid].alive = False
        state.players[pid].death_day = 1
    session = _session(controller)

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 30))

    stages = {stage for _seat, stage in table.turns}
    assert not stages & CLOSING
    assert {seat for seat, _stage in table.turns} == set(AI_IDS) - set(masons)
    assert session.discussion_round.complete is True
    assert session.discussion_round.summary_done is True
    assert coordinator._context.direction_lines(2) == []
    # The audit reads the day as one without a designation.
    assert _Record(coordinator._recorder.transcript).designation(2) == (None, "")  # type: ignore[union-attr]


def test_a_living_freemason_claimant_still_sums_up_and_designates():
    table = _Table(script={SUMMARY: _line("今日はPlayer5さんで。", target="p5")})
    controller, coordinator = _day_two(table, lead="p10")

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    assert ("p10", SUMMARY) in table.turns
    assert ("p10", "final_designation") in table.turns


def test_the_scheduler_never_appoints_the_last_seat_ready_to_vote():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    coordinator = AICoordinator(state, ["p1", "p2", "p3"], _Table(), seed=1)
    ready = DiscussionOutput(public_message="投票できます。", ready_to_vote=True)
    round_state = DiscussionRoundState(
        day=state.day,
        order=["p1", "p2", "p3"],
        cursor=3,
        stage="initial_view",
        outputs=[("p1", ready), ("p2", ready), ("p3", ready)],
        speech_counts={"p1": 1, "p2": 1, "p3": 1},
        major_targets_ready=True,
        max_total=3,
    )

    assert coordinator._next_discussion_speaker(state, round_state) == (None, SUMMARY)
    assert round_state.summary_done is True
    assert coordinator._next_discussion_speaker(state, round_state)[0] is None


def _vote_body(state: GameState) -> str:
    _system, messages = _builder(state).build_vote_context(
        state, "p2", ["p4", "p6"], board_memo="【盤面メモ】"
    )
    return str(messages[0].content)


def test_the_ballot_is_told_plainly_that_there_was_no_facilitator():
    state = make_controller(seed=4).state
    state.day = 3

    body = _vote_body(state)

    assert "【進行役の方針】" not in body
    assert "【進行役】生存している共有CO者がいないため、今日は進行役のまとめ・指定は" in body


def test_with_a_living_freemason_claimant_the_ballot_has_no_such_note():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.co("p10", "freemason")
    controller.state.day = 3

    assert "【進行役】" not in _vote_body(controller.state)


def test_the_discussion_says_from_day_two_that_the_day_has_no_facilitator():
    state = make_controller(seed=4).state
    for day, expected in ((1, False), (2, True)):
        state.day = day
        _system, messages = _builder(state).build_discussion_context(
            state, "p2", "initial_view", board_memo="【盤面メモ】", required_lines=[]
        )
        assert ("今日は進行役がいない" in str(messages[0].content)) is expected, day


def test_a_medium_with_two_blacks_is_not_told_it_is_the_facilitator():
    controller = make_controller(seed=4)
    state = controller.state
    from app.engine.state import MediumRecord

    state.medium_records = [
        MediumRecord(medium_id="p14", target_id="p5", day=1, is_werewolf=True),
        MediumRecord(medium_id="p14", target_id="p7", day=2, is_werewolf=True),
    ]

    card = _builder(state)._layer_b_role(state, "p14")

    assert "【霊媒の仕事終了】" in card
    assert "進行役" not in card


# -- X6: a seat asked by name gets its answer turn --


def _quiet_seat() -> str:
    personalities = assign_personalities(AI_IDS, seed=4)
    return min(AI_IDS[1:], key=lambda pid: personalities[pid].talkativeness)


def test_a_quiet_seat_asked_after_the_budget_is_spent_still_answers_before_the_close():
    """Seed 2, day 2: マコト was asked five times and answered only after the summary."""
    quiet = _quiet_seat()
    controller, coordinator = _day_two(_Table(), lead="p10")
    state = controller.state
    for pid in AI_IDS:
        state.chat_log.append(_message(state, pid, "見解を述べます。"))
    asker = next(pid for pid in AI_IDS if pid not in (quiet, "p10"))
    question = _message(state, asker, f"Player{quiet[1:]}さん、どう思う？")
    state.chat_log.append(question)
    state.pending_questions[quiet] = [
        PendingQuestion(asker, quiet, question.content, question.message_id, 2)
    ]
    outputs = [(pid, DiscussionOutput(public_message="見解")) for pid in AI_IDS]
    round_state = DiscussionRoundState(
        day=2,
        order=list(AI_IDS),
        cursor=len(AI_IDS),
        stage="initial_view",
        outputs=outputs,
        # Well past the old cap of round(2.5 x talkativeness).
        speech_counts={**{pid: 1 for pid in AI_IDS}, quiet: 5},
        major_targets_ready=True,
        max_total=len(outputs),
    )

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) == (quiet, ANSWER)
    assert round_state.max_total == len(outputs) + 1
    state.pending_questions[quiet][0].served = True
    assert coordinator._next_discussion_speaker(state, round_state) == ("p10", SUMMARY)


def _message(state: GameState, author: str, text: str) -> Any:
    from app.engine.state import ChatMessage

    return ChatMessage(
        message_id=f"m{len(state.chat_log) + 1}",
        author_id=author,
        content=text,
        channel=ChatChannel.PUBLIC,
        day=state.day,
    )


def test_owed_answers_stop_after_two_a_seat_a_day():
    controller, coordinator = _day_two(_Table(), lead="p10")
    state = controller.state
    round_state = DiscussionRoundState(
        day=2, order=[], cursor=0, stage="initial_view", major_targets_ready=True
    )
    book = coordinator._book_for(round_state)
    given = []
    for turn in range(4):
        state.pending_questions["p5"] = [PendingQuestion("p6", "p5", "どう？", f"m{turn}", 2)]
        given.append(coordinator._take_owed_answer(state, round_state, book))
    assert given == ["p5", "p5", None, None]


def test_a_seat_asked_in_the_reply_queue_answers_in_sixty_characters():
    quiet = _quiet_seat()
    asking = _line(f"Player{quiet[1:]}さん、どう思う？")
    asking.directed_questions = [DirectedQuestion(target_id=quiet, question="どう思う？")]
    askers = [pid for pid in AI_IDS if pid not in (quiet, "p10")][:5]
    table = _Table(
        by_seat={
            (pid, stage): asking for pid in askers for stage in ("initial_view", OPEN_FLOOR)
        }
    )
    controller, coordinator = _day_two(table, lead="p10")

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))

    summary = table.turns.index(("p10", SUMMARY))
    before = table.turns[:summary]
    assert (quiet, ANSWER) in before
    # Its last turn before the summary came after the last question put to it.
    last_question = max(i for i, turn in enumerate(before) if turn[0] in askers)
    assert any(seat == quiet for seat, _stage in before[last_question:])
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    assert {u.effective_length_limit for u in utterances if u.stage == ANSWER} == {60}


def test_a_seat_that_wants_another_word_gets_it_whatever_its_persona():
    quiet = _quiet_seat()
    controller, coordinator = _day_two(_Table(), lead="p10")
    round_state = DiscussionRoundState(day=2, order=[], cursor=0)
    more = DiscussionOutput(public_message="もう一言。", needs_another_statement=True)

    coordinator._queue_turn_replies(controller.state, round_state, quiet, more)

    assert round_state.reply_queue == [quiet]


def test_the_answer_turn_says_what_it_is_for():
    state = make_controller(seed=4).state
    state.day = 2
    _system, messages = _builder(state).build_discussion_context(
        state, "p2", ANSWER, board_memo="【盤面メモ】", required_lines=[]
    )
    body = str(messages[0].content)

    assert "【いまの場面】質問への回答。名指しで聞かれた。まずその問いに答える。60字以内。" in body


def _listening_block(system: str) -> str:
    start = system.index("【聞き入れ方】")
    return system[start : system.index("【考え方】")]


def test_how_a_seat_takes_an_argument_is_the_same_for_every_role():
    """Seed 3: three of four hanged villagers had conceded; the wolves were told
    they could choose tactically, the village to concede and change."""
    state = make_controller(seed=4).state
    builder = _builder(state)
    blocks = {
        role: _listening_block(builder._layer_a_system(state, pid))
        for role, pid in (("villager", "p2"), ("werewolf", "p5"), ("madman", "p12"), ("fox", "p3"))
    }

    assert len(set(blocks.values())) == 1
    block = blocks["villager"]
    assert "戦術として選ぶ" not in block
    assert "自分で決める" in block
    assert "認めて考えを変える" not in block


# -- N2: yesterday's questions and answers --


def test_the_next_day_sees_an_answered_question_as_answered():
    """Seed 1: ユイ's 「まだ答えてないぞ」 survived; ケント's 「ランダム」 did not."""
    asking = _line("Player7さん、占い先の理由は？")
    asking.directed_questions = [DirectedQuestion(target_id="p7", question="占い先の理由は？")]
    unanswered = _line("Player0さん、どう思う？")
    unanswered.directed_questions = [DirectedQuestion(target_id="p0", question="どう思う？")]
    by_seat = {("p7", ANSWER): _line("理由はない、ランダム。")}
    for stage in ("initial_view", OPEN_FLOOR):
        by_seat[("p5", stage)] = asking
        by_seat[("p6", stage)] = unanswered
    table = _Table(by_seat=by_seat)
    controller, coordinator = _day_two(table, lead="p10")
    state = controller.state

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))
    asyncio.run(coordinator._generate_day_summary(controller, state))

    past = coordinator._day_summaries.render(chat=True)
    assert "【2日目の名指しの質問と、聞かれた人の次の発言】" in past
    answered = next(line for line in past.splitlines() if line.startswith("- Player7へ"))
    assert "Player5" in answered and "Player7の次の発言" in answered
    open_line = next(line for line in past.splitlines() if line.startswith("- Player0へ"))
    assert open_line.endswith("未回答（この日はこのあと発言なし）")


def test_only_the_latest_day_keeps_its_exchanges():
    summaries = DaySummaryManager()
    summaries.set_summary(1, "", "【公開事実】1日目")
    summaries.set_exchanges(1, "【1日目の名指しの質問と、聞かれた人の次の発言】\n- a")
    summaries.set_summary(2, "", "【公開事実】2日目")
    summaries.set_exchanges(2, "【2日目の名指しの質問と、聞かれた人の次の発言】\n- b")

    rendered = summaries.render(chat=True)

    assert "【2日目の名指しの質問" in rendered
    assert "【1日目の名指しの質問" not in rendered


def test_exchanges_are_capped_and_keep_the_latest():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    questions = []
    for index in range(10):
        message = controller.chat("p2", f"Player{index + 3}さん、質問{index}？")
        questions.append(PendingQuestion("p2", f"p{index + 3}", f"質問{index}？", message, 2))

    block = render_question_exchanges(state, 2, questions)

    lines = block.splitlines()
    assert lines[1] == "- （先の2件は省略）"
    assert len(lines) == 1 + 1 + 8
    assert "質問9" in block and "質問0" not in block


# -- X7: the order of the morning's COs says nothing about who is real --


def test_the_real_medium_speaks_in_the_morning_block_beside_a_fake_one():
    """Seed 2: the fake medium COed at m4 from the morning block, the real one at m16."""
    controller = _seer_morning(seed=2)
    state = controller.state
    coordinator = _v3(controller, _Table(), seed=2)
    fakes = coordinator._wolf_deception.fake_role_by_player
    fake_medium = next(pid for pid, role in fakes.items() if role is RoleName.MEDIUM)
    medium = state.players_by_role(RoleName.MEDIUM)[0].player_id
    alive = [pid for pid in state.players if state.players[pid].alive]

    round_state = coordinator._coded_discussion_round(state, alive)

    duty = round_state.order[: round_state.immediate_count]
    assert medium in duty and fake_medium in duty


def test_the_real_seer_does_not_always_open_the_day():
    """Run 14: the real seer was the day's first line in all three games."""
    firsts = []
    for seed in range(1, 9):
        controller = _seer_morning(seed=seed)
        state = controller.state
        coordinator = _v3(controller, _Table(), seed=seed)
        alive = [pid for pid in state.players if state.players[pid].alive]
        order = coordinator._coded_discussion_round(state, alive).order
        firsts.append(state.players[order[0]].role)
        # Drawn per game and day: the same seed gives the same order.
        again = _v3(controller, _Table(), seed=seed)._coded_discussion_round(state, alive).order
        assert again == order
    assert RoleName.SEER in firsts
    assert any(role is not RoleName.SEER for role in firsts)


def _day_two_without(controller: Any, dead: str) -> None:
    state = controller.state
    state.players[dead].alive = False
    state.players[dead].death_day = 1
    state.day = 2


def test_the_hidden_partners_takeover_co_closes_the_morning_block():
    """Seed 3, day 4: the takeover CO came as the day's twelfth line."""
    controller = _all_ai_morning(seed=1)
    table = _Table()
    coordinator = _v3(controller, table)
    leader, partner, _reveal = coordinator._freemason_public_plan  # type: ignore[misc]
    coordinator._freemason_public_plan = (leader, partner, False)
    _day_two_without(controller, leader)
    state = controller.state
    alive = [pid for pid in state.players if state.players[pid].alive]

    round_state = coordinator._coded_discussion_round(state, alive)

    assert round_state.order[round_state.immediate_count - 1] == partner


def test_the_leaders_word_that_the_partner_died_closes_the_morning_block():
    controller = _all_ai_morning(seed=1)
    coordinator = _v3(controller, _Table())
    leader, partner, _reveal = coordinator._freemason_public_plan  # type: ignore[misc]
    coordinator._freemason_public_plan = (leader, partner, False)
    controller.co(leader, "freemason")
    _day_two_without(controller, partner)
    state = controller.state
    alive = [pid for pid in state.players if state.players[pid].alive]

    round_state = coordinator._coded_discussion_round(state, alive)

    assert round_state.order[round_state.immediate_count - 1] == leader


# -- X8: the hunter keeps the doctrine's condition --


def test_the_hunter_doctrine_states_the_two_black_condition_first():
    state = make_controller(seed=4).state
    state.day = 2
    bodies = [
        doctrine.body
        for doctrine in KnowledgeBase().select(KnowledgeContext(state=state, player_id="p1"))
        if doctrine.metadata["id"] == "hunter-night"
    ]

    assert len(bodies) == 1
    body = bodies[0]
    assert "条件はひとつだけ: 3狼村で、その霊媒が黒（人狼判定）を2回出していること" in body
    assert "黒が0回・1回の霊媒は、白を何回出していても仕事終了ではなく" in body
    assert "進行役" not in body


def test_the_hunter_is_told_each_medium_claimants_black_count():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 3
    co = controller.chat("p14", "霊媒CO。Player5は黒。")
    controller.co("p14", "medium", source_message_id=co)
    controller.public_result("p14", "medium", "p5", True, source_message_id=co)

    card = _builder(state)._layer_b_role(state, "p1")

    assert "【霊媒CO者が公開した黒の数】Player14=1回" in card
    assert "3狼村で黒を2回出した霊媒だけ" in card


# -- N-7: a fake claimant may give either colour --


def test_a_fake_claimant_may_report_either_colour_and_an_executed_partners_black():
    state = make_controller(seed=4).state
    builder = _builder(state, fake_roles={"p5": RoleName.MEDIUM})

    wolf = builder._layer_b_role(state, "p5")
    madman = builder._layer_b_role(state, "p12")

    assert "嘘の結果で名指し" not in wolf
    for card in (wolf, madman):
        assert "騙りの結果は、白でも黒でも出せます" in card
    assert "生きている仲間を黒と名指ししてはいけません" in wolf
    assert "処刑された仲間の霊媒結果を黒と言うのは" in wolf


def test_the_guard_refuses_a_black_on_a_living_partner_only():
    guard = FakeClaimGuard(wolf_team_ids={"w1", "w2"})

    assert guard.can_claim_result("w1", "w2", claimed_is_werewolf=True) is False
    assert guard.can_claim_result("w1", "w2", claimed_is_werewolf=True, target_alive=False)
    assert guard.can_claim_result("w1", "v1", claimed_is_werewolf=False)


# -- X5: the fox's facts read the same for every side --


def test_every_side_reads_the_same_fox_facts():
    state = make_controller(seed=4).state
    state.day = 3
    bodies = {}
    for pid in ("p2", "p5", "p12", "p3", "p1"):
        bodies[pid] = [
            doctrine.body
            for doctrine in KnowledgeBase().select(KnowledgeContext(state=state, player_id=pid))
            if doctrine.metadata["id"] == "fox-care"
        ]
    assert all(len(found) == 1 for found in bodies.values())
    assert len({found[0] for found in bodies.values()}) == 1
    body = bodies["p2"][0]
    assert "狼側も" not in body
    assert "村の勝ちにも人狼陣営の勝ちにも" in body


def test_the_village_card_says_how_the_village_loses_as_the_wolf_card_does():
    state = make_controller(seed=4).state
    card = _builder(state)._layer_b_role(state, "p2")

    assert "人狼が人狼以外（狂人・妖狐も数える）と同数以上になれば負け" in card


# -- X11: lone survivors, and the code-written summary --


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ) -> SchemaT | None:
        del system, messages, response_schema, kwargs
        self.calls += 1
        return None


def _night(controller: Any) -> GameState:
    state = controller.state
    state.phase = Phase.NIGHT
    state.day = 3
    return state


def _kill(state: GameState, players: list[PlayerState]) -> None:
    for player in players:
        player.alive = False
        player.death_day = 2


def test_a_lone_wolf_and_a_lone_freemason_are_not_asked_to_chat():
    controller = make_controller(seed=4)
    state = _night(controller)
    provider = _CountingProvider()
    coordinator = AICoordinator(state, AI_IDS, provider, seed=1)
    _kill(state, state.players_by_role(RoleName.WEREWOLF)[1:])
    _kill(state, state.players_by_role(RoleName.FREEMASON)[1:])

    asyncio.run(coordinator._run_wolf_chat_round(controller, state))
    asyncio.run(coordinator._run_freemason_chat_round(controller, state))

    assert provider.calls == 0
    assert not [m for m in state.chat_log if m.channel is not ChatChannel.PUBLIC]


def test_two_living_members_still_chat():
    controller = make_controller(seed=4)
    state = _night(controller)
    provider = _CountingProvider()
    coordinator = AICoordinator(state, AI_IDS, provider, seed=1)
    _kill(state, state.players_by_role(RoleName.WEREWOLF)[2:])

    asyncio.run(coordinator._run_wolf_chat_round(controller, state))

    assert provider.calls > 0


def test_a_summary_written_in_code_names_no_author():
    table = _Table()
    controller, coordinator = _day_two(table, lead="p10")
    state = controller.state

    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))
    asyncio.run(coordinator._generate_day_summary(controller, state))

    transcript = coordinator._recorder.transcript  # type: ignore[union-attr]
    rendered = render_transcript_full(transcript)
    assert "- 要約（コードが作成）:" in rendered
    narrator = next(u for u in transcript.utterances if u.kind == "summary").player_name
    assert f"要約（{narrator}）" not in rendered
    assert _public(state, day=2)
