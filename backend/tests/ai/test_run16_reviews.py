"""Turns and prompts, from the owner's review of paid run 16 (Claude, seed 1, 2026-11-18).

- X23-1: day 1's grey views (m20-m28) were written one after another, each
  prompt carrying the views before it, and seven of eight came back as the
  first one's wording (「ハルトの見方に同意」). The round is now written at
  once: every seat's view from the table as the round began, none of the
  round's other lines, posted afterwards in a seeded order.
- X23-2: the first view (レン, m7) came when the table had heard one line from
  a seat without a CO; day 1's log now says, beside each such seat's first
  line, how many lines from seats without a CO came before it.
- X24-1: a deduction from the board (マコト genuine, so カイト fake) was called
  「マコト真読み一本で薄い」 and hanged 7-1. "Thin" now reads as no content
  (reasons, a line of argument), never as "not independent material".
- X12-2: a fake seer passed over a seat found dead that morning as 「死亡済みで
  占えない」. Every seat reads that divination is done at night, and the
  fake seer's owed result line says so too.
- X15: the hunter's day and night disagreed (a ballot reason one way, the
  guard that night the other). The night prompt quotes the seat's own lines
  and ballot reasons of the day; the next day's prompts its own night action
  and reason. Only the seat's own.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.coordinator import AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.ai.personalities import assign_personalities
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.mock import MockProvider
from app.ai.provider.scenario import ScenarioProvider
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DirectedQuestion, MorningIntentOutput
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, GameState
from app.eval.transcript import TranscriptRecorder
from tests.ai.reasoning.solver import boards
from tests.ai.test_day_turn_schedule import _all_ai_morning, _line, _session, _Table, _v3
from tests.conftest import make_controller

GREY_VIEW = "grey_view"
SUMMARY = "consensus_summary"


# -- X23-1: the grey-view round is written at once and posted after --


class _ReadingTable(_Table):
    """`_Table` that keeps every request's prompt, and requests and posts in order."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompts: list[tuple[str, str, str]] = []
        self.events: list[tuple[str, str, str]] = []

    async def generate_structured(  # type: ignore[no-untyped-def, override]
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):
        if response_schema is not MorningIntentOutput:
            self.prompts.append((self.speaker, self.stage, messages[-1].content))
            self.events.append(("ask", self.speaker, self.stage))
        return await super().generate_structured(
            system=system, messages=messages, response_schema=response_schema, **kwargs
        )

    def watch(self, coordinator: AICoordinator) -> None:
        super().watch(coordinator)
        speak = coordinator._speak

        async def posting(controller: Any, state: GameState, player_id: str, stage: str) -> Any:
            output = await speak(controller, state, player_id, stage)
            if output is not None:
                self.events.append(("post", player_id, stage))
            return output

        coordinator._speak = posting  # type: ignore[method-assign]


def _grey_text(seat: str) -> str:
    return f"灰の見え方その{seat[1:]}番。"


def _grey_table(state: GameState, asked: str | None = None) -> _ReadingTable:
    """Each seat's grey view says something only it says; with `asked`, each
    also puts a question to that seat (another seat when it is the asker)."""
    by_seat = {}
    for pid in state.players:
        output = _line(_grey_text(pid))
        if asked is not None:
            target = asked if asked != pid else "p11"
            name = state.players[target].name
            output.public_message += f"{name}はどう見てる？"
            output.directed_questions = [
                DirectedQuestion(target_id=target, question="どう見てる？")
            ]
        by_seat[(pid, GREY_VIEW)] = output
    return _ReadingTable(by_seat=by_seat)


def _log_of(prompt: str) -> str:
    return prompt.split("【当日のログ】", 1)[1].split("\n\n", 1)[0]


def _day_one(asked: str | None = None) -> tuple[GameController, AICoordinator, _ReadingTable]:
    controller = _all_ai_morning(seed=1)
    table = _grey_table(controller.state, asked)
    coordinator = _v3(controller, table)
    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 60))
    return controller, coordinator, table


def test_each_grey_view_is_written_from_the_table_as_the_round_began():
    controller, _coordinator, table = _day_one()

    viewers = [seat for seat, stage in table.turns if stage == GREY_VIEW]
    prompts = [text for _seat, stage, text in table.prompts if stage == GREY_VIEW]
    assert len(viewers) >= 6 and len(prompts) == len(viewers)
    # No view of the round reaches another's prompt ...
    for text in prompts:
        for seat in viewers:
            assert _grey_text(seat) not in text
    # ... and every seat read the same day's log.
    assert len({_log_of(text) for text in prompts}) == 1
    # They were all said, and the turn after the round reads them.
    said = [m.content for m in controller.state.chat_log if m.channel is ChatChannel.PUBLIC]
    assert all(any(_grey_text(seat) in line for line in said) for seat in viewers)
    after = next(text for _seat, stage, text in table.prompts if stage == SUMMARY)
    assert all(_grey_text(seat) in after for seat in viewers)


def test_every_grey_view_is_requested_before_the_first_is_posted_and_the_round_is_said_whole():
    # Every view asks リク something: the answers come once the round is said.
    _controller, _coordinator, table = _day_one(asked="p10")

    def at(kind: str) -> list[int]:
        return [i for i, event in enumerate(table.events) if event[::2] == (kind, GREY_VIEW)]

    asks, posts = at("ask"), at("post")
    assert asks and max(asks) < min(posts)
    stages = [stage for _seat, stage in table.turns]
    first = stages.index(GREY_VIEW)
    last = len(stages) - 1 - stages[::-1].index(GREY_VIEW)
    assert set(stages[first : last + 1]) == {GREY_VIEW}
    # The question is answered after the round, not between its lines.
    assert ("p10", "question_answer") in table.turns[last + 1 :]
    # One request per view: no more calls than before.
    assert len(asks) == len(posts)


def test_the_round_is_posted_in_the_seeded_order_the_same_every_time():
    orders = []
    for _ in range(2):
        controller, coordinator, table = _day_one()
        orders.append([seat for seat, stage in table.turns if stage == GREY_VIEW])
    assert orders[0] == orders[1]

    state = controller.state
    plan = [pid for pid in coordinator._ai_player_ids if state.players[pid].alive]
    random.Random("1:grey-view:1").shuffle(plan)
    assert orders[0] == [pid for pid in plan if pid in set(orders[0])]


def test_with_a_human_at_the_table_the_round_pauses_between_segments_and_still_posts_whole():
    controller = _all_ai_morning(seed=1)
    state = controller.state
    human = "p6"  # レン, a villager
    ai_ids = [pid for pid in state.players if pid != human]
    table = _grey_table(state)
    coordinator = AICoordinator(
        state,
        ai_ids,
        table,
        seed=1,
        reasoning=ReasoningRuntime(state, ai_ids, seed=1),
        model_decides=True,
        recorder=TranscriptRecorder(),
    )
    table.watch(coordinator)
    session = SimpleNamespace(
        controller=controller,
        discussion_lock=asyncio.Lock(),
        human_id=human,
        discussion_paused=False,
        discussion_pause_requested=False,
        discussion_step_budget=None,
    )
    human_line = "人間の席から、灰はこう見える。"
    segments: list[list[tuple[str, str]]] = []

    async def run() -> None:
        spoke = False
        for _ in range(60):
            round_state = getattr(session, "discussion_round", None)
            if round_state is not None and round_state.complete:
                return
            before = len(table.turns)
            coordinator.resume_after_human(session)
            await coordinator.advance_discussion(session)
            segments.append(table.turns[before:])
            if not spoke and any(stage == GREY_VIEW for _s, stage in segments[-1]):
                # The human speaks while the round is being said.
                controller.chat(human, human_line, "public")
                spoke = True
        raise AssertionError("the round never closed")

    asyncio.run(run())

    viewers = [seat for seat, stage in table.turns if stage == GREY_VIEW]
    assert human not in viewers and len(viewers) == len(set(viewers)) >= 6
    # The table stopped for the human between segments, mid-round included.
    assert all(len(segment) <= coordinator._segment_size for segment in segments)
    assert sum(any(stage == GREY_VIEW for _s, stage in seg) for seg in segments) >= 2
    # Written before the human spoke: no view saw the line, and none was lost.
    for _seat, stage, text in table.prompts:
        if stage == GREY_VIEW:
            assert human_line not in text
    public = [m for m in state.chat_log if m.channel is ChatChannel.PUBLIC]
    assert sum(_grey_text(seat) in m.content for m in public for seat in viewers) == len(viewers)
    assert any(m.author_id == human and m.content == human_line for m in public)


# -- X23-2: day 1's log says what the table had heard at each first line --


def _builder(state: GameState, engine: str = "v3") -> ContextBuilder:
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


def _say(controller: GameController, seat: str, text: str, claim: str | None = None) -> str:
    message_id = controller.chat(seat, text, "public")
    if claim is not None:
        controller.co(seat, claim, source_message_id=message_id)
    return message_id


def _run16_morning() -> GameController:
    """Run 16's day-1 opening in shape: COs, one line without a CO, then the first view."""
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.state.day = 1
    _say(controller, "p1", "占い師CO。Player5は白。", claim="seer")
    _say(controller, "p2", "占い師COです。Player8は白。", claim="seer")
    _say(controller, "p3", "Player2、決め手は本当にないのか？")
    _say(controller, "p4", "霊媒師CO。", claim="medium")
    _say(controller, "p5", "今日はグレランで、みんなの話を聞いてから決めたいな。")
    _say(controller, "p6", "灰の中身を見たい。")
    _say(controller, "p5", "もう一言だけ。")
    # A seat that claims on its second line: its first line was said without one.
    _say(controller, "p6", "共有者CO。", claim="freemason")
    return controller


def _discussion_user(builder: ContextBuilder, state: GameState, seat: str) -> str:
    return builder.build_discussion_context(state, seat, "grey_view", board_memo="【盤面メモ】")[1][
        0
    ].content


def test_day_one_log_marks_each_first_line_without_a_co_with_what_had_been_heard():
    controller = _run16_morning()
    state = controller.state

    log = _log_of(_discussion_user(_builder(state), state, "p10"))

    lines = log.strip().splitlines()
    assert "〈この時点でCO無しの発言N件〉は" in lines[0]
    by_id = {line.split("]", 1)[0].lstrip("["): line for line in lines[1:]}
    assert by_id["m3"].startswith("[m3] Player3〈この時点でCO無しの発言0件〉: ")
    assert by_id["m5"].startswith("[m5] Player5〈この時点でCO無しの発言1件〉: ")
    assert by_id["m6"].startswith("[m6] Player6〈この時点でCO無しの発言2件〉: ")
    # CO lines and second lines carry nothing.
    for message_id in ("m1", "m2", "m4", "m7", "m8"):
        assert "〈" not in by_id[message_id]


def test_the_mark_reaches_the_ballot_and_the_night_but_no_other_day_or_register():
    controller = _run16_morning()
    state = controller.state
    builder = _builder(state)

    vote = builder.build_vote_context(state, "p10", ["p3", "p5"], board_memo="【盤面メモ】")
    assert "Player5〈この時点でCO無しの発言1件〉" in vote[1][0].content
    structured = _builder(state, engine="v2").build_discussion_context(state, "p10", "initial")
    assert "〈この時点で" not in structured[1][0].content

    state.day = 2
    _say(controller, "p7", "2日目の最初の発言。")
    assert "〈この時点で" not in _discussion_user(builder, state, "p10")


# -- X24-1: "thin" is no content, and a deduction from the board is content --


def _doctrine(doctrine_id: str) -> str:
    return next(d.body for d in KnowledgeBase().doctrines if d.metadata["id"] == doctrine_id)


def test_thin_means_no_content_and_a_sound_deduction_from_the_board_is_content():
    register = _doctrine("jinro-17a-chat-register")

    convention = register.split("【灰の置き方の常識】", 1)[1].split("\n", 1)[0]
    # Convention 5, still a table convention ...
    assert "発言が薄い人はグレランの投票先になりやすい" in convention
    # ... with "thin" read as no content, not as a wait-and-see line or a
    # conclusion someone else reached first.
    assert "薄いとは中身(理由や筋道)がないこと" in convention
    assert (
        "盤面(CO・判定・投票・死体)から筋の通る推理を出した発言は、"
        "ほかの人と同じ結論でも、根拠が一つの事実だけでも薄くない"
    ) in convention
    assert "様子見" not in convention


def test_the_reading_viewpoints_ask_for_reasons_not_independent_material():
    expert = _doctrine("grey-reading-expert")
    average = _doctrine("grey-reading-average")

    for body in (expert, average):
        assert "自分の材料" not in body
        assert "理由なしに他人の結論をなぞっているだけか" in body
    assert "同じ結論でも、理由を自分で言えていれば後追いではない" in expert


# -- X12-2: divination is done at night --


def _register_for(role: RoleName) -> list[str]:
    state = make_controller(seed=4).state
    seat = next(p.player_id for p in state.players_by_role(role))
    return [
        d.metadata["id"]
        for d in KnowledgeBase().select(KnowledgeContext(state, seat, engine="v3"), limit=32)
    ]


def test_every_seat_reads_that_a_seat_dead_in_the_morning_was_alive_when_divined():
    register = _doctrine("jinro-17a-chat-register")

    assert "【夜の順番】占い・護衛・襲撃は夜のうちに行われ、結果は翌朝わかる。" in register
    assert (
        "朝に死体で見つかった人も、その夜の占いの時点では生きていたので、前夜の占い先でありうる"
        in register
    )
    for role in (RoleName.VILLAGER, RoleName.WEREWOLF, RoleName.MADMAN, RoleName.HUNTER):
        assert "jinro-17a-chat-register" in _register_for(role)


def test_the_fake_seer_reads_the_timing_in_its_guidance_and_in_its_owed_result_line():
    assert "今朝の死体もその夜の占いの時点では生きていた" in _doctrine("fake-seer-objectives")

    ai_ids = [f"p{i}" for i in range(1, 17)]
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FREEMASON}, day=2
    )
    boards.claim(state, "p1", RoleName.SEER, day=1)
    boards.verdict(state, "p1", "seer", "p9", False, day=1, referenced_day=0)
    runtime = ReasoningRuntime(state, ai_ids, seed=1)
    coordinator = AICoordinator(state, ai_ids, MockProvider(seed=1), seed=1, reasoning=runtime)

    lines = coordinator._required_lines(state, "p1", runtime.discussion_decision(state, "p1"))

    owed = next(line for line in lines if "1日目夜の占い結果" in line)
    assert "占いは夜のうちに行うので、占い先はその夜に生きていた人" in owed
    assert "今朝の死体も、その夜の占いの時点では生きていた" in owed


# -- X15: the seat's own day at night, and its own night the next day --


def _night_prompt(builder: ContextBuilder, state: GameState, seat: str, action: str) -> str:
    return builder.build_night_action_context(
        state, seat, action, list(state.alive_ids()), board_memo="【盤面メモ】"
    )[1][0].content


def test_the_night_prompt_quotes_the_seat_s_own_lines_and_ballot_reason_only():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    hunter = state.players_by_role(RoleName.HUNTER)[0].player_id
    other = next(pid for pid in state.alive_ids() if pid != hunter)
    said = _say(controller, hunter, "占い2人とも、昨夜の占い先が生きている以上は真ではありえない。")
    _say(controller, other, "別の席の発言。")
    builder = _builder(state)
    builder.record_ballot(
        hunter, day=2, round_number=1, target_id="p3", reason="占い先が生きているので偽。"
    )
    builder.record_ballot(other, day=2, round_number=1, target_id="p4", reason="他席の理由。")
    builder.record_ballot(hunter, day=1, round_number=1, target_id="p5", reason="昨日の理由。")
    state.phase = Phase.NIGHT

    body = _night_prompt(builder, state, hunter, "guard")

    block = body.split("【今日の自分の発言と投票】", 1)[1].split("\n\n", 1)[0]
    assert f"- [{said}] 占い2人とも、昨夜の占い先が生きている以上は真ではありえない。" in block
    assert "- 投票: Player3（そのとき書いた理由: 占い先が生きているので偽。）" in block
    assert "別の席の発言" not in block
    assert "他席の理由" not in body and "昨日の理由" not in body
    # The wolves' team chat at night carries the same block for each wolf.
    wolf = state.players_by_role(RoleName.WEREWOLF)[0].player_id
    _say_at_night = builder.build_wolf_chat_context(state, wolf, board_memo="M")[1][0].content
    assert "【今日の自分の発言と投票】" not in _say_at_night  # this wolf said nothing today
    # A seat that neither spoke nor voted today gets no block.
    assert "【今日の自分の発言と投票】" not in _night_prompt(builder, state, wolf, "attack")


def _day_user(builder: ContextBuilder, state: GameState, seat: str) -> tuple[str, str]:
    discussion = builder.build_discussion_context(
        state, seat, "initial_view", board_memo="【盤面メモ】"
    )[1][0].content
    vote = builder.build_vote_context(state, seat, ["p3"], board_memo="【盤面メモ】")[1][0].content
    return discussion, vote


def test_the_next_day_carries_the_seat_s_own_night_action_and_nobody_else_s():
    state = make_controller(seed=4).state
    state.phase = Phase.DISCUSSION
    state.day = 5
    hunter = state.players_by_role(RoleName.HUNTER)[0].player_id
    wolf = state.players_by_role(RoleName.WEREWOLF)[0].player_id
    partner = state.players_by_role(RoleName.WEREWOLF)[1].player_id
    villager = state.players_by_role(RoleName.VILLAGER)[0].player_id
    builder = _builder(state)
    builder.record_night_action(
        hunter, day=4, action_type="guard", target_id="p1", reason="本物の共有と思うので守る。"
    )
    builder.record_night_action(
        wolf, day=4, action_type="attack", target_id="p1", reason="進行役を落とす。"
    )

    for body in _day_user(builder, state, hunter):
        assert (
            "【昨夜の自分の行動】4日目夜 護衛 → Player1"
            "（そのとき書いた理由: 本物の共有と思うので守る。）。卓には出ていないこと"
        ) in body
        assert "進行役を落とす" not in body
    for body in _day_user(builder, state, wolf):
        assert "【昨夜の自分の行動】4日目夜 襲撃 → Player1" in body
        assert "本物の共有と思うので守る" not in body
    # Not a partner's, not anyone else's.
    for seat in (partner, villager):
        for body in _day_user(builder, state, seat):
            assert "【昨夜の自分の行動】" not in body
    # Last night's only.
    state.day = 6
    assert all("【昨夜の自分の行動】" not in body for body in _day_user(builder, state, hunter))


class _CapturingScenario:
    def __init__(self) -> None:
        self.inner = ScenarioProvider(seed=3)
        self.prompts: list[tuple[str, str]] = []

    async def generate_structured(self, *, system, messages, response_schema, **kwargs):  # type: ignore[no-untyped-def]
        self.prompts.append((response_schema.__name__, messages[-1].content))
        return await self.inner.generate_structured(
            system=system, messages=messages, response_schema=response_schema, **kwargs
        )


def test_the_coordinator_records_ballots_and_night_actions_for_the_seat_s_own_prompts():
    ai_ids = [f"p{i}" for i in range(1, 17)]
    controller = make_controller(seed=4)
    state = controller.state
    session = SimpleNamespace(controller=controller)
    controller.start_game()
    runtime = ReasoningRuntime(state, ai_ids, seed=4)
    provider = _CapturingScenario()
    coordinator = AICoordinator(
        state, ai_ids, provider, seed=4, reasoning=runtime, model_decides=True
    )
    asyncio.run(coordinator.run_night_phase(session))
    controller.start_discussion()
    hunter = state.players_by_role(RoleName.HUNTER)[0].player_id
    asyncio.run(coordinator._speak(controller, state, hunter, "initial_view"))
    controller.end_discussion()
    asyncio.run(coordinator._cast_vote(controller, state, hunter))
    ballot = state.pending_votes[hunter]
    target = next(pid for pid in state.alive_ids() if pid not in (hunter, ballot))
    for voter in state.alive_ids():
        if voter not in state.pending_votes:
            controller.vote(voter, target if voter != target else ballot)
    controller.resolve_votes()
    controller.start_night()
    asked = len(provider.prompts)
    asyncio.run(coordinator.run_night_phase(session))

    guard_prompt = next(
        text
        for name, text in provider.prompts[asked:]
        if name == "NightActionOutput" and "【夜行動: 護衛(guard)】" in text
    )
    block = guard_prompt.split("【今日の自分の発言と投票】", 1)[1].split("\n\n", 1)[0]
    assert f"- 投票: {state.players[ballot].name}（そのとき書いた理由: " in block
    guarded = next(r.target_id for r in state.guard_records if r.day == 1)

    controller.start_discussion()
    asked = len(provider.prompts)
    asyncio.run(coordinator._speak(controller, state, hunter, "initial_view"))
    day_two = provider.prompts[asked][1]
    assert f"【昨夜の自分の行動】1日目夜 護衛 → {state.players[guarded].name}（" in day_two
