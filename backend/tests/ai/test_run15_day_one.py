"""Day 1 discussion and grey reading, from the owner's review of runs 14 and 15 (2026-11-18).

- X18 / H1: day 1 gave each grey one line, said before most greys had spoken.
  Every seat without a CO now gets a second turn once all first views are in
  (`grey_view`), to say how the grey looks -- white placements and doubts.
- X21: a first view could put a seat that had not spoken yet down as the hang
  (run 15: 6 of 8). The discussion prompt now names the seats whose turn has
  not come, as the vote prompt already did, and the plan note's hang target is
  told not to pick one for its silence.
- X19 / H5: the plan note's `trusted` reads as "white, not today's hang
  position"; the schema is unchanged.
- X17: day 1's first-view menu names the seat's own way of reading the grey;
  the 17A conventions say night-0 reasons are usually "なんとなく", and carry
  the owner's table conventions 3 and 5.
- Conventions 4, 6, 7 and the reading viewpoints come by skill tier.
- X20: the seer's night prompt shows how the table placed each unclaimed seat.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from anthropic import transform_schema

from app.ai.context import (
    CHAT_DISCUSSION_OUTPUT_INSTRUCTION,
    CHAT_TEAM_READ_LINE,
    ContextBuilder,
    DaySummaryManager,
)
from app.ai.coordinator import AICoordinator
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.ai.personalities import assign_personalities, chat_stage_max_chars
from app.ai.schema_limits import count_schema_parameters
from app.ai.schemas import ChatDiscussionOutput, DirectedQuestion
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState, VoteRecord
from tests.ai.test_day_turn_schedule import (
    _all_ai_morning,
    _day_two_table,
    _line,
    _session,
    _Table,
    _v3,
)
from tests.conftest import make_controller

GREY_VIEW = "grey_view"
SUMMARY = "consensus_summary"


def _builder(state: GameState, skills: dict[str, str] | None = None) -> ContextBuilder:
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
        engine="v3",
        skills=skills,
    )


def _user(builder: ContextBuilder, state: GameState, seat: str, stage: str) -> str:
    _system, messages = builder.build_discussion_context(
        state, seat, stage, board_memo="【盤面メモ】"
    )
    return messages[0].content


def _scene(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("【いまの場面】"))


# -- X18 / H1: a second turn for every grey once the grey has spoken --


def _day_one_round(table: _Table) -> tuple[GameState, list[tuple[str, str]]]:
    controller = _all_ai_morning(seed=1)
    coordinator = _v3(controller, table)
    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 60))
    return controller.state, table.turns


def test_every_seat_without_a_co_gets_one_grey_view_turn_after_all_first_views():
    state, turns = _day_one_round(_Table())

    stages = [stage for _seat, stage in turns]
    claimed = {claim.player_id for claim in state.co_declarations}
    viewers = [seat for seat, stage in turns if stage == GREY_VIEW]
    unclaimed = [pid for pid in state.alive_ids() if pid not in claimed]
    # Every living unclaimed seat, once each; no claimant.
    assert sorted(viewers) == sorted(unclaimed)
    assert not set(viewers) & claimed
    # After every first view, before the summary.
    first_grey_view = stages.index(GREY_VIEW)
    last_first_view = max(i for i, stage in enumerate(stages) if stage == "initial_view")
    assert last_first_view < first_grey_view
    assert max(i for i, stage in enumerate(stages) if stage == GREY_VIEW) < stages.index(SUMMARY)
    # Each seat's grey view comes after its own first turn (a planned claimant
    # whose scripted line made no CO spoke first in the morning block).
    for seat in viewers:
        first = next(i for i, (s, _stage) in enumerate(turns) if s == seat)
        assert turns[first][1] in ("initial_view", "immediate")
        assert first < turns.index((seat, GREY_VIEW))


def test_a_seat_that_has_already_said_two_lines_is_passed_over():
    asking = _line("ソウタはどう見てる？")
    asking.directed_questions = [DirectedQuestion(target_id="p4", question="どう見てる？")]
    table = _Table(by_seat={("p7", "initial_view"): asking})

    state, turns = _day_one_round(table)

    position = turns.index(("p4", GREY_VIEW)) if ("p4", GREY_VIEW) in turns else None
    lines_before = [
        seat for seat, stage in (turns[:position] if position is not None else turns)
    ].count("p4")
    # ソウタ answered サクラ's question: two lines before any grey view, so none.
    assert lines_before >= 2
    assert position is None


def test_grey_view_turns_are_counted_and_capped_at_eighty_characters():
    table = _Table()
    controller = _all_ai_morning(seed=1)
    coordinator = _v3(controller, table)
    session = _session(controller)
    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(session), 60))

    round_state = session.discussion_round
    assert len(round_state.outputs) <= round_state.max_total
    assert chat_stage_max_chars(GREY_VIEW) == 80
    utterances = coordinator._recorder.transcript.utterances  # type: ignore[union-attr]
    assert {u.effective_length_limit for u in utterances if u.stage == GREY_VIEW} == {80}


def test_no_grey_view_on_day_two_or_outside_v3():
    table = _Table()
    controller, coordinator = _day_two_table(table)
    asyncio.run(asyncio.wait_for(coordinator.run_discussion_round(_session(controller)), 30))
    assert GREY_VIEW not in {stage for _seat, stage in table.turns}

    legacy_table = _Table()
    controller = _all_ai_morning(seed=1)
    legacy = AICoordinator(controller.state, list(controller.state.players), legacy_table, seed=1)
    legacy_table.watch(legacy)
    asyncio.run(asyncio.wait_for(legacy.run_discussion_round(_session(controller)), 60))
    assert GREY_VIEW not in {stage for _seat, stage in legacy_table.turns}


def test_the_grey_view_scene_asks_for_white_placements_beside_doubts():
    state = make_controller(seed=4).state
    state.day = 1

    scene = _scene(_user(_builder(state), state, "p1", GREY_VIEW))

    assert scene.startswith("【いまの場面】灰の見え方。全員が一度ずつ話したあとの、灰の見え方。")
    assert "白く見える人(今日の吊り位置ではないと置く人)" in scene
    assert "気になる人" in scene
    assert "80字以内" in scene


# -- X17-1: day 1's first-view menu --


def test_day_one_first_view_names_reading_the_grey_and_other_days_keep_their_menu():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _builder(state)

    day_one = _scene(_user(builder, state, "p1", "initial_view"))
    assert "初日の最初の発言" in day_one
    assert "灰をどう見ていくか" in day_one
    assert "誰かへの質問" in day_one

    state.day = 2
    day_two = _scene(_user(builder, state, "p1", "initial_view"))
    assert "初日" not in day_two
    assert "内訳の見立て、吊り方針、誰かへの質問のどれか" in day_two


# -- X21: the seats whose turn has not come --


def _post(state: GameState, number: int, author: str, text: str) -> None:
    state.chat_log.append(
        ChatMessage(f"m{number}", author, text, ChatChannel.PUBLIC, day=state.day)
    )


def test_the_discussion_prompt_names_the_seats_whose_turn_has_not_come():
    controller = _all_ai_morning(seed=1)
    state = controller.state
    _post(state, 1, "p16", "占い師CO。マコト白。")
    _post(state, 2, "p7", "グレランで。")

    user = _user(_builder(state), state, "p10", "initial_view")

    line = next(
        item for item in user.splitlines() if item.startswith("【今日まだ手番が来ていない席】")
    )
    assert "ソウタ" in line and "ハルト" in line
    # Not the seats that spoke, not the seat itself, not the dead.
    for name in ("ケント", "サクラ", "リク", "ノゾミ"):
        assert name not in line.split("（")[0]
    assert "黙っているのではない" in line


def test_no_waiting_line_once_everyone_has_spoken_and_the_human_is_never_listed():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 1
    for number, pid in enumerate(state.alive_ids(), start=1):
        if pid not in ("p0", "p1"):
            _post(state, number, pid, "見解です。")

    user = _user(_builder(state), state, "p1", "initial_view")

    # p0 is the human: it speaks when it chooses, so it is not "waiting".
    assert "【今日まだ手番が来ていない席】" not in user


def test_the_hang_target_is_not_picked_for_a_silence_the_schedule_imposed():
    assert "まだ手番が来ていない席を、話していないことで選ばない" in (
        CHAT_DISCUSSION_OUTPUT_INSTRUCTION
    )


# -- X19 / H5: trusted is a white placement, and the wire schema is unchanged --


def test_trusted_reads_as_white_and_not_todays_hang_position():
    assert (
        "白く見ていて、今日の吊り位置ではないと置くplayer_id" in CHAT_DISCUSSION_OUTPUT_INSTRUCTION
    )
    assert "信用しているplayer_id" not in CHAT_DISCUSSION_OUTPUT_INSTRUCTION
    assert "trusted" in CHAT_TEAM_READ_LINE
    assert count_schema_parameters(transform_schema(ChatDiscussionOutput)) == (17, 6)


# -- X17 / conventions 3 and 5: the 17A register --


def _register() -> str:
    return next(
        doctrine.body
        for doctrine in KnowledgeBase().doctrines
        if doctrine.metadata["id"] == "jinro-17a-chat-register"
    )


def test_the_register_says_day_one_is_about_the_grey_and_night_zero_reasons_are_usually_none():
    body = _register()

    assert "話題の中心は灰の発言" in body
    assert "理由は「なんとなく」が普通で、聞くとしても一度で足りる" in body
    assert "白く見える人を「今日の吊り位置ではない」と置くこともする" in body
    # Convention 5; what "thin" means is pinned in test_run16_reviews.py (X24-1).
    assert "発言が薄い人はグレランの投票先になりやすい" in body
    assert "まだ手番が来ていない人は黙っているのではない" in body


# -- conventions 4, 6, 7 and the viewpoints, by skill --


def _grey_reading(skill: str) -> list[str]:
    state = make_controller(seed=4).state
    selected = KnowledgeBase().select(
        KnowledgeContext(state, "p1", engine="v3", skill=skill), limit=32
    )
    return [d.body for d in selected if d.metadata["id"].startswith("grey-reading-")]


def test_each_skill_reads_exactly_one_grey_reading_doctrine_of_its_depth():
    expert, average, novice = (_grey_reading(s) for s in ("expert", "average", "novice"))
    assert len(expert) == len(average) == len(novice) == 1
    expert_body, average_body, novice_body = expert[0], average[0], novice[0]
    # Convention 4 for everyone.
    for body in (expert_body, average_body, novice_body):
        assert "村目線のいい発言" in body and "信用されやすい" in body
    # Conventions 6 and 7: content, not volume; hard to hang; a divination target.
    for body in (expert_body, average_body):
        assert "量ではなく中身" in body
        assert "おかしな動きがない限り" in body
        assert "占い先になりやすい" in body
    assert "占い先" not in novice_body and "量ではなく中身" not in novice_body
    # Only the expert hears the voluble-without-content case and the split.
    assert "よく話しても中身がなければ当たらず" in expert_body
    assert "薄い所は吊りで、強いが未確認の所は占いで見る" in expert_body
    assert "よく話しても" not in average_body
    # Viewpoints: more for the expert, some for the average seat, none for the novice.
    assert expert_body.count("か。") > average_body.count("か。") > 0
    assert "見るところ" not in novice_body


def test_the_tiers_never_name_the_seat_s_skill_and_reach_no_other_engine():
    for doctrine in KnowledgeBase().doctrines:
        if not doctrine.metadata["id"].startswith("grey-reading-"):
            continue
        for word in ("初心者", "熟練", "普通の", "あなたは"):
            assert word not in doctrine.body, (doctrine.metadata["id"], word)
    state = make_controller(seed=4).state
    for engine in ("legacy", "v2"):
        ids = {
            d.metadata["id"]
            for d in KnowledgeBase().select(KnowledgeContext(state, "p1", engine=engine))
        }
        assert not any(i.startswith("grey-reading-") for i in ids)


def test_a_doctrine_can_name_the_skills_it_is_written_for(tmp_path: Path):
    (tmp_path / "deep.md").write_text("---\nid: deep\nskills: expert\n---\ndeep", "utf-8")
    (tmp_path / "all.md").write_text("---\nid: all\n---\nall", "utf-8")
    state = make_controller(seed=4).state
    knowledge = KnowledgeBase(tmp_path)

    def ids(skill: str) -> list[str]:
        return [
            d.metadata["id"] for d in knowledge.select(KnowledgeContext(state, "p1", skill=skill))
        ]

    assert ids("expert") == ["all", "deep"]
    assert ids("novice") == ["all"]
    # A seat with no skill on record reads as expert, as the runtime's default.
    assert ids("") == ["all", "deep"]


def test_the_prompt_carries_the_seat_s_own_tier():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _builder(state, skills={"p1": "novice", "p2": "expert"})

    novice = builder.build_discussion_context(state, "p1", "initial_view", board_memo="M")[1]
    expert = builder.build_discussion_context(state, "p2", "initial_view", board_memo="M")[1]

    assert "【灰の見方(卓の傾向)】" in novice[0].content
    assert "量ではなく中身" not in novice[0].content
    assert "量ではなく中身" in expert[0].content


def test_the_coordinator_hands_the_dealt_skills_to_the_prompts():
    controller = _all_ai_morning(seed=1)
    coordinator = _v3(controller, _Table())

    skills = coordinator._context._skills  # type: ignore[attr-defined]
    assert skills == {pid: s.value for pid, s in coordinator.reasoning.skills.items()}  # type: ignore[union-attr]


# -- X20: the seer's night prompt shows how the table placed the grey --


def _night_after_day_one() -> tuple[GameState, ContextBuilder, str]:
    controller = _all_ai_morning(seed=1)
    state = controller.state
    builder = _builder(state)
    turns = [
        # サクラ places リク white and doubts ソウタ in one line.
        ("p7", "リクは村目線で白く見る。ソウタは様子見で気になる。", ["p10"], ["p4"], "p4"),
        # ユイ only asks リク something: no stance on リク.
        ("p3", "リク、誰が怪しい？", [], ["p10"], "p10"),
        # ツムギ doubts ソウタ.
        ("p11", "ソウタに入れるかな。", [], ["p4"], "p4"),
    ]
    for number, (seat, text, trusted, suspects, target) in enumerate(turns, start=1):
        builder.set_reasoning_memo(
            seat,
            {"trusted": trusted, "suspects": suspects, "execution_target": target},
            day=1,
            stage="initial_view",
        )
        _post(state, number, seat, text)
        builder.record_key_point(1, f"m{number}", seat, "", stage="initial_view")
    state.vote_records += [
        VoteRecord(voter_id="p7", target_id="p4", day=1, round=1),
        VoteRecord(voter_id="p11", target_id="p4", day=1, round=1),
    ]
    state.phase = Phase.NIGHT
    seer = state.players_by_role(RoleName.SEER)[0].player_id
    return state, builder, seer


def _night_user(builder: ContextBuilder, state: GameState, seat: str, action: str) -> str:
    _system, messages = builder.build_night_action_context(
        state, seat, action, list(state.alive_ids()), board_memo="【盤面メモ】"
    )
    return messages[0].content


def test_the_seer_sees_who_the_table_placed_white_and_who_it_doubted():
    state, builder, seer = _night_after_day_one()

    user = _night_user(builder, state, seer, "divine")

    block = user.split("【1日目の卓での置かれ方】", 1)[1].split("\n\n", 1)[0]
    rows = {line.split("（")[0].removeprefix("- "): line for line in block.splitlines()[1:]}
    assert "白置き: サクラ" in rows["リク"]
    # ユイ only asked リク: no doubt counted.
    assert "疑い: なし" in rows["リク"]
    assert "疑い: サクラ、ツムギ" in rows["ソウタ"]
    assert "票2" in rows["ソウタ"]
    assert "発言1回" in rows["サクラ"]
    # The seer itself and the claimants are not listed.
    assert "ケント" not in rows


def test_only_the_divination_prompt_carries_the_table_picture():
    state, builder, _seer = _night_after_day_one()
    hunter = state.players_by_role(RoleName.HUNTER)[0].player_id

    assert "卓での置かれ方" not in _night_user(builder, state, hunter, "guard")
