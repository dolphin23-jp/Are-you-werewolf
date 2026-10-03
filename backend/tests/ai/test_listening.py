"""Listening at the table: the v3 prompt hands a seat what happened since it spoke.

A person at a chat table reacts to what was said to them, what was said about the
player they were pushing, and what newly came out. Left to find those in a long
log the model mostly restated its own position -- in the first v3 game 6 of 64
follow-up turns moved the stated candidate of their own accord, and only two read
as a reaction to anyone. These tests pin the pieces that put those three things
in front of the model.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import OPENNESS_TEXT, PERSONALITIES, assign_personalities
from app.engine.roles import RoleName
from app.engine.state import PendingQuestion
from tests.ai.reasoning.fixtures import declare_co, publish_result
from tests.conftest import make_controller


def _v3_builder(state) -> ContextBuilder:  # type: ignore[no-untyped-def]
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
    )


def _table():  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    controller.state.day = 2
    return controller, controller.state, _v3_builder(controller.state)


# -- the digest --


def test_the_digest_names_what_was_said_to_the_seat_and_about_its_candidate():
    controller, state, builder = _table()
    mine = controller.chat("p1", "Player5が怪しい", "public")
    builder.set_reasoning_memo("p1", {"execution_target": "p5"})
    to_me = controller.chat("p2", "Player1、根拠は票だけ？", "public", mine)
    about_target = controller.chat("p3", "Player5は白をもらってるよ", "public")
    controller.chat("p4", "今日は静かだね", "public")

    digest = builder._layer_since_last_turn(state, "p1")

    assert "あなたが推していた吊り先: Player5" in digest
    assert f"[{to_me}] Player2: Player1、根拠は票だけ？" in digest
    assert f"[{about_target}] Player3: Player5は白をもらってるよ" in digest
    assert "今日は静かだね" not in digest
    assert "筋が通っていれば認めて変え" in digest
    # The table is spoken to in names; an id in the digest would be copied into speech.
    assert "(p" not in digest


def test_only_what_came_after_the_seats_own_last_message_counts_as_since():
    controller, state, builder = _table()
    controller.chat("p2", "Player1、前の質問です", "public")
    controller.chat("p1", "答えました", "public")
    builder.set_reasoning_memo("p1", {"execution_target": "p5"})
    controller.chat("p3", "Player1、あとの質問です", "public")

    digest = builder._layer_since_last_turn(state, "p1")

    assert "あとの質問です" in digest
    assert "前の質問です" not in digest


def test_a_candidate_who_has_died_is_not_pushed_again():
    controller, state, builder = _table()
    controller.chat("p1", "Player5が怪しい", "public")
    builder.set_reasoning_memo("p1", {"execution_target": "p5"})
    state.players["p5"].alive = False
    controller.chat("p2", "Player5について話すと…", "public")

    digest = builder._layer_since_last_turn(state, "p1")

    assert "Player5(すでに死亡)" in digest
    assert "Player5について出た発言" not in digest


def test_new_claims_and_results_since_the_seat_spoke_are_listed():
    controller, state, builder = _table()
    controller.chat("p1", "様子を見ます", "public")
    message = controller.chat("p3", "占いCO。Player6は黒でした。", "public")
    declare_co(state, "p3", RoleName.SEER, day=2, source_message_id=message)
    publish_result(state, "p3", "seer", "p6", True, day=2, source_message_id=message)

    digest = builder._layer_since_last_turn(state, "p1")

    assert "Player3が占い師CO" in digest
    assert "Player3がPlayer6=黒と結果を出した" in digest


def test_a_question_already_shown_in_its_own_layer_is_not_repeated_in_the_digest():
    controller, state, builder = _table()
    controller.chat("p1", "様子を見ます", "public")
    asked = controller.chat("p2", "Player1、理由は？", "public")
    state.pending_questions["p1"] = [PendingQuestion("p2", "p1", "理由は？", asked, 2)]

    digest = builder._layer_since_last_turn(state, "p1")

    assert f"[{asked}]" not in digest
    # Nothing else was said to the seat, so the digest has nothing to add.
    assert "あなた宛て・あなたの推し先に触れた発言・新しい結果は、まだありません。" in digest


def test_the_digest_says_so_when_nothing_new_has_come_out():
    controller, state, builder = _table()
    controller.chat("p1", "Player5が怪しい", "public")
    builder.set_reasoning_memo("p1", {"execution_target": "p5"})
    controller.chat("p4", "今日は静かだね", "public")

    digest = builder._layer_since_last_turn(state, "p1")

    assert "まだありません。" in digest
    assert "考えを変えるか決めてから話す" not in digest


def test_the_first_speaker_of_the_day_is_told_nobody_has_spoken():
    _controller, state, builder = _table()

    digest = builder._layer_since_last_turn(state, "p1")

    assert "あなたが推していた吊り先: まだありません" in digest
    assert "まだ誰も話していません" in digest


def test_a_long_message_is_clipped_in_the_digest():
    controller, state, builder = _table()
    controller.chat("p1", "様子を見ます", "public")
    controller.chat("p2", "Player1、" + "あ" * 200, "public")

    digest = builder._layer_since_last_turn(state, "p1")

    line = next(item for item in digest.splitlines() if "Player2:" in item)
    assert line.endswith("…")
    assert len(line) < 120


def test_the_digest_sits_in_the_v3_discussion_prompt():
    controller, state, builder = _table()
    controller.chat("p1", "様子を見ます", "public")

    _system, messages = builder.build_discussion_context(
        state, "p1", "initial", board_memo="【盤面メモ】", required_lines=[]
    )

    assert "【前回の発言のあとに出たこと】" in messages[0].content


# -- the doctrine and the persona --


def test_the_v3_system_prompt_teaches_listening_and_the_older_prompts_are_untouched():
    controller, state, builder = _table()
    v3 = builder._layer_a_system(state, "p1")

    assert "【聞き入れ方】" in v3
    assert "認めたなら疑い先・吊り先も実際に変え" in v3
    assert "強く言われただけで折れない" in v3

    legacy = ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment("all_lurk", "全潜伏", {}, []),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set()),
    )._layer_a_system(state, "p1")
    assert "聞き入れ方" not in legacy
    assert "意見の聞き方" not in legacy


def test_every_persona_has_an_openness_and_all_three_kinds_exist():
    kinds = {personality.openness for personality in PERSONALITIES}

    assert kinds == set(OPENNESS_TEXT)
    for kind in OPENNESS_TEXT:
        assert sum(p.openness == kind for p in PERSONALITIES) >= 3


def test_the_chat_persona_states_how_it_listens_and_the_structured_one_does_not():
    for personality in PERSONALITIES:
        chat = personality.to_prompt_section(chat=True)
        assert OPENNESS_TEXT[personality.openness] in chat
        assert "意見の聞き方" not in personality.to_prompt_section()
