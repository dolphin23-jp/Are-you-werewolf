"""Prompt-assembly regressions.

Every machine-readable answer the model gives is a player_id, but the parts
of the prompt it reasons from -- the chat log, its wolf allies, its freemason
partner -- were written purely with names. Nothing tied the two together, so
the model was effectively voting blind. These tests keep the roster mapping
in place.
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.ai.strategy import StrategyAnalyzer, render_board_analysis
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import (
    CoDeclaration,
    DivineRecord,
    MediumRecord,
    PendingQuestion,
    VoteRecord,
)
from app.engine.vote import VoteManager
from tests.conftest import make_controller


def _builder(state, observer_player_ids: set[str] | None = None) -> ContextBuilder:
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
        observer_player_ids=observer_player_ids,
    )


def test_vote_candidates_carry_both_name_and_id():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)

    _, messages = builder.build_vote_context(state, "p1", ["p2", "p3"])
    body = messages[0].content

    assert "Player2(p2)" in body
    assert "Player3(p3)" in body


def test_wolf_allies_carry_ids_so_a_wolf_can_avoid_voting_for_a_partner():
    controller = make_controller(seed=4)
    state = controller.state
    wolves = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    builder = _builder(state)

    system, _ = builder.build_vote_context(state, wolves[0], state.alive_ids()[1:])

    for ally in wolves[1:]:
        assert f"{state.players[ally].name}({ally})" in system


def test_board_analysis_lists_the_living_roster():
    controller = make_controller(seed=4)
    state = controller.state
    rendered = render_board_analysis(StrategyAnalyzer().analyze(state), state)

    for pid in state.alive_ids():
        assert f"{state.players[pid].name}({pid})" in rendered


def test_runoff_prompt_explains_why_the_field_shrank():
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.VOTING
    ids = state.alive_ids()

    # Force an exact tie so the manager narrows the field.
    state.pending_votes.update({ids[2]: ids[0], ids[3]: ids[1]})
    VoteManager().tally(state)
    assert state.runoff_candidates

    builder = _builder(state)
    _, messages = builder.build_vote_context(state, ids[4], state.votable_ids(ids[4]))
    body = messages[0].content

    assert "決選投票" in body
    assert "同数" in body


def test_ordinary_round_does_not_mention_a_runoff():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)

    _, messages = builder.build_vote_context(state, "p1", ["p2", "p3"])

    assert "決選投票" not in messages[0].content


def test_discussion_prompt_localizes_stage_and_includes_vote_history():
    controller = make_controller(seed=4)
    state = controller.state
    state.vote_records.append(VoteRecord(voter_id="p1", target_id="p2", day=1, round=1))
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p3", "initial_view")
    body = messages[0].content

    assert "【投票履歴】" in body
    assert "Player1(p1) → Player2(p2)" in body
    assert "【議論段階】初回意見" in body
    assert "initial_view" not in body
    assert "200文字" not in (_system + body)


def test_discussion_prompt_uses_role_neutral_evidence_standards():
    controller = make_controller(seed=4)
    builder = _builder(controller.state)

    system, _messages = builder.build_discussion_context(controller.state, "p3")

    assert "初日・0日目の占い先は発言情報がないランダム選択でも自然" in system
    assert "CO文が短いこと、丁寧さや強い口調そのものを偽要素" in system
    assert "自吊りを拒むこと自体を黒要素にせず" in system
    assert "黒一致だけで占い師の真は確定しません" in system


def test_vote_history_is_limited_to_the_most_recent_two_days():
    controller = make_controller(seed=4)
    state = controller.state
    state.vote_records.extend(
        VoteRecord(voter_id="p1", target_id="p2", day=day, round=1)
        for day in (1, 2, 3)
    )
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p3")
    body = messages[0].content

    assert "1日目R1" not in body
    assert "2日目R1" in body
    assert "3日目R1" in body


def test_reply_quote_round_trip_is_rendered_in_ai_log():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    first = controller.chat("p1", "元の発言", "public")
    controller.chat("p2", "回答です", "public", reply_to=first, quote="元の発言")
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p3")

    assert "[m1] Player1(p1): 元の発言" in messages[0].content
    assert "[m2 →m1] Player2(p2): 回答です" in messages[0].content


def test_pending_question_enters_target_prompt_and_reply_resolves_it():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    source = controller.chat("p1", "理由を教えてください", "public")
    state.pending_questions["p2"] = [
        PendingQuestion("p1", "p2", "理由は何ですか?", source, state.day)
    ]
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p2")
    assert "理由は何ですか?" in messages[0].content

    controller.chat("p2", "この理由です", "public", reply_to=source)
    assert state.pending_questions["p2"] == []


def test_existing_key_points_are_rendered_with_message_ids():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    builder = _builder(state)
    builder.record_key_point(1, "m7", "p1", "初日の投票先を比較する")

    _system, messages = builder.build_discussion_context(state, "p2")

    assert "【すでに卓に出ている論点】" in messages[0].content
    assert "[m7] Player1(p1): 初日の投票先を比較する" in messages[0].content
    assert "agrees_with" in messages[0].content


def test_role_owner_receives_private_results_but_other_player_does_not():
    controller = make_controller(seed=4)
    state = controller.state
    seer = state.players_by_role(RoleName.SEER)[0]
    medium = state.players_by_role(RoleName.MEDIUM)[0]
    role_owner_ids = {seer.player_id, medium.player_id}
    target = next(p for p in state.alive_players() if p.player_id not in role_owner_ids)
    state.divine_records.append(DivineRecord(seer.player_id, target.player_id, 1, True))
    state.medium_records.append(MediumRecord(medium.player_id, target.player_id, 1, False))
    builder = _builder(state)

    seer_system, _ = builder.build_discussion_context(state, seer.player_id)
    medium_system, _ = builder.build_discussion_context(state, medium.player_id)
    other_system, _ = builder.build_discussion_context(state, target.player_id)

    assert "占い結果" in seer_system and "=人狼" in seer_system
    assert "霊媒結果" in medium_system and "=人狼ではない" in medium_system
    assert "占い結果" not in other_system
    assert "霊媒結果" not in other_system


def test_private_divine_target_stays_in_public_gray_list():
    controller = make_controller(seed=4)
    state = controller.state
    seer = state.players_by_role(RoleName.SEER)[0]
    target = next(p for p in state.alive_players() if p.player_id != seer.player_id)
    state.divine_records.append(DivineRecord(seer.player_id, target.player_id, 1, False))

    analysis = StrategyAnalyzer().analyze(state)

    assert target.player_id in analysis.gray_player_ids


def test_prompt_anchors_self_identity_and_marks_observer():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state, {"p0"})

    system, messages = builder.build_discussion_context(state, "p1")

    assert "Player1(p1)」はあなた自身" in system
    assert "非参戦席" in messages[0].content
    assert "Player0(p0)" in messages[0].content


def test_dead_private_ally_is_labeled_dead_in_role_context():
    controller = make_controller(seed=4)
    state = controller.state
    wolves = state.players_by_role(RoleName.WEREWOLF)
    wolves[1].alive = False
    wolves[1].death_day = 2
    builder = _builder(state)

    system, _ = builder.build_discussion_context(state, wolves[0].player_id)

    assert f"{wolves[1].name}({wolves[1].player_id})[2日目死亡済み]" in system


def test_previous_private_reasoning_memo_is_reused():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)
    builder.set_reasoning_memo("p1", {"execution_target": "p2", "overall_thought": "継続思考"})

    _, messages = builder.build_discussion_context(state, "p1")

    assert "前回の非公開思考メモ" in messages[0].content
    assert "継続思考" in messages[0].content


def _fake_seer_builder(state, fake_seer_id: str) -> ContextBuilder:
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="fake_seer",
            pattern_label="偽占い",
            fake_role_by_player={fake_seer_id: RoleName.SEER},
            lurking_player_ids=[w for w in wolf_ids if w != fake_seer_id],
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
    )


def test_vote_prompt_carries_the_faction_doctrine_the_discussion_gets():
    # The vote is where a bluff costs something. Without doctrine here a fake seer
    # argues its claim all day and then votes like a plain villager.
    controller = make_controller(seed=4)
    state = controller.state
    fake_seer = state.players_by_role(RoleName.WEREWOLF)[0].player_id
    builder = _fake_seer_builder(state, fake_seer)
    state.co_declarations.append(
        CoDeclaration(player_id=fake_seer, claimed_role=RoleName.SEER, day=1)
    )

    # A phrase unique to fake-seer-objectives.md; "偽占い" alone also appears in the
    # perspective doctrine that everyone receives.
    doctrine = "【偽占いとしての目的】"
    _, messages = builder.build_vote_context(state, fake_seer, ["p2", "p3"])
    assert doctrine in "\n".join(m.content for m in messages)

    villager = state.players_by_role(RoleName.VILLAGER)[0].player_id
    _, plain = builder.build_vote_context(state, villager, ["p2", "p3"])
    assert doctrine not in "\n".join(m.content for m in plain)


def test_discussion_prompt_pushes_replies_and_verbatim_quotes():
    # Message ids are rendered in the log, but the log alone does not make a model
    # use them. The instruction has to name the fields and demand a verbatim excerpt.
    controller = make_controller(seed=4)
    state = controller.state
    controller.chat("p1", "おはよう", "public")
    builder = _builder(state)

    _, messages = builder.build_discussion_context(state, "p2", "initial_view")
    prompt = "\n".join(m.content for m in messages)

    assert "[m1]" in prompt, "the log must expose ids for reply_to to be usable"
    assert "reply_to" in prompt
    assert "quote" in prompt
    assert "原文" in prompt


def test_reaction_stage_still_asks_for_an_anchored_target():
    controller = make_controller(seed=4)
    state = controller.state
    controller.chat("p1", "おはよう", "public")
    builder = _builder(state)

    _, messages = builder.build_discussion_context(state, "p2", "reaction")
    prompt = "\n".join(m.content for m in messages)
    assert "reply_to" in prompt and "quote" in prompt


def test_minority_review_requires_countercase_and_alternative():
    controller = make_controller(seed=4)
    builder = _builder(controller.state)

    _, messages = builder.build_discussion_context(
        controller.state, "p2", "minority_review:p0"
    )
    prompt = "\n".join(message.content for message in messages)

    assert "Player0(p0)へ集中" in prompt
    assert "反対仮説" in prompt
    assert "別の処刑候補" in prompt
