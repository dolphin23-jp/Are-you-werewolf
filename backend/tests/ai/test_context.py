"""Prompt-assembly regressions.

Every machine-readable answer the model gives is a player_id, but the parts
of the prompt it reasons from -- the chat log, its wolf allies, its freemason
partner -- were written purely with names. Nothing tied the two together, so
the model was effectively voting blind. These tests keep the roster mapping
in place.
"""

from __future__ import annotations

import re

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import PERSONALITIES, assign_personalities
from app.ai.schemas import ChatDiscussionOutput, PlanNote
from app.ai.strategy import StrategyAnalyzer, render_board_analysis
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import (
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


def test_vote_prompt_requires_independent_evidence_and_countercase():
    controller = make_controller(seed=4)
    state = controller.state
    builder = _builder(state)

    _, messages = builder.build_vote_context(state, "p1", ["p2", "p3"])
    body = messages[0].content

    assert "decisive_evidence" in body
    assert "countercase" in body
    assert "多数派、共有指定、他者への同意だけを投票理由" in body
    assert "自吊り拒否、COの短さ、初日占い理由の薄さ" in body


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


def test_a_served_question_is_not_put_to_the_target_again():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    first = controller.chat("p1", "理由を教えてください", "public")
    second = controller.chat("p3", "どうですか", "public")
    state.pending_questions["p2"] = [
        PendingQuestion("p1", "p2", "投票の理由は何ですか?", first, state.day, served=True),
        PendingQuestion("p3", "p2", "狐候補は誰ですか?", second, state.day),
    ]
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p2")

    assert "投票の理由は何ですか?" not in messages[0].content
    assert "狐候補は誰ですか?" in messages[0].content
    # The model is told how to close a question and that it will not be asked twice.
    assert "reply_to" in messages[0].content
    assert "繰り返さないでください" in messages[0].content


def test_with_every_question_served_the_prompt_says_there_are_none():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    source = controller.chat("p1", "理由を教えてください", "public")
    state.pending_questions["p2"] = [
        PendingQuestion("p1", "p2", "投票の理由は何ですか?", source, state.day, served=True)
    ]
    builder = _builder(state)

    _system, messages = builder.build_discussion_context(state, "p2")

    assert "投票の理由は何ですか?" not in messages[0].content
    assert "【あなたへの未回答の質問】(ありません)" in messages[0].content


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
    state.day = 1
    controller.co(fake_seer, RoleName.SEER.value)

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


# (The minority-review stage and its text were removed after the owner's
# review of seed 1: a voice against the majority should emerge from the
# table, not be appointed. tests/ai/test_day_turn_schedule.py pins that no
# such turn is scheduled.)


# -- v3: the chat register --


def _chat_builder(state) -> ContextBuilder:  # type: ignore[no-untyped-def]
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


def test_chat_discussion_prompt_hands_over_the_memo_and_asks_for_names_only():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 1
    controller.chat("p2", "おはよう。", "public")
    builder = _chat_builder(state)

    system, messages = builder.build_discussion_context(
        state,
        "p1",
        "initial_view",
        board_memo="【盤面メモ】1日目 / あなた: Player1\n- 生存(16人): ...",
        required_lines=["占い師CO"],
    )
    body = messages[0].content

    assert "【盤面メモ】" in body
    assert "【この発言で必ず言うこと】\n- 占い師CO" in body
    assert "【名簿(JSONのplayer_id用)】" in body
    assert "IDは絶対に書かない" in system
    assert "17人村(17A)" in system
    # The brief's wording and the structured contract's long form are gone.
    assert "第一処刑候補" not in body
    assert "role_hypotheses" not in body
    assert "reassessments" not in body
    # The log is written in names, not `名前(pN)`.
    assert "] Player2: おはよう。" in body
    assert "Player2(p2)" not in body


def test_chat_prompt_carries_the_17a_doctrine_and_the_chat_register_lines():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _chat_builder(state)

    system, messages = builder.build_discussion_context(
        state, "p1", "initial_view", board_memo="【盤面メモ】"
    )

    assert "【17A村の進行の常識】" in messages[0].content
    assert "口調の例" in system


def test_structured_prompt_is_unchanged_by_the_chat_doctrine():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _builder(state)

    _, messages = builder.build_discussion_context(state, "p1", "initial_view")

    assert "【17A村の進行の常識】" not in messages[0].content
    assert "Player2(p2)" in messages[0].content or "(p" in messages[0].content


def test_chat_vote_prompt_states_what_the_seat_said_and_uses_names():
    state = make_controller(seed=4).state
    state.day = 2
    builder = _chat_builder(state)

    _, messages = builder.build_vote_context(
        state, "p1", ["p2", "p3"], board_memo="【盤面メモ】", stated_target="p2"
    )
    body = messages[0].content

    # The note's target, labelled as the note, and flagged: p1 never named
    # Player2 at the table today.
    assert "【今日のメモの吊り先】Player2(今日の卓ではまだこの名前を出していない)" in body
    assert "【投票候補】Player2（グレー）、Player3（グレー）" in body
    assert "Player2(p2)" not in body
    assert "卓で名指しした吊り先と違う相手に入れるなら" in body


def test_chat_night_prompt_names_the_action_in_japanese_and_carries_the_roster():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _chat_builder(state)

    _, messages = builder.build_night_action_context(
        state, "p1", "divine", ["p2", "p3"], board_memo="【盤面メモ】"
    )
    body = messages[0].content

    assert "【夜行動: 占い(divine)】候補: Player2、Player3" in body
    assert "【名簿(JSONのplayer_id用)】" in body


# -- v3: nothing asks for the model's thinking --
#
# A prompt or schema that asks the model to write out its thinking can be
# refused as reasoning extraction, and that refusal is not retried. The v3
# prompts speak of a plan note instead; v2 and legacy keep their wording.

_THINKING_WORDS = re.compile(r"思考|thought|reasoning", re.IGNORECASE)


def _every_v3_prompt(state, builder) -> list[str]:  # type: ignore[no-untyped-def]
    stages = [
        "initial_view",
        "immediate",
        "reaction",
        "rebuttal_or_reassessment",
        "freemason_confirmation",
        "consensus_summary",
        "human_followup",
        "open_floor",
        "grey_view",
        "facilitator_followup",
        "provisional_designation",
        "summary_answer",
    ]
    texts: list[str] = []
    for player_id in state.players:
        # A memo as the coordinator stores it: ReasoningMemo's own field names.
        builder.set_reasoning_memo(
            player_id,
            {
                "execution_target": "p2",
                "overall_thought": "方針",
                "private_team_thought": "狙い",
                "role_hypotheses": ["p1真"],
            },
        )
        built = [
            builder.build_discussion_context(state, player_id, stage, board_memo="【盤面メモ】")
            for stage in stages
        ]
        built.append(
            builder.build_vote_context(state, player_id, ["p2", "p3"], board_memo="【盤面メモ】")
        )
        built += [
            builder.build_night_action_context(
                state, player_id, action, ["p2", "p3"], board_memo="【盤面メモ】"
            )
            for action in ("divine", "guard", "attack")
        ]
        built.append(builder.build_wolf_chat_context(state, player_id, board_memo="【盤面メモ】"))
        built.append(
            builder.build_freemason_chat_context(state, player_id, board_memo="【盤面メモ】")
        )
        built.append(builder.build_morning_intent_context(state, player_id))
        built.append(builder.build_summary_context(state, player_id))
        for system, messages in built:
            texts.append(system)
            texts.extend(message.content for message in messages)
    return texts


def test_no_v3_prompt_asks_for_thinking_for_any_seat_or_personality():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 2
    controller.chat("p2", "おはよう。", "public")
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    builder = ContextBuilder(
        # Every preset, so a personality trait that says 思考 is caught too.
        personalities={
            pid: PERSONALITIES[index % len(PERSONALITIES)]
            for index, pid in enumerate(state.players)
        },
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="fake_seer",
            pattern_label="偽占い",
            fake_role_by_player={wolf_ids[0]: RoleName.SEER},
            lurking_player_ids=wolf_ids[1:],
        ),
        madman_fake_role=RoleName.SEER,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine="v3",
    )

    offending = [
        text[max(0, match.start() - 30) : match.end() + 10]
        for text in _every_v3_prompt(state, builder)
        for match in _THINKING_WORDS.finditer(text)
    ]

    assert offending == []


def test_chat_discussion_contract_names_the_wire_keys():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _chat_builder(state)

    system, messages = builder.build_discussion_context(
        state, "p1", "initial_view", board_memo="【盤面メモ】"
    )
    body = messages[0].content

    assert "自分の見立てと方針はplan_noteに短く書く" in system
    assert '"plan_note": {"plan": "非公開の方針メモ。' in body
    # The note is listed before the message, the order it is written in.
    assert body.index('"plan_note"') < body.index('"public_message"')
    assert '"plan": "非公開の方針メモ。' in body
    assert '"team_plan": "非公開。人狼・狂人は陣営としての狙いと次の一手を短く' in body
    assert '"needs_another_statement": このあともう一言続けたいならtrue' in body
    # Every key the contract names is a key of the schema actually sent.
    contract = body.split("次のJSONだけを返してください", 1)[1]
    named = set(re.findall(r'"([a-z_]+)":', contract))
    sent = set(ChatDiscussionOutput.model_fields) | set(PlanNote.model_fields)
    sent |= {"result_type", "target_id", "is_werewolf", "referenced_day", "question"}
    assert named <= sent


def test_chat_previous_memo_is_shown_under_the_wire_names():
    state = make_controller(seed=4).state
    state.day = 1
    builder = _chat_builder(state)
    builder.set_reasoning_memo(
        "p1",
        {
            "trusted_seer": "p3",
            "suspects": ["p2"],
            "trusted": [],
            "execution_target": "p2",
            "overall_thought": "継続方針",
            "role_hypotheses": ["p3真なら"],
            "fox_candidates": [],
            "private_team_thought": "",
        },
    )

    _, messages = builder.build_discussion_context(
        state, "p1", "initial_view", board_memo="【盤面メモ】"
    )
    body = messages[0].content
    layer = body.split("【前回の方針メモ】\n", 1)[1].split("\n\n", 1)[0]

    # Wire names, names for ids (with each seat's status), and no empty team note.
    assert layer.splitlines() == [
        "- execution_target: Player2［グレー］",
        "- suspects: Player2［グレー］",
        "- trusted: なし",
        "- trusted_seer: Player3［グレー］",
        "- fox_candidates: なし",
        "- plan: 継続方針",
    ]
    assert "前回の非公開思考メモ" not in body


def test_chat_previous_memo_placeholder_uses_the_new_title():
    state = make_controller(seed=4).state
    builder = _chat_builder(state)

    _, messages = builder.build_discussion_context(state, "p1", "initial_view", board_memo="")

    assert "【前回の方針メモ】(まだありません)" in messages[0].content


def test_structured_previous_memo_keeps_its_title_and_keys():
    state = make_controller(seed=4).state
    builder = _builder(state)
    builder.set_reasoning_memo("p1", {"overall_thought": "継続", "role_hypotheses": ["x"]})

    _, messages = builder.build_discussion_context(state, "p1")

    assert '【前回の非公開思考メモ】\n{"overall_thought": "継続", "role_hypotheses": ["x"]}' in (
        messages[0].content
    )


def test_chat_personality_section_avoids_the_thinking_label_but_structured_keeps_it():
    for personality in PERSONALITIES:
        chat = personality.to_prompt_section(chat=True)
        assert "思考" not in chat
        # The thinking habit is not in the v3 persona at all (tone and amount only).
        assert "- 考え方の癖: " not in chat
        assert f"- 思考スタイル: {personality.thinking_style}" in personality.to_prompt_section()


# -- v3: night and private chat read the day's public discussion --


def _played_into_night():  # type: ignore[no-untyped-def]
    """A real day 1, through the engine's own transitions, ending at night."""
    controller = make_controller(seed=4)
    state = controller.state
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.chat("p2", "Player5の票の理由が気になる。", "public")
    controller.chat("p5", "昨日は情報がなかっただけ。", "public")
    controller.end_discussion()
    for voter in state.alive_ids():
        controller.vote(voter, "p5" if voter != "p5" else "p2")
    controller.resolve_votes()
    controller.start_night()
    return controller, state


def test_during_the_night_the_day_is_still_the_discussion_that_just_ended():
    # The filter the night layer relies on: the engine only advances the day
    # when the next discussion starts.
    _controller, state = _played_into_night()

    assert state.phase == Phase.NIGHT
    assert state.day == 1
    assert {m.day for m in state.chat_log if m.channel.value == "public"} == {1}


def _night_seat(state, role: RoleName) -> str:  # type: ignore[no-untyped-def]
    return next(p.player_id for p in state.players_by_role(role) if p.alive)


def _in_order(body: str, *headings: str) -> bool:
    positions = [body.index(heading) for heading in headings]
    return positions == sorted(positions)


def test_chat_night_action_prompt_carries_the_days_public_discussion():
    _controller, state = _played_into_night()
    builder = _chat_builder(state)
    seer = _night_seat(state, RoleName.SEER)

    _, messages = builder.build_night_action_context(
        state, seer, "divine", ["p2", "p3"], board_memo="【盤面メモ】"
    )
    body = messages[0].content

    assert "【今日の公開議論のログ】\n" in body
    assert "] Player2: Player5の票の理由が気になる。" in body
    assert "] Player5: 昨日は情報がなかっただけ。" in body
    # Placed right after the past-day summaries, before the seat's own memo.
    assert _in_order(body, "【過去日の要約】", "【今日の公開議論のログ】", "【前回の方針メモ】")


def test_chat_private_chat_prompts_carry_summaries_and_the_days_public_discussion():
    controller, state = _played_into_night()
    wolf = _night_seat(state, RoleName.WEREWOLF)
    controller.chat(wolf, "今夜は占いを噛もう。", "wolf")
    builder = _chat_builder(state)

    memo = "【盤面メモ】"
    _, wolf_messages = builder.build_wolf_chat_context(state, wolf, board_memo=memo)
    mason = _night_seat(state, RoleName.FREEMASON)
    _, mason_messages = builder.build_freemason_chat_context(state, mason, board_memo=memo)

    for body in (wolf_messages[0].content, mason_messages[0].content):
        assert "] Player2: Player5の票の理由が気になる。" in body
        assert _in_order(
            body, "【過去日の要約】", "【今日の公開議論のログ】", "【過去を含む内輪ログ】"
        )
    assert "今夜は占いを噛もう。" in wolf_messages[0].content
    assert "今夜は占いを噛もう。" not in mason_messages[0].content


def test_structured_night_and_private_prompts_are_unchanged_by_the_public_layer():
    controller, state = _played_into_night()
    builder = _builder(state)
    seer = _night_seat(state, RoleName.SEER)
    wolf = _night_seat(state, RoleName.WEREWOLF)

    _, night = builder.build_night_action_context(state, seer, "divine", ["p2", "p3"])
    _, private = builder.build_wolf_chat_context(state, wolf)

    for body in (night[0].content, private[0].content):
        assert "【今日の公開議論のログ】" not in body
        assert "Player5の票の理由が気になる。" not in body
    assert "【過去日の要約】" not in private[0].content
