"""The analyzers are what turn a transcript into the reviewable metrics, so
each check needs both a case that fires and a case that must stay silent --
a detector that flags everything is as useless as one that flags nothing."""

from __future__ import annotations

from app.eval.analyzers import analyze
from app.eval.transcript import GameTranscript, Utterance

NAMES = {"p0": "あなた", "p1": "アカリ", "p2": "ハルト", "p3": "ユイ"}


def _transcript(**overrides) -> GameTranscript:
    base = GameTranscript(
        seed=1,
        names=dict(NAMES),
        roles={"p0": "villager", "p1": "seer", "p2": "werewolf", "p3": "werewolf"},
        teams={"p0": "village", "p1": "village", "p2": "werewolf", "p3": "werewolf"},
        personalities={pid: "冷静な論客" for pid in NAMES},
        deception={
            "wolf_pattern": "alpha",
            "wolf_pattern_label": "偽占い1+潜伏1",
            "fake_role_by_player": {"p2": "seer"},
            "lurking_player_ids": ["p3"],
        },
        final_state={"death_records": [], "winner": "village"},
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _say(player_id: str, text: str, day: int = 1, **kwargs) -> Utterance:
    t = _transcript()
    return Utterance(
        day=day,
        phase="discussion",
        kind=kwargs.pop("kind", "discussion"),
        player_id=player_id,
        player_name=t.names[player_id],
        role=t.roles[player_id],
        team=t.teams[player_id],
        personality="冷静な論客",
        deception_role=None,
        text=text,
        **kwargs,
    )


def test_villager_claiming_a_divination_result_is_flagged():
    result = {"result_type": "seer", "target_id": "p3", "is_werewolf": True}
    t = _transcript(
        utterances=[_say("p0", "ユイを占った結果、人狼でした。", public_results=[result])]
    )
    assert analyze(t).count("result_claim_without_role") == 1


def test_real_seer_reporting_a_result_is_not_flagged():
    result = {"result_type": "seer", "target_id": "p3", "is_werewolf": False}
    t = _transcript(
        utterances=[
            _say(
                "p1",
                "ユイを占った結果、人狼ではありませんでした。",
                public_results=[result],
            )
        ]
    )
    assert analyze(t).count("result_claim_without_role") == 0


def test_assigned_fake_seer_reporting_a_result_is_not_flagged():
    """p2 is a wolf pre-committed to faking seer -- that is the plan working,
    not an inconsistency."""
    public_result = {"result_type": "seer", "target_id": "p3", "is_werewolf": False}
    t = _transcript(
        utterances=[
            _say(
                "p2",
                "占い師CO。ユイを占った結果、白でした。",
                public_results=[public_result],
            )
        ]
    )
    result = analyze(t)
    assert result.count("result_claim_without_role") == 0
    assert result.count("co_role_mismatch") == 0


def test_changing_claimed_role_across_days_is_flagged():
    t = _transcript(
        utterances=[
            _say("p0", "占い師COします。", day=1),
            _say("p0", "やっぱり霊媒師です。", day=2),
        ]
    )
    assert analyze(t).count("co_role_changed") == 1


def test_merely_discussing_someone_elses_co_is_not_a_claim():
    t = _transcript(
        utterances=[
            _say("p0", "占い師のCOを待ってから動いた方がいいと思います。"),
            _say("p0", "霊媒師は誰ですか。", day=2),
        ]
    )
    result = analyze(t)
    assert result.count("co_role_mismatch") == 0
    assert result.count("co_role_changed") == 0


def test_wolf_naming_a_teammate_as_a_wolf_is_flagged():
    t = _transcript(utterances=[_say("p2", "ユイは人狼だと思います。")])
    assert analyze(t).count("wolf_named_teammate_with_wolf_word") == 1


def test_wolf_suspecting_a_villager_is_not_flagged():
    t = _transcript(utterances=[_say("p2", "あなたが人狼だと思います。")])
    assert analyze(t).count("wolf_named_teammate_with_wolf_word") == 0


def test_wolf_voting_a_teammate_is_flagged():
    t = _transcript(utterances=[_say("p2", "理由", kind="vote", target="p3")])
    assert analyze(t).count("wolf_voted_teammate") == 1


def test_assigned_faker_that_never_claims_is_flagged():
    t = _transcript(utterances=[_say("p2", "特に意見はありません。")])
    result = analyze(t)
    assert result.count("assigned_faker_never_claimed") == 1
    assert result.stats["wolf_plan_execution"]["p2"]["executed"] is False


def test_lurker_that_claims_is_flagged():
    t = _transcript(utterances=[_say("p3", "占い師COします。")])
    assert analyze(t).count("lurker_broke_cover") == 1


def test_historical_reference_to_dead_player_is_not_flagged():
    t = _transcript(
        utterances=[_say("p0", "アカリの意見に賛成です。", day=3)],
        final_state={"death_records": [{"player_id": "p1", "cause": "executed", "day": 2}]},
    )
    assert analyze(t).count("treats_dead_player_as_active") == 0


def test_asking_a_dead_player_to_answer_is_flagged():
    t = _transcript(
        utterances=[
            _say(
                "p0",
                "アカリは投票理由を説明してください。",
                day=3,
                directed_question_targets=["p1"],
            )
        ],
        final_state={"death_records": [{"player_id": "p1", "cause": "executed", "day": 2}]},
    )
    assert analyze(t).count("treats_dead_player_as_active") == 1


def test_medium_result_about_dead_player_is_not_flagged():
    t = _transcript(
        utterances=[_say("p0", "アカリの霊媒結果は白でした。", day=3)],
        final_state={"death_records": [{"player_id": "p1", "cause": "executed", "day": 2}]},
    )
    assert analyze(t).count("treats_dead_player_as_active") == 0


def test_remote_black_word_does_not_turn_teammate_reference_into_accusation():
    t = _transcript(
        utterances=[_say("p2", "ユイ吊りは妥当。ただアカリの自己黒判定は破綻です。")]
    )
    assert analyze(t).count("wolf_named_teammate_with_wolf_word") == 0


def test_self_treated_as_another_player_is_flagged():
    t = _transcript(
        utterances=[
            _say(
                "p0",
                "自分自身は処刑対象にできません。",
                reasoning_memo={"suspects": [], "execution_target": "p0"},
            )
        ]
    )
    assert analyze(t).count("self_treated_as_other_player") == 1


def test_non_p0_claiming_p0_identity_is_flagged():
    t = _transcript(utterances=[_say("p1", "私がp0本人です。")])
    assert analyze(t).count("claimed_p0_identity") == 1


def test_true_seer_claiming_opposite_result_is_flagged():
    t = _transcript(
        utterances=[
            _say(
                "p1",
                "ユイを占った結果、白でした。",
                public_results=[
                    {"result_type": "seer", "target_id": "p3", "is_werewolf": False}
                ],
            )
        ],
        final_state={
            "death_records": [],
            "divine_records": [
                {"seer_id": "p1", "target_id": "p3", "day": 0, "is_werewolf": True}
            ],
            "medium_records": [],
        },
    )
    assert analyze(t).count("true_role_result_conflict") == 1


def test_true_seer_claiming_matching_result_is_not_flagged():
    t = _transcript(
        utterances=[
            _say(
                "p1",
                "ユイを占った結果、人狼でした。",
                public_results=[
                    {"result_type": "seer", "target_id": "p3", "is_werewolf": True}
                ],
            )
        ],
        final_state={
            "death_records": [],
            "divine_records": [
                {"seer_id": "p1", "target_id": "p3", "day": 0, "is_werewolf": True}
            ],
            "medium_records": [],
        },
    )
    assert analyze(t).count("true_role_result_conflict") == 0


def test_meta_phrase_leak_is_flagged():
    t = _transcript(utterances=[_say("p0", "AIとして客観的に申し上げます。")])
    assert analyze(t).count("meta_phrase_leaked") == 1


def test_speech_stats_track_fallbacks_and_repeats():
    t = _transcript(
        utterances=[
            _say("p0", "同じ発言です。"),
            _say("p0", "同じ発言です。", day=2),
            _say("p1", "様子を見ます。", used_fallback=True),
        ]
    )
    stats = analyze(t).stats["speech"]
    assert stats["utterances"] == 3
    assert stats["verbatim_repeat_players"] == 1
    assert stats["fallback_lines"] == 1


def test_discussion_quality_metrics_cover_overlap_balance_replies_and_questions():
    utterances = [
        _say("p1", "占い結果を比較して判断します。", directed_question_targets=["p2"]),
        _say("p2", "占い結果を比較して判断します。"),
        _say("p1", "投票先も比較します。"),
    ]
    t = _transcript(
        utterances=utterances,
        final_state={
            "chat_log": [
                {"channel": "public", "reply_to": None},
                {"channel": "public", "reply_to": "m1"},
            ],
            "pending_questions": {"p2": [{"source_message_id": "m1"}]},
        },
    )

    stats = analyze(t).stats["speech"]

    assert stats["cross_player_topic_overlap_rate"] > 0
    assert stats["speech_count_gini"] > 0
    assert stats["reply_rate"] == 0.5
    assert stats["unanswered_question_rate"] == 1.0
    assert stats["length_variance"] > 0


def test_human_likeness_stats_count_templates_ids_and_volume():
    """The seed-11 live game had every AI line open with the injected candidate
    sentence and name players as `Player12(p12)`. These rates are what the v3
    engine is measured against; a human table scores zero on both."""
    t = _transcript(
        utterances=[
            _say("p1", "現時点の第一処刑候補はユイ(p3)です。根拠は票です。"),
            _say("p2", "ハルトの方が疑わしいという見方も残っています。", day=2),
            _say("p1", "ユイ、昨日の票の理由は？", day=2),
            _say("p0", "現時点の第一処刑候補はユイ(p3)です。"),
        ],
        final_state={
            "chat_log": [
                {"channel": "public", "day": 1, "content": "現時点の第一処刑候補はユイ(p3)です。"},
                {"channel": "public", "day": 2, "content": "ハルトの方が疑わしい。"},
                {"channel": "public", "day": 2, "content": "ユイ、昨日の票の理由は？"},
                {"channel": "wolf", "day": 2, "content": "今夜はユイ。"},
            ]
        },
    )

    stats = analyze(t).stats["speech"]

    assert stats["template_phrase_rate"] == 0.75
    assert stats["id_mention_rate"] == 0.5
    assert stats["public_messages_per_day"] == 1.5
    assert stats["public_mean_length"] > 0


def test_a_chat_register_table_scores_zero_on_templates_and_ids():
    t = _transcript(
        utterances=[
            _say("p1", "2-1。グレランでいいと思う。"),
            _say("p2", "ユイ、昨日の票の理由きかせて。"),
        ]
    )

    stats = analyze(t).stats["speech"]

    assert stats["template_phrase_rate"] == 0.0
    assert stats["id_mention_rate"] == 0.0


def test_length_limit_uses_each_recorded_personality_limit():
    t = _transcript(
        utterances=[
            _say("p1", "短" * 101, effective_length_limit=100),
            _say("p2", "長" * 201, effective_length_limit=400),
        ]
    )

    assert analyze(t).stats["speech"]["over_length_limit"] == 1


def test_clean_transcript_produces_no_findings():
    t = _transcript(
        utterances=[
            _say("p1", "占い師CO。ハルトを占いましたが人狼ではありませんでした。"),
            _say("p2", "占い師CO。ユイを占った結果、白でした。"),
            _say("p0", "二人のCOが出ましたね。慎重に見ていきましょう。"),
        ]
    )
    result = analyze(t)
    assert result.findings == [], [f.detail for f in result.findings]


# -- the public record against what was said and seen --


def _result_event(actor: str, target: str, *, confidence: float, wolf: bool = False) -> dict:
    return {
        "event_id": "e1",
        "source_message_id": "m1",
        "actor_id": actor,
        "event_type": "ability_result",
        "day": 2,
        "target_id": target,
        "role": "seer",
        "result_is_werewolf": wolf,
        "referenced_day": None,
        "confidence": confidence,
    }


def _ledger_transcript(*events: dict, divined: tuple[str, ...] = ()) -> GameTranscript:
    return _transcript(
        final_state={
            "winner": "village",
            "death_records": [],
            "divine_records": [
                {"seer_id": "p1", "target_id": target, "day": 1, "is_werewolf": False}
                for target in divined
            ],
            "speech_events": list(events),
            "chat_log": [
                {"message_id": "m1", "content": "アオイのソウタ白は信用できない。"},
            ],
        }
    )


def test_a_true_seer_credited_with_a_result_they_never_had_is_flagged():
    # A quoted rival's verdict that the public record took for the seer's own.
    t = _ledger_transcript(_result_event("p1", "p2", confidence=0.9), divined=("p3",))

    analysis = analyze(t)

    assert analysis.count("true_role_result_without_record") == 1
    assert analysis.count("undeclared_spoken_result") == 1
    flagged = next(f for f in analysis.findings if f.check == "true_role_result_without_record")
    assert flagged.text == "アオイのソウタ白は信用できない。"


def test_a_declared_result_the_seer_really_had_is_silent():
    t = _ledger_transcript(_result_event("p1", "p3", confidence=1.0), divined=("p3",))

    analysis = analyze(t)

    assert analysis.count("true_role_result_without_record") == 0
    assert analysis.count("undeclared_spoken_result") == 0


def test_a_spoken_result_the_seer_really_had_is_only_noted_as_undeclared():
    t = _ledger_transcript(_result_event("p1", "p3", confidence=0.9), divined=("p3",))

    analysis = analyze(t)

    assert analysis.count("true_role_result_without_record") == 0
    assert analysis.count("undeclared_spoken_result") == 1


def test_a_fake_seer_has_no_records_so_only_undeclared_results_are_noted():
    declared = _ledger_transcript(_result_event("p2", "p3", confidence=1.0))
    spoken = _ledger_transcript(_result_event("p2", "p3", confidence=0.9))

    assert analyze(declared).count("true_role_result_without_record") == 0
    assert analyze(declared).count("undeclared_spoken_result") == 0
    assert analyze(spoken).count("true_role_result_without_record") == 0
    assert analyze(spoken).count("undeclared_spoken_result") == 1


def test_a_wolf_discussing_a_black_result_is_not_calling_its_teammate_a_wolf():
    # "黒だけで" is the particle だけ, not "黒だ": nobody called anyone a wolf here.
    t = _transcript(
        utterances=[_say("p2", "ユイの黒だけで真を決め打つのは危険です。", day=4)]
    )

    assert analyze(t).count("wolf_named_teammate_with_wolf_word") == 0


def test_a_wolf_asserting_its_teammate_is_a_wolf_is_still_flagged():
    t = _transcript(utterances=[_say("p2", "ユイは黒だ。間違いない。", day=4)])

    assert analyze(t).count("wolf_named_teammate_with_wolf_word") == 1


# -- listening: does the stated candidate move, and for what reason --


def _turn(player_id: str, text: str, target: str | None, day: int, reply_to: str | None = None):
    return _say(
        player_id,
        text,
        day=day,
        reasoning_memo={"execution_target": target},
        reply_to=reply_to,
    )


def test_a_candidate_that_moved_of_the_seats_own_accord_is_counted_and_a_death_is_not():
    t = _transcript(
        utterances=[
            _turn("p1", "p2が怪しい", "p2", 1),
            _turn("p1", "確かに、その点は認める。p3に変える", "p3", 1, reply_to="m4"),
            # p3 was executed on day 1, so moving off it later is no change of mind.
            _turn("p1", "では今日はp0", "p0", 2),
        ],
        final_state={
            "winner": "village",
            "death_records": [{"player_id": "p3", "day": 1, "cause": "executed"}],
        },
    )

    stance = analyze(t).stats["stance"]

    assert stance["follow_up_turns"] == 2
    assert stance["voluntary_changes"] == 1
    assert stance["voluntary_changes_same_day"] == 1
    assert stance["voluntary_changes_replying"] == 1
    assert stance["voluntary_changes_acknowledged"] == 1
    assert stance["forced_changes"] == 1


def test_saying_one_has_been_persuaded_without_moving_is_counted_on_its_own():
    t = _transcript(
        utterances=[
            _turn("p1", "p2が怪しい", "p2", 1),
            _turn("p1", "根拠が弱いのは認めます。でもp2です", "p2", 1),
        ]
    )

    stance = analyze(t).stats["stance"]

    assert stance["voluntary_changes"] == 0
    assert stance["acknowledged_without_change"] == 1


def test_transcripts_from_before_replies_were_recorded_report_no_reply_count():
    t = _transcript(
        utterances=[_turn("p1", "p2が怪しい", "p2", 1), _turn("p1", "p3にする", "p3", 1)]
    )

    stance = analyze(t).stats["stance"]

    assert stance["voluntary_changes"] == 1
    assert stance["voluntary_changes_replying"] is None


def test_the_report_shows_how_often_a_candidate_moved():
    from app.eval.report import _render_speech_stats

    t = _transcript(
        utterances=[
            _turn("p1", "p2が怪しい", "p2", 1),
            _turn("p1", "確かに。p3に変える", "p3", 1, reply_to="m1"),
        ]
    )

    table = _render_speech_stats([(t, analyze(t))])

    assert (
        "吊り先の自発的な変更(両方の手番に吊り先があり、元の吊り先が生きていた対) | "
        "1 / 1対(全後続手番 1)" in table
    )
    assert "うち相手の発言に返信しての変更 | 1" in table


# -- listening, as the first Claude game showed it has to be read --
#
# The report counted 19 "acknowledged without change" turns in that game. On a
# full re-read, 11 had no stated candidate on one side (so "unchanged" could
# not be seen), 3 used 認め about somebody else's admission or an earlier
# concession, and the rest kept the view with a stated reason. None was a
# hollow concession (backend/artifacts/live-v3-claude-opus55-seed1-2026-11-16/
# analysis.md, section 3).


def test_an_acknowledgement_with_no_stated_candidate_is_not_counted_as_unmoved():
    t = _transcript(
        utterances=[
            _turn("p1", "様子を見ます", None, 1),
            _turn("p1", "確かにそうだね。さっきのはいったん引っ込めるよ", None, 1),
            _turn("p1", "その通り。なのでユイで見ます", "p3", 1),
        ]
    )

    stance = analyze(t).stats["stance"]

    assert stance["acknowledged_without_change"] == 0
    assert stance["target_unstated_follow_ups"] == 2
    assert stance["target_unstated_acknowledged"] == 2
    # The second pair went from no candidate to one, while taking a point.
    assert stance["target_adopted_acknowledged"] == 1
    assert stance["observable_follow_ups"] == 0


def test_thanking_or_pointing_back_to_an_admission_is_not_taking_a_point():
    t = _transcript(
        utterances=[
            _turn("p1", "ハルトが怪しい", "p2", 3),
            _turn("p1", "ハルトさん、認めてくださってありがとう。でも変えません", "p2", 3),
            _turn("p1", "ハルト、認めたのは評価する。ただ吊りは変えない", "p2", 3),
            _turn("p1", "票乗りの指摘は前に認めた通り。ハルトで", "p2", 3),
        ]
    )

    stance = analyze(t).stats["stance"]

    assert stance["observable_follow_ups"] == 3
    assert stance["acknowledged_without_change"] == 0


def test_a_concession_that_keeps_the_view_with_a_reason_is_told_from_an_empty_one():
    t = _transcript(
        utterances=[
            _turn("p1", "ハルトが怪しい", "p2", 5),
            _turn(
                "p1",
                "確かにその筋は一応ありますね。でも霊媒対抗が出てないので、ハルトのままにします",
                "p2",
                5,
            ),
            _turn("p1", "なるほど、そうですね。ハルトのままで", "p2", 5),
        ]
    )

    stance = analyze(t).stats["stance"]

    assert stance["acknowledged_without_change"] == 2
    assert stance["acknowledged_without_change_with_reason"] == 1
    assert stance["acknowledged_without_change_no_reason"] == 1


def test_the_observable_denominator_is_the_pairs_that_could_show_a_change():
    t = _transcript(
        utterances=[
            _turn("p1", "ハルト", "p2", 1),
            _turn("p1", "ハルトのまま", "p2", 1),  # observable, unchanged
            _turn("p1", "ユイにする", "p3", 1),  # observable, moved
            _turn("p1", "まだ決めきれない", None, 2),  # unstated
            _turn("p1", "あなたで", "p0", 2),  # unstated
            _turn("p1", "あなたのまま", "p0", 3),  # observable
        ],
        final_state={"winner": "village", "death_records": []},
    )

    stance = analyze(t).stats["stance"]

    assert stance["follow_up_turns"] == 5
    assert stance["observable_follow_ups"] == 3
    assert stance["target_unstated_follow_ups"] == 2
    assert stance["voluntary_changes"] == 1


def test_saying_one_changes_ones_mind_counts_as_acknowledging():
    t = _transcript(
        utterances=[
            _turn("p1", "灰に縄を回したい", None, 5),
            _turn("p1", "ただアカリさんの指摘で考えを変えます。ユイ吊りに同意します", "p3", 5),
        ]
    )

    assert analyze(t).stats["stance"]["target_adopted_acknowledged"] == 1


def test_the_report_shows_the_observable_denominator_and_the_unstated_bucket():
    from app.eval.report import _render_speech_stats

    t = _transcript(
        utterances=[
            _turn("p1", "ハルト", "p2", 1),
            _turn("p1", "確かに。でも票がそろっているので、ハルトのまま", "p2", 1),
            _turn("p1", "まだ決めきれない", None, 1),
        ]
    )

    table = _render_speech_stats([(t, analyze(t))])

    assert "| 0 / 1対(全後続手番 2) |" in table
    assert "うち逆接と理由を添えた部分的な譲歩 | 1 |" in table
    assert "うち理由のない空約束 | 0 |" in table
    assert "吊り先が片方でも空欄の対(変化を観測できない) | 1 |" in table


# -- what the code wrote into a player's line --


def test_fact_sentences_the_code_prepended_count_as_template_lines():
    """`ensure_fact_sentences` prepends the canonical CO and result sentences
    when the prose does not state them. The first Claude game had 11 such
    lines (14%), one of them saying the same result twice; the template rate
    counted none of them."""
    t = _transcript(
        utterances=[
            _say("p1", "霊媒CO。霊媒結果、ハルトは黒(人狼)です。おはよう。霊媒結果です。"),
            _say("p2", "占いCO。占い結果、ユイは白(人狼ではない)です。お待たせしました。"),
            _say("p3", "占いCO。リク、そこは認める！"),
            # The model's own claim wording, not the canonical sentence.
            _say("p0", "霊媒師COします。まだ結果はありません。"),
            _say("p1", "占い師CO。0日目の結果、リクは白。"),
            _say("p2", "第二候補はユイ。"),
        ]
    )

    stats = analyze(t).stats["speech"]

    assert stats["code_fact_sentence_lines"] == 3
    assert stats["code_fact_sentence_rate"] == 0.5
    assert stats["prompt_template_phrase_rate"] == round(1 / 6, 4)
    # Either source is wording no player chose.
    assert stats["template_phrase_rate"] == round(4 / 6, 4)


def test_a_fact_sentence_inside_the_prose_is_not_a_prefix():
    t = _transcript(utterances=[_say("p1", "昨日の結果を言うね。霊媒CO。ハルトは黒(人狼)です。")])

    assert analyze(t).stats["speech"]["code_fact_sentence_lines"] == 0


# -- unanswered questions: no turn, or a turn without a reply link --


def test_unanswered_questions_split_never_served_from_served_without_a_link():
    """In the first Claude game the 35% "unanswered" was 4 questions whose
    target never got a turn (all asked in the day-closing summary) and 10 the
    target did get a turn for but answered without a reply link."""
    t = _transcript(
        utterances=[
            _say(
                "p1",
                "ハルト、昨日の票は？ユイ、占い先は？あなたは？",
                directed_question_targets=["p2", "p3", "p0"],
            ),
            _say("p2", "票は情報なし。"),
        ],
        final_state={
            "pending_questions": {
                "p2": [{"source_message_id": "m1", "served": True}],
                "p3": [{"source_message_id": "m1", "served": False}],
            },
        },
    )

    stats = analyze(t).stats["speech"]

    assert stats["unanswered_question_rate"] == round(2 / 3, 4)
    assert stats["unanswered_never_served_rate"] == round(1 / 3, 4)
    assert stats["unanswered_served_without_reply_link_rate"] == round(1 / 3, 4)


def test_questions_recorded_before_served_was_kept_cannot_be_split():
    t = _transcript(
        utterances=[_say("p1", "ハルト、昨日の票は？", directed_question_targets=["p2"])],
        final_state={"pending_questions": {"p2": [{"source_message_id": "m1"}]}},
    )

    stats = analyze(t).stats["speech"]

    assert stats["unanswered_question_rate"] == 1.0
    assert stats["unanswered_never_served_rate"] is None
    assert stats["unanswered_served_without_reply_link_rate"] is None


# -- echo: the next seat restating the line before it --


def test_adjacent_lines_that_restate_each_other_are_measured_on_their_own():
    """The all-pairs overlap averaged a restatement away (0.04% in the first
    Claude game). The next seat repeating the line before it is the echo a
    reader notices, so adjacent lines are measured apart."""
    restated = "1日目夜に占われたのはシオンとソウタだけなので、溶けたのはソウタだけ。"
    t = _transcript(
        utterances=[
            _say("p1", restated, day=3),
            _say("p2", "リクの整理、確かに。" + restated, day=3),
            _say("p3", "ではハルトに質問。昨日の票の理由は？", day=3),
            # Another day: never an adjacent pair with the line before it.
            _say("p0", "ではハルトに質問。昨日の票の理由は？", day=4),
        ]
    )

    stats = analyze(t).stats["speech"]

    assert stats["adjacent_pairs"] == 2
    assert stats["adjacent_overlap_rate"] == 0.5
    assert stats["adjacent_mean_jaccard"] > stats["cross_player_mean_jaccard"]
    (example,) = stats["adjacent_overlap_examples"]
    assert example["day"] == 3 and example["second"] == "p2"


# -- the discussion stage of each line --


def test_utterances_without_a_recorded_stage_report_none():
    """Transcripts do not record the discussion stage yet; the stat says so
    instead of reporting an empty breakdown as if every line were stageless."""
    t = _transcript(utterances=[_say("p1", "様子見。")])

    assert analyze(t).stats["speech"]["utterances_by_stage"] is None


def test_a_recorded_stage_is_counted_per_stage():
    first, second = _say("p1", "おはよう。"), _say("p2", "まとめます。")
    first.stage = "immediate"  # type: ignore[attr-defined]
    second.stage = "consensus_summary"  # type: ignore[attr-defined]
    t = _transcript(utterances=[first, second])

    assert analyze(t).stats["speech"]["utterances_by_stage"] == {
        "immediate": 1,
        "consensus_summary": 1,
    }
