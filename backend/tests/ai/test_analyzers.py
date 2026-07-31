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
