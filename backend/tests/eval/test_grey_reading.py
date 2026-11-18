"""The report's day-1 discussion and grey-reading counts (owner's review of runs 14 and 15).

Fixed rules over the record, pooled per run and kept out of the release
gate: the day-1 lines on the night-0 divination reasons, the day-1 ballot
grounds by type, grey seats trusted from speech, memo hang targets on seats
that had not spoken yet that day, the genuine seer's divination reasons, and
the same readings by skill tier. The numbers on runs 14 and 15 are pinned so
a rule change shows up as a diff here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.analyzers import AnalysisResult
from app.eval.grey_reading import (
    GREY_READING_HEADING,
    by_tier,
    day1_vote_grounds,
    discusses_divination_reason,
    divination_reason_talk,
    grey_trust,
    line_evaluations,
    render_grey_reading,
    seer_pick_type,
    seer_picks,
    trust_mention,
    unspoken_targets,
    vote_ground_types,
)
from app.eval.release_gate import HARD_FAILURE_FIELDS
from app.eval.report import render_report
from app.eval.transcript import GameTranscript

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RUN14 = ARTIFACTS / "live-v3-claude-sonnet55-seeds1-3-2026-11-17"
RUN15 = ARTIFACTS / "live-v3-claude-sonnet55-seed1-2026-11-18"
NAMES = {"p1": "ユイ", "p2": "マコト", "p3": "リク", "p4": "ツムギ"}


def _load(path: Path) -> GameTranscript:
    return GameTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _run14() -> list[GameTranscript]:
    return [_load(RUN14 / f"seed{seed}" / f"transcript-seed{seed}.json") for seed in (1, 2, 3)]


def _run15() -> list[GameTranscript]:
    return [_load(RUN15 / "transcript-seed1.json")]


# -- the rules, sentence by sentence --


@pytest.mark.parametrize(
    "sentence",
    [
        "ケントとアオイは、なぜマコトとユイを占ったのか答えて",
        "白先の選んだ理由を、ハルトとダイキに聞きたい",
        "理由はない、ランダム",
        "ノゾミは発言も情報もない初日だったので、ほぼ無作為で選んだよ",
        "占い先の理由は皆さん同じでしたし、まずグレーから考えたいです",
    ],
)
def test_a_sentence_on_the_night0_divination_reason(sentence: str):
    assert discusses_divination_reason(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "今夜の占い先とその理由を見てから真偽を考えます",  # tonight's, not night 0's
        "今日はグレランで、グレーの発言をのんびり見たいなあ",
        "理由は結果と一緒に出す！",
    ],
)
def test_not_a_sentence_on_the_night0_divination_reason(sentence: str):
    assert not discusses_divination_reason(sentence)


@pytest.mark.parametrize(
    ("line", "kinds"),
    [
        ("ソウタは質問を回すだけで自分の見立てを出してない。", {"thin"}),
        ("ツムギさんは意見を出すと言いつつ、まだ具体的な疑い先を示していません。", {"thin"}),
        ("ユイさんは理由が毎回動くし、弱いと認めた後に疑いを散らした。", {"content"}),
        ("白を出した相手に2回連続で投票した説明がつきません。", {"content", "result"}),
        ("マコトさんの本指定どおりリクさんに入れます。", {"designation"}),
        ("リクは黒だったから入れる。", {"result"}),
        ("さっき言った通り、ツムギさんを軸に見ているのでそのまま入れます。", {"other"}),
        # A result denied or still to come is no result ground.
        ("黒判定がなく、決定的な黒材料はまだない。", {"other"}),
        ("今夜の占い結果を見て判断する。", {"other"}),
    ],
)
def test_vote_ground_types(line: str, kinds: set[str]):
    assert vote_ground_types(line) == kinds


def test_grey_trust_words_name_the_nearest_seat_before_them():
    assert [
        s for s, _ in trust_mention("みんなツムギばっかだけど、薄いだけで吊るのは怖いよ", NAMES)
    ] == ["p4"]
    assert [s for s, _ in trust_mention("リクは村っぽいと思う", NAMES)] == ["p3"]
    assert [s for s, _ in trust_mention("マコトは今日の吊り位置じゃない", NAMES)] == ["p2"]
    assert trust_mention("ユイの白もらったけど、それだけで真とは見ない", NAMES) == []
    assert trust_mention("その話は面白いね", NAMES) == []


@pytest.mark.parametrize(
    ("sentence", "found"),
    [
        ("占い3人の理由はユイの質問に同意で、回答の差を見たいです", [("p1", 1)]),
        ("マコトさん、ツムギさんの言う通り理由が薄いかな", [("p4", 1)]),
        ("ツムギさんの指摘は一理あると思います", [("p4", 1)]),
        ("リクさんの反論は筋が通っていると思います", [("p3", 1)]),
        ("ユイの発言が引っかかる", [("p1", -1)]),
        ("リクの答えは筋が通らない", [("p3", -1)]),
        ("マコトに同意", [("p2", 1)]),
        # Not an evaluation of a line: a condition, a person, an act.
        ("ユイさん、マコトさんが不自然なら本命は誰なの", []),
        ("ツムギは怪しい", []),
        ("リクに白を出して票も入れたのは確かに不自然", []),
    ],
)
def test_line_evaluations(sentence: str, found: list[tuple[str, int]]):
    assert line_evaluations(sentence, NAMES) == found


def test_seer_pick_types():
    assert seer_pick_type("情報が何もないので、無作為に一人選ぶ", target_claimed=False) == "random"
    assert seer_pick_type("投票の動きが気になるシオンを占う", target_claimed=False) == "suspicion"
    assert (
        seer_pick_type("吊られにくい位置のリクを先に確定させる", target_claimed=False) == "strong"
    )
    assert seer_pick_type("ダイキは人狼か妖狐の見立て", target_claimed=True) == "claimant"
    assert seer_pick_type("グレーを狭める", target_claimed=False) == "other"


# -- run 14 (three games, no skills recorded) --


def test_run14s_day1_lines_on_the_divination_reasons():
    found = divination_reason_talk(_run14())
    assert (found.matched, found.lines) == (23, 57)
    assert any(text == "理由はない、ランダム" for *_, text in found.examples)


def test_run14s_day1_ballots_are_thin():
    found = day1_vote_grounds(_run14())
    assert found.ballots == 48
    assert (found.line["thin"], found.line["content"], found.line["result"]) == (46, 0, 0)
    assert (found.line["designation"], found.line["other"]) == (1, 2)
    assert (found.evidence["thin"], found.evidence["content"]) == (48, 1)
    assert found.thin_only == 45


def test_run14s_grey_trust_from_speech():
    found = grey_trust(_run14())
    assert (found.memo_lines_with, found.memo_lines) == (0, 339)
    assert sum(found.memo.values()) == 0
    # One public line, by a villager, on a grey who was a werewolf.
    assert dict(found.public) == {("village", "werewolf"): 1}
    assert (
        found.public_examples[0][3] == "えっ、みんなツムギばっかだけど、薄いだけで吊るのは怖いよ！"
    )


def test_run14s_hang_targets_on_seats_that_had_not_spoken():
    found = unspoken_targets(_run14())
    assert (found.targets_unspoken, found.targets) == (84, 304)
    assert (found.targets_unspoken_day1, found.targets_day1) == (11, 28)
    assert (found.suspects_unspoken, found.suspects) == (188, 635)
    assert (found.suspects_unspoken_day1, found.suspects_day1) == (18, 50)


def test_run14s_seer_picks():
    found = seer_picks(_run14())
    assert dict(found.kinds) == {"random": 3, "suspicion": 3, "claimant": 1}


def test_run14_has_no_skills():
    found = by_tier(_run14())
    assert (found.games_with_skills, found.games_without) == (0, 3)
    assert "記録なし" in render_grey_reading(_run14())


# -- run 15 (one game, skills recorded) --


def test_run15s_day1():
    games = _run15()
    talk = divination_reason_talk(games)
    assert (talk.matched, talk.lines) == (14, 21)
    votes = day1_vote_grounds(games)
    assert votes.ballots == 16
    assert (votes.line["thin"], votes.line["other"], votes.line["content"]) == (15, 1, 0)
    assert (votes.evidence["thin"], votes.evidence["other"]) == (15, 1)


def test_run15s_grey_trust_targets_and_seer():
    games = _run15()
    trust = grey_trust(games)
    assert (trust.memo_lines_with, trust.memo_lines) == (0, 107)
    assert sum(trust.public.values()) == 0
    targets = unspoken_targets(games)
    assert (targets.targets_unspoken, targets.targets) == (24, 99)
    assert (targets.targets_unspoken_day1, targets.targets_day1) == (7, 14)
    assert (targets.suspects_unspoken, targets.suspects) == (45, 201)
    assert (targets.suspects_unspoken_day1, targets.suspects_day1) == (11, 26)
    assert dict(seer_picks(games).kinds) == {"random": 1, "suspicion": 1}


def test_run15_by_skill_tier():
    found = by_tier(_run15())
    assert (found.games_with_skills, found.games_without) == (1, 0)
    expert, average, novice = (found.rows[tier] for tier in ("expert", "average", "novice"))
    assert (expert.seats, average.seats, novice.seats) == (6, 6, 5)
    assert (expert.content_ballots, expert.ballots) == (4, 24)
    assert (average.content_ballots, average.ballots) == (3, 18)
    assert (novice.content_ballots, novice.ballots) == (3, 21)
    assert (
        expert.content_ballots_day1 + average.content_ballots_day1 + novice.content_ballots_day1
        == 0
    )
    assert sum(found.memo_trust.values()) + sum(found.public_trust.values()) == 0
    assert (expert.evaluations, average.evaluations, novice.evaluations) == (0, 3, 1)
    assert (average.evaluations_right, novice.evaluations_right) == (2, 1)
    # The miss: a villager agreeing with a werewolf's question on day 1.
    assert any("カイトの質問に同意" in text and "外れ" in text for *_, text in found.examples)


# -- the report --


def test_the_report_carries_the_section_outside_the_gate():
    games = _run15()
    page = render_report(
        games=[(game, AnalysisResult()) for game in games],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )

    assert f"## 8. {GREY_READING_HEADING}" in page
    section = page.split(GREY_READING_HEADING, 1)[1]
    assert "- 占い理由に触れた発言: 14/21（67%）" in section
    assert "| 考え・疑い先が見えない／薄い／発言が少ない | 15/16（94%） | 15/16（94%） |" in section
    assert (
        "| 普通 | 6 | 3/18（17%） | 0/5（0%） | 0 | 0 | 2/3（67%） | 2/3（67%） | 0/0 |" in section
    )
    assert "公開発言で灰を信用した文はありません。" in section
    assert not any("grey" in name or "divination" in name for name in HARD_FAILURE_FIELDS)


def test_no_section_without_a_recorded_game():
    assert render_grey_reading([GameTranscript()]) == ""
