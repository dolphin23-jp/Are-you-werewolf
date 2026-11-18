"""The release gate measures a seat's public stance against its vote (owner, 2026-11-17).

A change of vote fails the gate only when all four hold: the seat's basis is
a hang target it named at the table that day, the vote differs, nothing new
and relevant was published after its last line, and its reason cites no
later public argument. The memo's target counts as the basis only when the
seat's own public lines that day say the name, matched whole.

Two kinds stay off the gate and out of 「変更なし」:

* `no_basis` -- the seat had no turn that day, or its memo named nobody.
  Such a ballot was compared with nothing, and the record used to print it as
  「変更なし」.
* `memo_target_unspoken` -- the seat left a memo target it never said aloud.
  The table could not see a change, so it is listed with quotes instead.
* `memo_target_only_reported_or_asked` -- the seat said the name, but only in
  its own result about that seat (in the line the result was registered
  from) or in a question to it (owner's review of run 14, X10). A name in
  one's own line was read as a hang target named; 「0日目夜はソウタさんを
  占って、結果は白でした」 and 「ケントの回答待ちだね」 are neither. Read from
  the sentence's form (`stance_excuse`), never from its intent: 「ダイキさん
  は吊れません」 and 「レンの真偽はまだ決め打てん」 stay stances.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.reasoning_analyzer import (
    JUSTIFIED_CHANGES,
    STANCE_EXCUSE_QUESTION,
    STANCE_EXCUSE_RESULT,
    BallotFacts,
    PublishedResult,
    ReasoningTranscriptAnalyzer,
    SpokenLine,
    VoteChangeKind,
    audit_ballots,
    classify_vote_change,
    stance_excuse,
)
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RUN13 = ARTIFACTS / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
RUN14 = ARTIFACTS / "live-v3-claude-sonnet55-seeds1-3-2026-11-17"
GAMES = {
    "seed1": RUN13 / "seed1" / "transcript-seed1.json",
    "seed2": RUN13 / "seed2" / "transcript-seed2.json",
    "seed3": RUN13 / "seed3" / "transcript-seed3.json",
    "first": ARTIFACTS / "live-v3-claude-opus55-seed1-2026-11-16" / "transcript-seed1.json",
    "run14-seed1": RUN14 / "seed1" / "transcript-seed1.json",
    "run14-seed2": RUN14 / "seed2" / "transcript-seed2.json",
    "run14-seed3": RUN14 / "seed3" / "transcript-seed3.json",
}

NAMES = {"p1": "ハルト", "p2": "サクラ", "p3": "レン", "p4": "ソウタ"}


def _facts(
    basis: str | None,
    vote: str,
    lines: list[tuple[str, str]],
    reason: str = "気が変わった。",
    **fields: object,
) -> BallotFacts:
    return BallotFacts(
        voter_id="p1",
        basis=basis,
        vote_target=vote,
        reason=reason,
        names=NAMES,
        day_lines=tuple(
            SpokenLine(f"m{index}", author, content)
            for index, (author, content) in enumerate(lines, start=1)
        ),
        alive_ids=frozenset(NAMES),
        **fields,  # type: ignore[arg-type]
    )


# -- the classifier --


def test_a_seat_with_no_turn_today_has_no_basis():
    kind, why = classify_vote_change(_facts(None, "p2", [("p2", "レンが怪しい。")]))

    assert kind is VoteChangeKind.NO_BASIS
    assert why == "その日の手番なし"


def test_a_turn_whose_memo_named_nobody_has_no_basis():
    kind, why = classify_vote_change(_facts(None, "p2", [("p1", "灰を見たい。")]))

    assert kind is VoteChangeKind.NO_BASIS
    assert why == "メモの吊り先が空"


def test_no_basis_is_neither_unchanged_nor_a_judged_change():
    assert VoteChangeKind.NO_BASIS is not VoteChangeKind.NONE
    assert VoteChangeKind.NO_BASIS not in JUSTIFIED_CHANGES
    assert VoteChangeKind.MEMO_TARGET_UNSPOKEN not in JUSTIFIED_CHANGES


def test_leaving_a_target_the_seat_named_with_nothing_new_is_unexplained():
    kind, _why = classify_vote_change(_facts("p2", "p3", [("p1", "サクラさんが怪しい。")]))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_leaving_a_memo_target_never_said_aloud_is_unspoken():
    """Someone else naming the target is not the seat's stance."""
    kind, why = classify_vote_change(
        _facts("p2", "p3", [("p1", "灰を見たい。"), ("p4", "サクラは白っぽい。")])
    )

    assert kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN
    assert "サクラ" in why


def test_the_target_must_be_said_whole():
    """レン inside チャレンジ is not レン."""
    kind, _why = classify_vote_change(_facts("p3", "p2", [("p1", "占いチャレンジは早い。")]))

    assert kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN


def test_a_vote_that_follows_an_unspoken_memo_target_is_unchanged():
    kind, _why = classify_vote_change(_facts("p2", "p2", [("p1", "灰を見たい。")]))

    assert kind is VoteChangeKind.NONE


def test_a_runoff_that_dropped_an_unspoken_target_is_still_the_runoff():
    kind, _why = classify_vote_change(
        _facts("p2", "p3", [("p1", "灰を見たい。")], runoff=True, candidates=("p3", "p4"))
    )

    assert kind is VoteChangeKind.RUNOFF_RESTRICTION


# -- a name that is no stance (X10) --


@pytest.mark.parametrize(
    "sentence",
    [
        "0日目夜はソウタさんを占って、結果は白でした",
        "0日目夜の結果はソウタが白",
        "昨夜の占いはソウタ=黒、人狼です",
        "占い結果、ソウタは白(人狼ではない)です",
    ],
)
def test_the_seats_own_result_sentence_is_no_stance(sentence: str):
    assert stance_excuse(sentence, "p4", NAMES, result_line=True) == STANCE_EXCUSE_RESULT


def test_a_result_sentence_counts_only_in_the_line_the_result_came_from():
    """「ソウタは白」 said without a registered result of one's own is a stance."""
    assert stance_excuse("0日目夜の結果はソウタが白", "p4", NAMES) == ""


@pytest.mark.parametrize(
    "sentence",
    [
        # A word about the hang makes it more than the result.
        "ソウタが黒だったので、今日はソウタを吊るでいいよね",
        "ソウタは白なので投票しない",
        # The verdict must follow the name: here 狼 is ハルト's, not ソウタ's.
        "昨日狼のハルトの票がソウタに入ってたから確かめたかった",
        # No verdict at all.
        "ソウタを占った理由はランダムです",
    ],
)
def test_more_than_the_result_stays_a_stance(sentence: str):
    assert stance_excuse(sentence, "p4", NAMES, result_line=True) == ""


@pytest.mark.parametrize(
    ("sentence", "asked"),
    [
        ("ケントの回答待ちだね", "ケント"),
        ("まずレンさんとケントさんの答えを聞きたい", "ケント"),
        ("まずレンさんとケントさんの答えを聞きたい", "レン"),
        ("ケント、マコトを選んだ理由を聞かせてくれ！", "ケント"),
        ("ハルトさん、なぜそこを占ったの？", "ハルト"),
        ("えっと、ダイキさんに聞きたいです", "ダイキ"),
        ("サクラのカイトへの質問、俺も聞きたい", "カイト"),
        ("リク、ここまでの自分の疑い先を答えて", "リク"),
    ],
)
def test_a_question_to_the_seat_is_no_stance_on_it(sentence: str, asked: str):
    names = {"q1": "ケント", "q2": "レン", "q3": "マコト", "q4": "ハルト", "q5": "ダイキ",
             "q6": "サクラ", "q7": "カイト", "q8": "リク"}  # fmt: skip
    seat = next(pid for pid, name in names.items() if name == asked)

    assert stance_excuse(sentence, seat, names) == STANCE_EXCUSE_QUESTION


@pytest.mark.parametrize(
    ("sentence", "named"),
    [
        # Intent is not read: a refusal, a withheld judgement and a push all
        # name the seat as the speaker's stance.
        ("ダイキさんは吊れません", "ダイキ"),
        ("レンの真偽はまだ決め打てん", "レン"),
        ("今日はレン吊りで", "レン"),
        # A question to someone else about the seat.
        ("ケントさんに質問。レンをどう見てる？", "レン"),
        ("ケント、マコトを選んだ理由を聞かせてくれ！", "マコト"),
        # Answer-shaped words with nothing asked.
        ("ケントの説明は苦しい", "ケント"),
        # A list of seats, not a call.
        ("レン、ケントが怪しくない？", "レン"),
        # A charge that no answer came, and thanks for one: no request.
        ("ケント、みんなの質問に答えてないよね", "ケント"),
        ("ダイキさん、正直に答えてくれてありがとう", "ダイキ"),
        # The seat pressed for an answer by the one asked is not asked.
        ("リクさん、レンさんに答えを迫る前に自分の考えを聞きたい", "レン"),
    ],
)
def test_any_other_sentence_naming_the_seat_stays_a_stance(sentence: str, named: str):
    names = {"q1": "ケント", "q2": "レン", "q3": "マコト", "q5": "ダイキ", "q8": "リク"}
    seat = next(pid for pid, name in names.items() if name == named)

    assert stance_excuse(sentence, seat, names) == ""


def test_leaving_a_target_named_only_in_ones_own_result_is_not_judged():
    lines = [("p1", "占いCO。0日目夜の結果はサクラが白。")]
    facts = _facts("p2", "p3", lines, results=(PublishedResult("p1", "p2", "m1"),))

    kind, why = classify_vote_change(facts)

    assert kind is VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED
    assert kind not in JUSTIFIED_CHANGES
    assert why.endswith(f"{STANCE_EXCUSE_RESULT}にだけ出ている: 「0日目夜の結果はサクラが白」")


def test_someone_elses_result_on_the_target_does_not_excuse_the_seats_own_sentence():
    """The excuse needs the voter's own result about the target, from that line."""
    lines = [("p1", "0日目夜の結果はサクラが白。")]
    facts = _facts("p2", "p3", lines, results=(PublishedResult("p4", "p2", "m1"),))

    assert classify_vote_change(facts)[0] is VoteChangeKind.UNEXPLAINED


def test_one_stance_sentence_beside_a_question_keeps_the_stance():
    lines = [("p1", "サクラさん、昨日の理由は？サクラが怪しい。")]

    assert classify_vote_change(_facts("p2", "p3", lines))[0] is VoteChangeKind.UNEXPLAINED


# -- the committed games, re-classified --


def _load(game: str) -> GameTranscript:
    return GameTranscript.from_dict(json.loads(GAMES[game].read_text(encoding="utf-8")))


def _verdict(game: str, day: int, name: str):  # type: ignore[no-untyped-def]
    transcript = _load(game)
    seat = next(pid for pid, seat_name in transcript.names.items() if seat_name == name)
    found = [v for v in audit_ballots(transcript) if v.player_id == seat and v.day == day]
    assert len(found) == 1
    return found[0]


def test_seed1_kaito_day5_left_a_memo_target_he_never_named():
    """カイト's memo said ホノカ all day; his lines that day never did."""
    verdict = _verdict("seed1", 5, "カイト")

    assert verdict.kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN
    assert verdict.own_lines and all("ホノカ" not in line for line in verdict.own_lines)


def test_seed1_daiki_day1_is_re_filed_as_a_public_argument():
    verdict = _verdict("seed1", 1, "ダイキ")

    assert verdict.kind is VoteChangeKind.PUBLIC_ARGUMENT
    assert "m25" in verdict.why


def test_the_first_games_aoi_day2_left_a_memo_target_she_never_named():
    """Her reason says 「昼はレンさんと言いましたが」; neither of her lines that day does."""
    verdict = _verdict("first", 2, "アオイ")

    assert verdict.kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN
    assert len(verdict.own_lines) == 2
    assert all("レン" not in line for line in verdict.own_lines)


@pytest.mark.parametrize("game", sorted(GAMES))
def test_no_ballot_without_a_basis_is_called_unchanged(game: str):
    verdicts = audit_ballots(_load(game))

    assert verdicts
    for verdict in verdicts:
        assert (verdict.basis is None) == (verdict.kind is VoteChangeKind.NO_BASIS)


@pytest.mark.parametrize(
    ("game", "unexplained", "reported_or_asked"),
    [
        ("seed1", 0, 2),
        ("seed2", 1, 0),
        ("seed3", 0, 1),
        ("first", 0, 0),
        ("run14-seed1", 1, 2),
        ("run14-seed2", 0, 3),
        ("run14-seed3", 0, 2),
    ],
)
def test_the_reclassified_games_unexplained_changes(
    game: str, unexplained: int, reported_or_asked: int
):
    """The gate's counts are unchanged by the X10 reading: run 13 0/1/0, the
    first game 0, run 14 1/0/0. Each of the ten ballots it moves to
    `memo_target_only_reported_or_asked` had been filed as justified
    (`public_argument` or `grey_random_policy`), never as unexplained:

    * run 13 seed 1, day 1: ハルト (ホノカ → ユイ) said 「ホノカの返事を先に聞き
      たいな」; ケント (アオイ → ユイ) said 「アオイ、ランダムで選んだ割にソウタ
      って理由は本当にないの？」.
    * run 13 seed 3, day 3: レン (ホノカ → ツムギ), 「昨夜はホノカさんを占って
      白でした」.
    * run 14 seed 1, day 1: ダイキ (シオン → レン), 「0日目夜の結果はシオンが
      白」; ハルト (ケント → レン), 「ケントの回答待ちだね」.
    * run 14 seed 2: アオイ day 2 (レン → マコト), 「…まずレンさんとケントさん
      の答えを聞きたいです」; サクラ day 3 (リク → ミオ), 「リク、ここまでの
      自分の疑い先を答えて」; アオイ day 3 (リク → ミオ), 「リクさんの回答を
      まず聞いてから、グレーを見たいです」.
    * run 14 seed 3: レン day 1 (ソウタ → ケント), 「0日目夜はソウタさんを占っ
      て、結果は白でした」; レン day 2 (アカリ → シオン), 「アカリさんを占って
      白でした」.
    """
    report = ReasoningTranscriptAnalyzer().analyze(_load(game))

    assert report.unexplained_vote_change_count == unexplained
    assert report.memo_target_only_reported_or_asked_count == reported_or_asked


@pytest.mark.parametrize(
    ("game", "day", "name", "sentence"),
    [
        ("run14-seed1", 1, "ダイキ", "0日目夜の結果はシオンが白"),
        ("run14-seed1", 1, "ハルト", "ケントの回答待ちだね"),
        ("run14-seed2", 2, "アオイ", "まずレンさんとケントさんの答えを聞きたいです"),
        ("run14-seed3", 1, "レン", "0日目夜はソウタさんを占って、結果は白でした"),
    ],
)
def test_run14s_reported_and_asked_names_are_listed_with_the_sentence(
    game: str, day: int, name: str, sentence: str
):
    verdict = _verdict(game, day, name)

    assert verdict.kind is VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED
    assert any(sentence in quoted for quoted, _excuse in verdict.excused_sentences)
    assert sentence in verdict.why


def test_run14_seed3_sotas_withheld_judgement_on_ren_is_still_his_stance():
    """「レンの真偽はまだ決め打てん」 is neither shape; telling it from a push is intent."""
    transcript = _load("run14-seed3")
    ren = next(pid for pid, name in transcript.names.items() if name == "レン")
    line = next(m for m in transcript.final_state["chat_log"] if m["message_id"] == "m31")

    assert transcript.names[line["author_id"]] == "ソウタ"
    assert stance_excuse("ふむ、レンの真偽はまだ決め打てん", ren, transcript.names) == ""
    assert "レンの真偽はまだ決め打てん" in line["content"]


def test_seed2_honoka_day3_is_the_one_change_the_gate_still_fails():
    """ホノカ named ダイキ in all three of her lines that day, so the rule counts
    him as her stance; her last line ruled him out (「ダイキさんは吊れません」)
    and she voted the grey seat シオン. Telling a line that rules a seat out from
    one that puts it forward is reading intent, which the audit does not do --
    the X10 reading sets aside only her question to him."""
    verdict = _verdict("seed2", 3, "ホノカ")

    assert verdict.kind is VoteChangeKind.UNEXPLAINED
    assert "ダイキさんは吊れません" in verdict.own_lines[-1]
    # Her question to him is set aside; the ruling-out sentence and her
    # thanks (「正直に答えてくれてありがとう」, no request) are not.
    assert verdict.excused_sentences == (
        ("ダイキさん、ツムギさんを選んだ理由を聞かせてもらえますか？", STANCE_EXCUSE_QUESTION),
    )


@pytest.mark.parametrize("game", sorted(GAMES))
def test_compared_and_unjudged_ballots_add_up(game: str):
    transcript = _load(game)
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    verdicts = audit_ballots(transcript)

    assert report.vote_count == len(verdicts)
    assert report.no_basis_vote_count == sum(
        v.kind is VoteChangeKind.NO_BASIS for v in verdicts
    )
    assert report.compared_vote_count == report.vote_count - report.no_basis_vote_count
    assert report.no_basis_vote_count > 0


def test_the_record_never_prints_a_ballot_without_a_basis_as_unchanged():
    transcript = _load("seed1")
    page = render_transcript(transcript)
    verdicts = audit_ballots(transcript)

    assert page.count("判定: 変更なし") == sum(v.kind is VoteChangeKind.NONE for v in verdicts)
    assert page.count("判定: 比較できない（その日の手番なし）") + page.count(
        "判定: 比較できない（メモの吊り先が空）"
    ) == sum(v.kind is VoteChangeKind.NO_BASIS for v in verdicts)
    assert "判定: 卓で言っていないメモの吊り先からの変更" in page


# -- the report --


def _report(*games: str) -> str:
    from app.eval.analyzers import AnalysisResult
    from app.eval.report import render_report

    return render_report(
        games=[(_load(game), AnalysisResult()) for game in games],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )


def test_the_report_says_how_many_ballots_could_be_compared():
    from app.eval.report import BALLOT_LISTS_HEADING

    page = _report("seed1", "seed2")

    assert BALLOT_LISTS_HEADING in page
    assert "| 1 | 33 / 83 | 50 | 1 |" in page
    assert "| 2 | 33 / 76 | 43 | 0 |" in page


def test_the_report_quotes_each_change_off_an_unspoken_memo_target():
    page = _report("seed1")

    row = next(line for line in page.splitlines() if line.startswith("| 1 | 5 | カイト |"))
    assert "| ホノカ → ケント |" in row
    # His last line that day, and his ballot's reason, whole.
    assert "ケントの黒は霊媒なしでは裏が取れない。" in row
    assert "ケントに入れる。3人連続黒で信用を取りにきてるが" in row


def test_the_report_quotes_each_change_off_a_reported_or_asked_target():
    page = _report("run14-seed1")

    section = page.split("（`memo_target_only_reported_or_asked`）", 1)[1].split("####", 1)[0]
    row = next(line for line in section.splitlines() if line.startswith("| 1 | 1 | ダイキ |"))
    assert "| シオン → レン |" in row
    assert "「0日目夜の結果はシオンが白」（自分の結果の発表）" in row
    assert "「ケントの回答待ちだね」（本人への質問）" in section
    assert "| 1 | 56 / 66 | 10 | 1 | 2 |" in page


def test_the_record_names_a_change_off_a_reported_or_asked_target():
    page = render_transcript(_load("run14-seed1"))

    assert (
        "判定: メモの吊り先に自分の結果の発表か本人への質問でだけ触れた変更（合否に数えない）"
        "（メモの吊り先シオンは、その日の自分の発言では自分の結果の発表にだけ出ている: "
        "「0日目夜の結果はシオンが白」）"
    ) in page


def test_the_report_lists_the_justified_changes_and_the_ballots_with_no_basis():
    page = _report("seed1")

    assert "| 1 | 1 | ダイキ | ケント → ユイ | 後の発言者の論を理由に挙げた |" in page
    assert "| 1 | 2(決選2回目) | アオイ | ケント → レン | 決選の候補から外れた |" in page
    no_basis = page.split("#### 比較できない票（`no_basis`）", 1)[1]
    assert "- seed 1 1日目: " in no_basis
    assert "（その日の手番なし）" in no_basis
    listed = [line for line in no_basis.splitlines() if line.startswith("- seed ")]
    assert listed and not any("変更なし" in line for line in listed)
