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
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.reasoning_analyzer import (
    JUSTIFIED_CHANGES,
    BallotFacts,
    ReasoningTranscriptAnalyzer,
    SpokenLine,
    VoteChangeKind,
    audit_ballots,
    classify_vote_change,
)
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RUN13 = ARTIFACTS / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
GAMES = {
    "seed1": RUN13 / "seed1" / "transcript-seed1.json",
    "seed2": RUN13 / "seed2" / "transcript-seed2.json",
    "seed3": RUN13 / "seed3" / "transcript-seed3.json",
    "first": ARTIFACTS / "live-v3-claude-opus55-seed1-2026-11-16" / "transcript-seed1.json",
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
    ("game", "unexplained"),
    [("seed1", 0), ("seed2", 1), ("seed3", 0), ("first", 0)],
)
def test_the_reclassified_games_unexplained_changes(game: str, unexplained: int):
    report = ReasoningTranscriptAnalyzer().analyze(_load(game))

    assert report.unexplained_vote_change_count == unexplained


def test_seed2_honoka_day3_is_the_one_change_the_gate_still_fails():
    """ホノカ named ダイキ in all three of her lines that day, so the rule counts
    him as her stance; her last line ruled him out (「ダイキさんは吊れません」)
    and she voted the grey seat シオン. Telling a line that rules a seat out from
    one that puts it forward is reading intent, which the audit does not do."""
    verdict = _verdict("seed2", 3, "ホノカ")

    assert verdict.kind is VoteChangeKind.UNEXPLAINED
    assert "ダイキさんは吊れません" in verdict.own_lines[-1]


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


def test_the_report_lists_the_justified_changes_and_the_ballots_with_no_basis():
    page = _report("seed1")

    assert "| 1 | 1 | ダイキ | ケント → ユイ | 後の発言者の論を理由に挙げた |" in page
    assert "| 1 | 2(決選2回目) | アオイ | ケント → レン | 決選の候補から外れた |" in page
    no_basis = page.split("#### 比較できない票（`no_basis`）", 1)[1]
    assert "- seed 1 1日目: " in no_basis
    assert "（その日の手番なし）" in no_basis
    listed = [line for line in no_basis.splitlines() if line.startswith("- seed ")]
    assert listed and not any("変更なし" in line for line in listed)
