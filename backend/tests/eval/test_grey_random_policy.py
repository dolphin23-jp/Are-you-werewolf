"""A ballot that follows the facilitator's grey-random policy is a change the table could see.

Run 14 failed seven ballots as unexplained changes. Six of them had the same
shape: on a day the facilitator closed as グレラン, after the seat's last
line, the seat left a seat outside the grey -- a seer claimant it was waiting
to hear from, or the seat its own fake result had called white -- for a grey
seat, and said so in its reason (「マコトがグレランって言ってるし」). The rule
only knew arguments that name one of the two targets, and a grey-random
declaration names nobody.

The cause is narrow on purpose: the declaration must be one of the day's
closing lines (the policy the vote prompt quotes), it must come after the
seat's last line, and the ballot must go from outside the grey into it.
Leaving one grey seat for another is no closer to the policy, so it stays
unexplained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.analyzers import AnalysisResult
from app.eval.reasoning_analyzer import (
    JUSTIFIED_CHANGES,
    BallotFacts,
    ReasoningTranscriptAnalyzer,
    SpokenLine,
    VoteChangeKind,
    audit_ballots,
    classify_vote_change,
)
from app.eval.report import UNEXPLAINED_LIST_HEADING, render_report
from app.eval.transcript import GameTranscript

NAMES = {"p1": "ハルト", "p2": "サクラ", "p3": "レン", "p4": "ソウタ", "p5": "マコト"}
GREY = frozenset({"p3", "p4"})  # ハルト and サクラ claimed seer; マコト freemason


def _facts(
    basis: str,
    vote: str,
    lines: list[tuple[str, str]],
    *,
    closing: tuple[str, ...] = (),
    grey: frozenset[str] | None = GREY,
    reason: str = "マコトがグレランって言ってるし、レンにする。",
) -> BallotFacts:
    return BallotFacts(
        voter_id="p4",
        basis=basis,
        vote_target=vote,
        reason=reason,
        names=NAMES,
        day_lines=tuple(
            SpokenLine(f"m{index}", author, content)
            for index, (author, content) in enumerate(lines, start=1)
        ),
        alive_ids=frozenset(NAMES),
        closing_message_ids=closing,
        grey_ids=grey,
    )


# Run 14, seed 1 day 1, as a shape: the seat speaks about a seer claimant,
# the facilitator then closes the day as grey random, the seat votes a grey
# seat. In run 14 the seat only waited for the claimant's answer (「ハルトの
# 回答待ちだね」); that is no stance since the X10 fix (see
# `test_waiting_on_a_claimants_answer_is_no_stance_to_leave`), so the shape
# says the name as a stance.
DAY = [
    ("p1", "占いCO。0日目夜の結果はマコトが白。"),
    ("p4", "ハルトの対抗は怪しい。答えを聞いてから決めたい。"),
    ("p5", "えっと、今日はグレランでお願いします…。"),
]


def test_waiting_on_a_claimants_answer_is_no_stance_to_leave():
    """Run 14's own line: the name only as the seat asked, so not judged at all."""
    day = [DAY[0], ("p4", "ハルトの回答待ちだね。答えを聞いてから決めたい。"), DAY[2]]

    kind, why = classify_vote_change(_facts("p1", "p3", day, closing=("m3",)))

    assert kind is VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED
    assert "「ハルトの回答待ちだね」" in why


def test_leaving_a_claimant_for_a_grey_seat_after_a_grey_random_close_is_explained():
    kind, why = classify_vote_change(_facts("p1", "p3", DAY, closing=("m3",)))

    assert kind is VoteChangeKind.GREY_RANDOM_POLICY
    assert kind in JUSTIFIED_CHANGES
    assert "m3" in why and "ハルトはグレー外" in why


def test_the_declaration_must_be_a_closing_line():
    """「今日はグレランでいく？」 from anyone is not the day's policy."""
    kind, _why = classify_vote_change(_facts("p1", "p3", DAY, closing=()))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_the_declaration_must_come_after_the_seats_last_line():
    lines = [DAY[0], DAY[2], DAY[1]]

    kind, _why = classify_vote_change(_facts("p1", "p3", lines, closing=("m2",)))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_leaving_one_grey_seat_for_another_is_still_unexplained():
    lines = [("p4", "ソウタじゃなくてレンが怪しい。"), ("p5", "今日はグレランで。")]

    kind, _why = classify_vote_change(
        _facts("p3", "p4", lines, closing=("m2",), reason="やっぱりソウタ。")
    )

    assert kind is VoteChangeKind.UNEXPLAINED


def test_a_grey_random_close_does_not_explain_a_vote_outside_the_grey():
    kind, _why = classify_vote_change(_facts("p1", "p2", DAY, closing=("m3",)))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_a_close_that_designates_is_not_a_grey_random_policy():
    """A 仮指定 is a designation; following it is an argument only when the reason cites it."""
    lines = [*DAY[:2], ("p5", "今日はグレランで、仮指定はレンさん。")]

    kind, _why = classify_vote_change(
        _facts("p1", "p3", lines, closing=("m3",), reason="グレランなのでレン。")
    )

    assert kind is VoteChangeKind.UNEXPLAINED


def test_without_a_known_grey_the_cause_is_never_found():
    kind, _why = classify_vote_change(_facts("p1", "p3", DAY, closing=("m3",), grey=None))

    assert kind is VoteChangeKind.UNEXPLAINED


# -- the committed run 14 records --

RUN14 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-17"
)


def _load(seed: int) -> GameTranscript:
    path = RUN14 / f"seed{seed}" / f"transcript-seed{seed}.json"
    return GameTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(("seed", "live", "now"), [(1, 4, 1), (2, 2, 0), (3, 1, 0)])
def test_run14_reclassified(seed: int, live: int, now: int):
    """The live gate counted 4/2/1; six of the seven followed a grey-random close.

    Since the X10 reading, four of those six are not compared at all: the
    seat had named the claimant only in its own result (ダイキ 「0日目夜の結果
    はシオンが白」, レン 「0日目夜はソウタさんを占って、結果は白でした」) or in
    a question to it (ハルト 「ケントの回答待ちだね」, アオイ 「まずレンさんと
    ケントさんの答えを聞きたいです」) -- `memo_target_only_reported_or_asked`.
    The gate's count is the same either way.
    """
    transcript = _load(seed)
    live_count = sum(
        audit.target_change_classification == "unexplained"
        for audit in transcript.decision_audits
    )

    assert live_count == live
    assert ReasoningTranscriptAnalyzer().analyze(transcript).unexplained_vote_change_count == now


def test_run14_seed1_haruto_day4_is_the_change_the_gate_still_fails():
    """ハルト (wolf) said 「今日はシオンから動かさない」, then voted ミオ.

    Nothing new about either seat came out after his line, and his reason
    cites nobody: a real change of public stance, which the gate fails.
    """
    transcript = _load(1)
    failed = [v for v in audit_ballots(transcript) if v.kind is VoteChangeKind.UNEXPLAINED]

    assert [(transcript.names[v.player_id], v.day) for v in failed] == [("ハルト", 4)]
    assert "シオンから動かさない" in failed[0].own_lines[-1]


def test_the_report_quotes_each_change_the_gate_fails():
    transcript = _load(1)

    page = render_report(
        games=[(transcript, AnalysisResult())],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )

    assert UNEXPLAINED_LIST_HEADING in page
    section = page.split(UNEXPLAINED_LIST_HEADING, 1)[1].split("###", 1)[0]
    assert "| 1 | 4 | ハルト | シオン → ミオ |" in section
    assert "シオンから動かさない" in section
