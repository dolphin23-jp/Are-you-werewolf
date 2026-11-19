"""A ballot that follows the day's final designation is a change the table could see (X10).

Run 16, day 2: カイト named リク, then the facilitator マコト closed the day
designating アオイ, and カイト voted アオイ with 「アオイに入れる。昨夜の占い先が生きて
いる以上、真はありえない。進行に乗る。」. His reason named nobody, so the
classifier found no argument and failed the ballot as unexplained, while four
seats that wrote 「マコトの本指定どおり」 passed as `public_argument` because
their reason happened to name the designator.

The designation is what the table votes on. Following it is a cause whatever
the reason says, mirroring `grey_random_policy`: the day's last closing line,
after the seat's own last line, designating exactly the seat the ballot went
to (its speaker's plan-note target, said in the line).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.reasoning_analyzer import (
    JUSTIFIED_CHANGES,
    BallotFacts,
    ReasoningTranscriptAnalyzer,
    SpokenLine,
    VoteChangeKind,
    audit_ballots,
    classify_vote_change,
    final_designation,
)
from app.eval.transcript import GameTranscript, Utterance

NAMES = {"p1": "マコト", "p2": "リク", "p3": "アオイ", "p4": "カイト", "p5": "ユイ"}
DAY = [
    ("p4", "リクの消去は早い。今日はグレーの中身を見たい。"),
    ("p1", "仮指定はアオイ。占い先が生きているので真ではありえない。"),
    ("p1", "本指定はアオイ。投票をお願いします。"),
]


def _facts(
    lines: list[tuple[str, str]],
    *,
    designation: str | None = "m3",
    designated: str | None = "p3",
    vote: str = "p3",
    reason: str = "アオイに入れる。昨夜の占い先が生きている以上、真はありえない。進行に乗る。",
) -> BallotFacts:
    return BallotFacts(
        voter_id="p4",
        basis="p2",
        vote_target=vote,
        reason=reason,
        names=NAMES,
        day_lines=tuple(
            SpokenLine(f"m{index}", author, content)
            for index, (author, content) in enumerate(lines, start=1)
        ),
        alive_ids=frozenset(NAMES),
        closing_message_ids=("m2", "m3"),
        designation_message_id=designation,
        designated_target=designated,
    )


def test_following_the_final_designation_is_a_cause_whatever_the_reason_says():
    kind, why = classify_vote_change(_facts(DAY))

    assert kind is VoteChangeKind.FINAL_DESIGNATION
    assert kind in JUSTIFIED_CHANGES
    assert "m3" in why and "アオイ" in why and "進行に乗る" in why


def test_a_designation_the_seat_heard_before_its_last_line_is_no_cause():
    lines = [DAY[1], DAY[2], DAY[0]]

    kind, _why = classify_vote_change(_facts(lines, designation="m2"))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_a_ballot_for_a_seat_the_designation_did_not_name_is_no_cause():
    kind, _why = classify_vote_change(_facts(DAY, vote="p5", reason="ユイに入れる。"))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_a_closing_line_that_designates_nobody_is_no_cause():
    kind, _why = classify_vote_change(_facts(DAY, designated=None))

    assert kind is VoteChangeKind.UNEXPLAINED


def test_an_earlier_closing_line_is_not_the_final_designation():
    """Only the last closing line counts: the 仮指定 m2 named アオイ, m3 another seat."""
    lines = [DAY[0], DAY[1], ("p1", "本指定はユイ。")]

    kind, _why = classify_vote_change(_facts(lines, designated="p5"))

    assert kind is VoteChangeKind.UNEXPLAINED


def _closing(text: str, target: str | None, stage: str = "final_designation") -> Utterance:
    return Utterance(
        day=2,
        phase="discussion",
        kind="discussion",
        player_id="p1",
        player_name="マコト",
        role="freemason",
        team="village",
        personality="-",
        deception_role=None,
        text=text,
        reasoning_memo={"execution_target": target},
        stage=stage,
    )


def test_the_final_designation_is_the_last_closing_line_and_its_named_target():
    lines = [SpokenLine(f"m{i}", a, c) for i, (a, c) in enumerate(DAY, start=1)]
    summary = _closing(DAY[1][1], "p3", stage="consensus_summary")
    designation = _closing(DAY[2][1], "p3")

    assert final_designation([summary, designation], 2, lines, NAMES) == ("m3", "p3")
    # A plan-note target the line does not say is no designation.
    unnamed = _closing(DAY[2][1], "p5")
    assert final_designation([summary, unnamed], 2, lines, NAMES) == ("m3", None)


def test_a_grey_random_close_designates_nobody_even_when_it_names_a_lean():
    text = "今日はグレランで、アオイさんの答えを聞いてから投票に入ろう。"
    lines = [SpokenLine("m1", "p1", text)]

    assert final_designation([_closing(text, "p3", "consensus_summary")], 2, lines, NAMES) == (
        "m1",
        None,
    )


# -- the committed run 16 --

RUN16 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seed1-2026-11-18b"
    / "transcript-seed1.json"
)


def _run16() -> GameTranscript:
    return GameTranscript.from_dict(json.loads(RUN16.read_text(encoding="utf-8")))


def test_run16_kaitos_ballot_follows_the_designation_and_the_gate_count_is_zero():
    transcript = _run16()
    live = sum(a.target_change_classification == "unexplained" for a in transcript.decision_audits)

    verdicts = audit_ballots(transcript)
    kaito = next(v for v in verdicts if v.player_id == "p8" and v.day == 2)

    assert live == 1
    assert kaito.kind is VoteChangeKind.FINAL_DESIGNATION
    assert (kaito.basis, kaito.vote_target) == ("p10", "p15")
    assert "m58" in kaito.why and "進行に乗る" in kaito.why
    assert not [v for v in verdicts if v.kind is VoteChangeKind.UNEXPLAINED]
    assert ReasoningTranscriptAnalyzer().analyze(transcript).unexplained_vote_change_count == 0


def test_run16_the_seats_that_cited_the_designator_keep_their_reading():
    """The new cause comes after the argument: 「マコトの本指定どおり」 still cites マコト."""
    transcript = _run16()
    by_seat = {
        transcript.names[v.player_id]: v.kind
        for v in audit_ballots(transcript)
        if v.day == 2 and v.kind is not VoteChangeKind.NONE
    }

    for seat in ("シオン", "アカリ", "ホノカ", "ミオ"):
        assert by_seat[seat] is VoteChangeKind.PUBLIC_ARGUMENT
