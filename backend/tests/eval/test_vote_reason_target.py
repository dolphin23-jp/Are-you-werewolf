"""Ballots whose reason says it goes to someone else (review item 2-A3).

Seed 1 of run 13, day 2: ツムギ's ballot went to ホノカ while its reason ends
「寡黙気味のレンさんに入れるよ」. The vote is the model's own (the code does not
turn a name into a ballot), and nothing re-decides it from the reason -- the
audit only lists such ballots, with the reason quoted, so a reader sees them.

The reading must not catch a reason that describes other seats' ballots:
「ケントに入れた理由」, 「ハルトに入れた人たち」, 「レンに入れてる」 report what
someone else did, and the first scan over the committed games listed those
next to the real case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.analyzers import AnalysisResult
from app.eval.reasoning_analyzer import (
    ReasoningTranscriptAnalyzer,
    audit_ballots,
    seats_reason_votes_for,
)
from app.eval.report import render_report, render_transcript
from app.eval.transcript import GameTranscript

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RUN13 = ARTIFACTS / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
GAMES = {
    "seed1": RUN13 / "seed1" / "transcript-seed1.json",
    "seed2": RUN13 / "seed2" / "transcript-seed2.json",
    "seed3": RUN13 / "seed3" / "transcript-seed3.json",
    "first": ARTIFACTS / "live-v3-claude-opus55-seed1-2026-11-16" / "transcript-seed1.json",
}
NAMES = {"p1": "ツムギ", "p2": "ホノカ", "p3": "レン", "p4": "ケント", "p5": "アオイ"}


def _load(game: str) -> GameTranscript:
    return GameTranscript.from_dict(json.loads(GAMES[game].read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "reason",
    [
        "材料がなさすぎて寡黙気味のレンさんに入れるよ",
        "今日はレンに投票します。",
        "レンに一票。",
        "迷ったけどレンに入れます、理由は発言の少なさ。",
        "レンへ入れとく",
        # Another seat is the subject of its own predicate, not of the ballot.
        "ケントが怪しいから迷ったけど、最後はケントじゃなくてレンに入れる。",
        "アオイが怪しいからレンに入れる。",
    ],
)
def test_a_reason_that_puts_the_ballot_on_another_seat_is_read(reason: str) -> None:
    assert seats_reason_votes_for(reason, "p1", "p2", NAMES) == ("p3",)


@pytest.mark.parametrize(
    "reason",
    [
        # Other seats' ballots, described.
        "ダイキ票でケントに入れた理由が結局薄いままだった。",
        "前回投票でレンに入れた人たちとは違う。",
        "アオイが2日目からずっとレンに入れてるのが気になる。",
        "アオイはレンに入れる。",
        # A quotation, a question, a refusal.
        "レンに入れるって言ってたよね。",
        "レンに入れるか迷う。",
        "レンに入れるのは違う。",
        "レンには入れない。",
        # The ballot's own target, and the voter itself.
        "ホノカさんに入れるよ。",
        "ツムギに入れろって言われた。",
        # A name inside a longer word.
        "占いチャレンジに入れる。",
    ],
)
def test_a_reason_that_only_mentions_a_ballot_is_not_read(reason: str) -> None:
    assert seats_reason_votes_for(reason, "p1", "p2", NAMES) == ()


def test_seed1_tsumugi_day2_is_the_one_ballot_in_four_games():
    found = []
    for game in sorted(GAMES):
        transcript = _load(game)
        for verdict in audit_ballots(transcript):
            if verdict.reason_names_other:
                found.append(
                    (
                        game,
                        verdict.day,
                        transcript.names[verdict.player_id],
                        transcript.names[verdict.vote_target or ""],
                        tuple(transcript.names[seat] for seat in verdict.reason_names_other),
                    )
                )

    assert found == [("seed1", 2, "ツムギ", "ホノカ", ("レン",))]


def test_the_analyzer_counts_it_and_the_report_quotes_it():
    transcript = _load("seed1")
    assert ReasoningTranscriptAnalyzer().analyze(transcript).vote_reason_names_other_count == 1

    report = render_report(
        games=[(transcript, AnalysisResult())],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )

    section = report.split("#### 理由が別の相手への投票を述べている票", 1)[1]
    row = next(line for line in section.splitlines() if line.startswith("| 1 | 2 |"))
    assert "| ツムギ → ホノカ | レン |" in row
    assert "寡黙気味のレンさんに入れるよ" in row


def test_the_record_notes_it_on_the_ballot():
    page = render_transcript(_load("seed1"))

    assert page.count("理由が別の相手（レン(p6)）への投票を述べている") == 1
