"""A wolf's vote on its partner that follows the day's designation and decides nothing.

The owner's gate (2026-11-17): an ally vote that follows the day's final
designation and does not change the execution -- the round recounted
without it executes the same seat -- is not a hard failure; it is listed in
the report with the designation it followed. An unplanned ally vote that was
decisive, or cast with no designation to follow, still fails.

Seed 3 of run 13 is the case: レン, a wolf, voted his partner マコト on day 1
(14 of 16 ballots) and ツムギ on day 3 (10 of 12), each the seat サクラ had
designated. Both were filed as unplanned and failed the game.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.eval.analyzers import AnalysisResult
from app.eval.reasoning_analyzer import AllyVoteKind, ReasoningTranscriptAnalyzer, audit_ballots
from app.eval.report import render_report, render_transcript
from app.eval.transcript import DecisionAuditRecord, GameTranscript, Utterance

SEED3 = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
    / "seed3"
    / "transcript-seed3.json"
)

NAMES = {"p1": "ハルト", "p2": "サクラ", "p3": "レン", "p4": "ソウタ", "p5": "ミオ", "p6": "ケント"}


def _seed3() -> GameTranscript:
    return GameTranscript.from_dict(json.loads(SEED3.read_text(encoding="utf-8")))


def _ren_ally_votes():  # type: ignore[no-untyped-def]
    transcript = _seed3()
    ren = next(pid for pid, name in transcript.names.items() if name == "レン")
    return transcript, [v for v in audit_ballots(transcript) if v.player_id == ren and v.ally_vote]


def test_seed3_rens_two_votes_on_his_partners_followed_the_designation_and_decided_nothing():
    transcript, votes = _ren_ally_votes()

    assert [(v.day, transcript.names[v.vote_target or ""]) for v in votes] == [
        (1, "マコト"),
        (3, "ツムギ"),
    ]
    for verdict in votes:
        assert verdict.ally_kind is AllyVoteKind.DESIGNATED
        assert verdict.ally_decisive is False
        assert verdict.designated_target == verdict.vote_target
        assert "本指定" in verdict.designation_line
        assert not verdict.hard_ally_vote


def test_seed3_has_no_hard_ally_vote_left():
    report = ReasoningTranscriptAnalyzer().analyze(_seed3())

    assert report.wolf_ally_vote_count == 2
    assert report.designated_wolf_ally_vote_count == 2
    assert report.unplanned_wolf_ally_vote_count == 0
    assert report.planned_wolf_ally_vote_count == 0


# -- hand-built days --


def _u(kind: str, player: str, day: int, **fields: Any) -> Utterance:
    base: dict[str, Any] = {
        "day": day,
        "phase": "discussion",
        "kind": kind,
        "player_id": player,
        "player_name": NAMES[player],
        "role": "werewolf" if player in ("p3", "p4") else "villager",
        "team": "werewolf" if player in ("p3", "p4") else "village",
        "personality": "冷静",
        "deception_role": None,
    }
    base.update(fields)
    return Utterance(**base)


def _day(
    *,
    designation: str | None,
    ballots: dict[str, str],
    stage: str = "final_designation",
) -> GameTranscript:
    """Day 1: サクラ designates `designation` (or nobody); wolf レン votes partner ソウタ."""
    utterances: list[Utterance] = []
    chat = [
        {
            "message_id": "m1",
            "author_id": "p1",
            "content": "おはよう。",
            "channel": "public",
            "day": 1,
        }
    ]
    if designation is not None:
        line = f"今日は{NAMES[designation]}に本指定します。"
        utterances.append(
            _u("discussion", "p2", 1, text=line, stage=stage,
               reasoning_memo={"execution_target": designation})
        )
        chat.append(
            {"message_id": "m2", "author_id": "p2", "content": line, "channel": "public", "day": 1}
        )
    for voter, target in ballots.items():
        utterances.append(_u("vote", voter, 1, phase="voting", text="投票する。", target=target))
    audits = [
        DecisionAuditRecord(
            game_id="g", day=1, phase="voting", player_id=voter, vote_target=target,
            vote_only=True, vote_round=1,
            ally_vote=(voter == "p3" and target == "p4"), ally_vote_planned=False,
        )
        for voter, target in ballots.items()
    ]
    return GameTranscript(
        seed=9,
        names=NAMES,
        utterances=utterances,
        final_state={
            "chat_log": chat,
            "vote_records": [
                {"voter_id": voter, "target_id": target, "day": 1, "round": 1}
                for voter, target in ballots.items()
            ],
            "death_records": [],
            "speech_events": [],
        },
        decision_audits=audits,
    )


def _ren(transcript: GameTranscript):  # type: ignore[no-untyped-def]
    return next(v for v in audit_ballots(transcript) if v.player_id == "p3")


def test_a_designated_ally_vote_that_decided_nothing_is_not_hard():
    transcript = _day(
        designation="p4", ballots={"p1": "p4", "p2": "p4", "p3": "p4", "p5": "p6", "p6": "p5"}
    )

    verdict = _ren(transcript)

    assert verdict.ally_kind is AllyVoteKind.DESIGNATED
    assert ReasoningTranscriptAnalyzer().analyze(transcript).unplanned_wolf_ally_vote_count == 0


def test_a_designated_ally_vote_that_decided_the_execution_stays_hard():
    """Without レン's ballot ソウタ and ケント tie: his vote is what hanged his partner."""
    transcript = _day(
        designation="p4", ballots={"p1": "p6", "p2": "p4", "p3": "p4", "p5": "p6", "p6": "p4"}
    )

    verdict = _ren(transcript)

    assert verdict.ally_decisive is True
    assert verdict.ally_kind is AllyVoteKind.UNPLANNED
    assert ReasoningTranscriptAnalyzer().analyze(transcript).unplanned_wolf_ally_vote_count == 1


def test_an_ally_vote_with_no_designation_to_follow_stays_hard():
    transcript = _day(
        designation=None, ballots={"p1": "p4", "p2": "p4", "p3": "p4", "p5": "p6", "p6": "p5"}
    )

    assert _ren(transcript).ally_kind is AllyVoteKind.UNPLANNED


def test_an_ally_vote_against_the_designation_stays_hard():
    transcript = _day(
        designation="p6", ballots={"p1": "p4", "p2": "p4", "p3": "p4", "p5": "p6", "p6": "p5"}
    )

    assert _ren(transcript).ally_kind is AllyVoteKind.UNPLANNED


def test_a_summary_that_closed_the_day_without_a_designation_turn_is_the_designation():
    """With nothing to wait for, the summary's 仮指定 is the day's last word."""
    transcript = _day(
        designation="p4",
        stage="consensus_summary",
        ballots={"p1": "p4", "p2": "p4", "p3": "p4", "p5": "p6", "p6": "p5"},
    )

    assert _ren(transcript).ally_kind is AllyVoteKind.DESIGNATED


def test_the_report_lists_each_designated_ally_vote_with_the_designation():
    report = render_report(
        games=[(_seed3(), AnalysisResult())],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )

    section = report.split("#### 本指定に従った、結果を変えない仲間への投票", 1)[1]
    rows = [line for line in section.splitlines() if line.startswith("| 3 |")]
    assert rows[0].startswith("| 3 | 1 | レン → マコト | ")
    assert "今日はマコトに本指定するよ。" in rows[0]
    assert "マコト 14票" in rows[0]
    assert rows[1].startswith("| 3 | 3 | レン → ツムギ | ")


def test_the_record_says_the_ally_vote_followed_the_designation():
    page = render_transcript(_seed3())

    assert page.count("仲間への投票: 計画なし・本指定どおりで、この票を除いても処刑は同じ") == 2
