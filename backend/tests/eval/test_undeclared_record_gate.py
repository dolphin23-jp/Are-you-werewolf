"""The release gate fails a CO or result its own line did not declare (X22-5).

v3's premise is that code guarantees the facts. The public claim record is
written by code from the seats' lines and read by every seat as fact, so a
record the line never declared is a factual error code made -- run 16's
third freemason (「共有CO二人は了解」), run 15's アオイ→リク=黒 (「リク黒が外れた
のはダイキさんの問題」). `undeclared_record_count` counts them; a record that
only repeats what already stood is listed, not counted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.eval.analyzers import AnalysisResult
from app.eval.public_record import describe_record, undeclared_records
from app.eval.reasoning_analyzer import ReasoningQualityReport, ReasoningTranscriptAnalyzer
from app.eval.release_gate import HARD_FAILURE_FIELDS, ReleaseDecision, ReleaseGate
from app.eval.report import UNDECLARED_RECORDS_HEADING, render_report
from app.eval.transcript import GameTranscript, Utterance
from app.eval.transcript_render import render_transcript_full

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
CONFIG = Path(__file__).resolve().parents[2] / "config" / "reasoning_release_gate.toml"


def _load(path: str) -> GameTranscript:
    data = json.loads((ARTIFACTS / path).read_text(encoding="utf-8"))
    return GameTranscript.from_dict(data)


# -- the committed records --

RUN16 = "live-v3-claude-sonnet55-seed1-2026-11-18b/transcript-seed1.json"
RUN15 = "live-v3-claude-sonnet55-seed1-2026-11-18/transcript-seed1.json"


@pytest.mark.parametrize(
    ("path", "counted", "listed"),
    [
        ("live-v3-claude-opus55-seed1-2026-11-16/transcript-seed1.json", 0, 0),
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-16/seed1/transcript-seed1.json", 0, 0),
        # ケント's 「霊媒COするって決めてたよ」: his medium claim stood since m4
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-16/seed2/transcript-seed2.json", 0, 1),
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-16/seed3/transcript-seed3.json", 0, 0),
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-17/seed1/transcript-seed1.json", 0, 0),
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-17/seed2/transcript-seed2.json", 0, 0),
        ("live-v3-claude-sonnet55-seeds1-3-2026-11-17/seed3/transcript-seed3.json", 0, 0),
        # アオイ→リク=黒 (m115), and ホノカ's 「共有だから…」 restatement (m103)
        (RUN15, 1, 2),
        # カイト = 共有者 (m9)
        (RUN16, 1, 1),
    ],
)
def test_the_committed_records_counts(path: str, counted: int, listed: int):
    transcript = _load(path)
    records = undeclared_records(transcript)

    assert len(records) == listed
    assert sum(record.changed_record for record in records) == counted
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    assert report.undeclared_record_count == counted


def test_run16_is_the_third_freemason_nobody_declared():
    transcript = _load(RUN16)

    (record,) = undeclared_records(transcript)

    assert (record.source_message_id, record.actor_id, record.event_type, record.role) == (
        "m9",
        "p8",
        "role_claim",
        "freemason",
    )
    assert record.author_id == "p8" and record.changed_record
    assert record.text.startswith("共有CO二人は了解。")
    assert describe_record(record, transcript.names) == "カイト = 共有者CO"


def test_run15_is_the_black_quoted_from_another_seer():
    transcript = _load(RUN15)

    counted = [record for record in undeclared_records(transcript) if record.changed_record]

    assert [(r.source_message_id, describe_record(r, transcript.names)) for r in counted] == [
        ("m115", "アオイ → リク = 黒（占い）")
    ]
    assert "リク黒が外れたのはダイキさんの問題" in counted[0].text


def test_the_gate_fails_run16_on_it():
    report = ReasoningTranscriptAnalyzer().analyze(_load(RUN16))

    result = ReleaseGate.from_toml(CONFIG).evaluate(
        report,
        {},
        live_games=2,
        mock_llm_reduction=0.6,
        human_review_complete=True,
        operational_complete=True,
    )

    assert "undeclared_record_count" in HARD_FAILURE_FIELDS
    assert result.decision is ReleaseDecision.FAIL
    assert "undeclared_record_count" in result.reasons
    # X10: カイト's ballot now follows the final designation; this is run 16's
    # only hard failure.
    assert result.reasons == ("undeclared_record_count",)


def test_the_report_and_the_record_quote_it():
    transcript = _load(RUN16)

    page = render_report(
        games=[(transcript, AnalysisResult())],
        metrics_summary={},
        judge_summary=None,
        provider="AnthropicClaudeProvider",
        model="m",
    )
    section = page.split(UNDECLARED_RECORDS_HEADING, 1)[1].split("###", 1)[0]
    assert "| 1 | 1 | m9 | カイト = 共有者CO | 共有CO二人は了解。" in section

    record = render_transcript_full(transcript)
    assert "- 1日目のCO: カイト(p8) = 共有者（m9）【本人の宣言なし】" in record
    assert "- 1日目: 本人が宣言していない記録（m9）: カイト = 共有者CO" in record


# -- hand-built records --

NAMES = {"p1": "マコト", "p2": "ホノカ", "p3": "カイト", "p4": "ミオ", "p5": "アオイ"}


def _utterance(
    seat: str, text: str, *, claim: str | None = None, results: list[dict[str, Any]] | None = None
) -> Utterance:
    return Utterance(
        day=1,
        phase="discussion",
        kind="discussion",
        player_id=seat,
        player_name=NAMES[seat],
        role="villager",
        team="village",
        personality="-",
        deception_role=None,
        text=text,
        public_claim_role=claim,
        public_results=results or [],
    )


def _transcript(
    lines: list[tuple[str, str, str | None]], events: list[dict[str, Any]]
) -> GameTranscript:
    """`lines` are (seat, text, declared role); `events` the record written from them."""
    utterances = [_utterance(seat, text, claim=claim) for seat, text, claim in lines]
    chat_log = [
        {"message_id": f"m{i}", "author_id": seat, "content": text, "channel": "public", "day": 1}
        for i, (seat, text, _claim) in enumerate(lines, start=1)
    ]
    speech_events = [
        {
            "event_id": f"e{i}",
            "day": 1,
            "target_id": None,
            "result_is_werewolf": None,
            "referenced_day": None,
            "confidence": 1.0,
            **event,
        }
        for i, event in enumerate(events, start=1)
    ]
    return GameTranscript(
        names=NAMES,
        utterances=utterances,
        final_state={"chat_log": chat_log, "speech_events": speech_events},
    )


def _claim(message: str, seat: str, role: str, confidence: float = 1.0) -> dict[str, Any]:
    return {
        "source_message_id": message,
        "actor_id": seat,
        "event_type": "role_claim",
        "role": role,
        "confidence": confidence,
    }


def test_a_declared_claim_passes_and_an_acknowledgement_fails():
    transcript = _transcript(
        [
            ("p1", "共有者CO。相方はホノカです。", "freemason"),
            ("p3", "共有CO二人は了解。", None),
        ],
        [_claim("m1", "p1", "freemason"), _claim("m2", "p3", "freemason", 0.9)],
    )

    records = undeclared_records(transcript)

    assert [(r.source_message_id, r.changed_record) for r in records] == [("m2", True)]


def test_a_claim_said_in_its_own_words_passes_without_the_field():
    transcript = _transcript(
        [("p3", "霊媒CO、まだ結果はありません。", None)],
        [_claim("m1", "p3", "medium", 0.9)],
    )

    assert undeclared_records(transcript) == ()


def test_a_claim_written_from_another_seats_line_fails():
    transcript = _transcript(
        [("p1", "占い師CO。", "seer")],
        [_claim("m1", "p3", "seer")],
    )

    (record,) = undeclared_records(transcript)

    assert (record.actor_id, record.author_id, record.changed_record) == ("p3", "p1", True)


def test_a_restatement_is_listed_but_not_counted():
    transcript = _transcript(
        [
            ("p2", "共有者CO。マコトの相方は私ホノカで間違いありません。", "freemason"),
            ("p2", "共有だから私は白で合ってるよ。", None),
        ],
        [_claim("m1", "p2", "freemason"), _claim("m2", "p2", "freemason", 0.9)],
    )

    records = undeclared_records(transcript)

    assert [(r.source_message_id, r.changed_record) for r in records] == [("m2", False)]
    assert ReasoningTranscriptAnalyzer().analyze(transcript).undeclared_record_count == 0


def test_an_event_below_the_binding_threshold_is_not_on_the_record():
    transcript = _transcript(
        [("p3", "さっき「占い師CO」と言っていましたよね。", None)],
        [_claim("m1", "p3", "seer", 0.4)],
    )

    assert undeclared_records(transcript) == ()


def test_a_result_quoted_from_another_seer_fails_and_one_declared_passes():
    transcript = _transcript(
        [
            ("p5", "占い師CO。ミオは白でした。", "seer"),
            ("p5", "占いの話だけど、リク黒が外れたのはダイキの問題。", None),
        ],
        [
            _claim("m1", "p5", "seer"),
            {
                "source_message_id": "m1",
                "actor_id": "p5",
                "event_type": "ability_result",
                "role": "seer",
                "target_id": "p4",
                "result_is_werewolf": False,
                "confidence": 0.9,
            },
            {
                "source_message_id": "m2",
                "actor_id": "p5",
                "event_type": "ability_result",
                "role": "seer",
                "target_id": "p3",
                "result_is_werewolf": True,
                "confidence": 0.9,
            },
        ],
    )
    transcript.names = {**NAMES, "p3": "リク"}

    records = undeclared_records(transcript)

    assert [(r.source_message_id, r.target_id) for r in records] == [("m2", "p3")]


def test_a_record_without_a_chat_log_is_not_audited():
    transcript = GameTranscript(
        names=NAMES, final_state={"speech_events": [_claim("m1", "p1", "seer")]}
    )

    assert undeclared_records(transcript) == ()
    assert ReasoningQualityReport().undeclared_record_count == 0
