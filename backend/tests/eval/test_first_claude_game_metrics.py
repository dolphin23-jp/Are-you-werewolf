"""The rule-based metrics, re-run on the first Claude game's committed transcript.

Every number asserted here was counted by hand when the game was read
(`backend/artifacts/live-v3-claude-opus55-seed1-2026-11-16/analysis.md`,
section 3, "誤解を招く指標"). The report of that game said 19 turns
acknowledged a point without changing, 2.6% of lines carried template
wording and 35% of questions went unanswered; the read found those numbers
measured something other than what their labels say. This pins the metrics
to the reading, on the game that exposed them, so a later change to a
pattern cannot quietly bring the old numbers back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.analyzers import _CODE_FACT_SENTENCE_RE, AnalysisResult, analyze
from app.eval.transcript import GameTranscript

TRANSCRIPT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-opus55-seed1-2026-11-16"
    / "transcript-seed1.json"
)


@pytest.fixture(scope="module")
def first_game() -> AnalysisResult:
    return analyze(GameTranscript.from_dict(json.loads(TRANSCRIPT.read_text(encoding="utf-8"))))


def test_unchanged_is_counted_only_where_it_could_be_seen(first_game: AnalysisResult):
    stance = first_game.stats["stance"]

    assert stance["follow_up_turns"] == 62
    # 16 pairs kept the same stated candidate, 1 moved of its own accord;
    # the 12 forced by a death and the 33 with a blank side cannot show it.
    assert stance["observable_follow_ups"] == 17
    assert stance["forced_changes"] == 12
    assert stance["voluntary_changes"] == 1
    assert stance["target_unstated_follow_ups"] == 33


def test_none_of_the_concessions_was_empty(first_game: AnalysisResult):
    stance = first_game.stats["stance"]

    # Of the 8 same-candidate turns the old count flagged, 3 used 認め about
    # someone else's admission or an earlier concession; the other 5 kept the
    # candidate with でも/ただ/それでも/だが and a reason.
    assert stance["acknowledged_without_change"] == 5
    assert stance["acknowledged_without_change_with_reason"] == 5
    assert stance["acknowledged_without_change_no_reason"] == 0


def test_the_code_written_fact_sentences_are_counted(first_game: AnalysisResult):
    speech = first_game.stats["speech"]

    # 11 of the 77 AI discussion lines open with a CO or result sentence
    # `ensure_fact_sentences` prepended. The prompt vocabulary is 7 more: 2
    # with the structured-era words, and 5 on days 5-6 reciting the v3
    # checklist's own labels (反対仮説, 二番手, 吊らない一番の理由).
    assert speech["code_fact_sentence_lines"] == 11
    assert speech["prompt_template_phrase_rate"] == round(7 / 77, 4)
    assert speech["template_phrase_rate"] == round(18 / 77, 4)


def test_a_transcript_recorded_before_the_split_keeps_its_length_numbers(
    first_game: AnalysisResult,
):
    """Lengths are now the model's own part of a line, but this game was
    recorded before `model_text_chars` existed, so they read the whole posted
    text -- the numbers its report gave."""
    speech = first_game.stats["speech"]

    assert speech["length_lines"] == 93
    assert speech["mean_length"] == 117.6
    assert speech["public_mean_length"] == 129.7
    assert speech["over_length_limit"] == 0


def test_recorded_now_its_code_written_prefixes_would_leave_the_lengths():
    """The same game as the recorder writes it now: each of the 11 lines that
    opens with a code-written CO or result sentence records only the model's
    part, and the 110 prefix characters leave the public mean (9989 -> 9879
    characters over the same 77 public lines)."""
    transcript = GameTranscript.from_dict(json.loads(TRANSCRIPT.read_text(encoding="utf-8")))
    prefix_chars = 0
    for utterance in transcript.utterances:
        match = _CODE_FACT_SENTENCE_RE.match(utterance.text)
        if utterance.kind == "discussion" and match is not None:
            utterance.model_text_chars = len(utterance.text) - len(match.group())
            prefix_chars += len(match.group())

    speech = analyze(transcript).stats["speech"]

    assert prefix_chars == 110
    assert speech["public_mean_length"] == round(9879 / 77, 1)
    assert speech["mean_length"] == 116.5
    assert speech["over_length_limit"] == 0


def test_unanswered_questions_are_split_by_whether_the_target_had_a_turn(
    first_game: AnalysisResult,
):
    speech = first_game.stats["speech"]

    assert speech["unanswered_question_rate"] == 0.35
    # 4 of 40 never reached their target: every one was asked in the
    # day-closing summary. The other 10 got a turn without a reply link.
    assert speech["unanswered_never_served_rate"] == 0.1
    assert speech["unanswered_served_without_reply_link_rate"] == 0.25


def test_the_day_three_restatement_is_the_strongest_adjacent_echo(first_game: AnalysisResult):
    speech = first_game.stats["speech"]

    top = speech["adjacent_overlap_examples"][0]
    # マコト, then アカリ, both "1日目夜に占われたのはシオンさんとソウタさんだけ…".
    assert (top["day"], top["first"], top["second"]) == (3, "p0", "p1")
    assert speech["adjacent_mean_jaccard"] > speech["cross_player_mean_jaccard"]
