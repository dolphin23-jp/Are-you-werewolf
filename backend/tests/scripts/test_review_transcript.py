import json
import sys

import pytest

from app.eval.release_report import REVIEW_ITEMS, HumanTranscriptReview
from app.eval.transcript import GameTranscript
from scripts.review_transcript import main, run_review


def _transcript(game_id: str = "live-1-v2") -> GameTranscript:
    return GameTranscript(game_id=game_id)


def _write_transcript(path, game_id: str) -> None:
    path.write_text(
        json.dumps(_transcript(game_id).to_dict(), ensure_ascii=False), encoding="utf-8"
    )


def test_run_review_produces_complete_review_from_canned_answers():
    transcript = _transcript()
    review = run_review(
        transcript, "alice", canned_answers={item: True for item in REVIEW_ITEMS}
    )
    assert review.complete is True
    assert review.game_id == transcript.game_id
    assert all(review.answers[item] is True for item in REVIEW_ITEMS)


def test_main_writes_review_named_by_transcript_game_id_not_filename(tmp_path, monkeypatch):
    transcript_path = tmp_path / "arbitrary-filename.json"
    _write_transcript(transcript_path, "live-99-v2")
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(
        json.dumps({item: True for item in REVIEW_ITEMS}), encoding="utf-8"
    )
    review_dir = tmp_path / "reviews"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_transcript.py",
            "--transcript",
            str(transcript_path),
            "--reviewer",
            "alice",
            "--review-dir",
            str(review_dir),
            "--answers-file",
            str(answers_path),
        ],
    )
    main()
    assert (review_dir / "live-99-v2.json").exists()
    assert not (review_dir / "arbitrary-filename.json").exists()


def test_main_refuses_to_overwrite_complete_review_without_force(tmp_path, monkeypatch):
    transcript_path = tmp_path / "t.json"
    _write_transcript(transcript_path, "live-5-v2")
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    complete = HumanTranscriptReview(
        game_id="live-5-v2",
        reviewer="bob",
        reviewed_at="2026-01-01T00:00:00Z",
        answers={item: True for item in REVIEW_ITEMS},
    )
    complete.write_json(review_dir / "live-5-v2.json")
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps({item: False for item in REVIEW_ITEMS}), encoding="utf-8")

    def build_argv(extra: list[str]) -> list[str]:
        return [
            "review_transcript.py",
            "--transcript",
            str(transcript_path),
            "--reviewer",
            "carol",
            "--review-dir",
            str(review_dir),
            "--answers-file",
            str(answers_path),
            *extra,
        ]

    monkeypatch.setattr(sys, "argv", build_argv([]))
    with pytest.raises(SystemExit):
        main()
    unchanged = HumanTranscriptReview.from_json(review_dir / "live-5-v2.json")
    assert unchanged.reviewer == "bob"

    monkeypatch.setattr(sys, "argv", build_argv(["--force"]))
    main()
    overwritten = HumanTranscriptReview.from_json(review_dir / "live-5-v2.json")
    assert overwritten.reviewer == "carol"
    assert overwritten.revision == 2


def test_resumes_incomplete_review_preserving_prior_answers(tmp_path, monkeypatch):
    transcript_path = tmp_path / "t.json"
    _write_transcript(transcript_path, "live-7-v2")
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    partial = HumanTranscriptReview(
        game_id="live-7-v2",
        reviewer="dave",
        reviewed_at="2026-01-01T00:00:00Z",
        answers={item: True for item in REVIEW_ITEMS[:6]},
    )
    partial.write_json(review_dir / "live-7-v2.json")

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(
        json.dumps({item: False for item in REVIEW_ITEMS[6:]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "review_transcript.py",
            "--transcript",
            str(transcript_path),
            "--reviewer",
            "dave",
            "--review-dir",
            str(review_dir),
            "--answers-file",
            str(answers_path),
        ],
    )
    main()
    result = HumanTranscriptReview.from_json(review_dir / "live-7-v2.json")
    assert result.complete is True
    for item in REVIEW_ITEMS[:6]:
        assert result.answers[item] is True
    for item in REVIEW_ITEMS[6:]:
        assert result.answers[item] is False
