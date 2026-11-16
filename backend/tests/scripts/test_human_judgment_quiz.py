"""`scripts/human_judgment_quiz.py`: build a quiz from files, then score it."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.eval.judgment_quiz import AI_LABEL, HUMAN_LABEL
from app.eval.progress import write_transcript_files
from app.eval.transcript import GameTranscript, Utterance
from scripts.human_judgment_quiz import build_parser, main

NAMES = {"p0": "マコト", "p1": "アカリ", "p2": "ハルト"}


def _utterance(text: str, *, kind: str = "discussion") -> Utterance:
    return Utterance(
        day=1,
        phase="discussion",
        kind=kind,
        player_id="p1",
        player_name="アカリ",
        role="villager",
        team="village",
        personality="calm",
        deception_role=None,
        text=text,
    )


def _write_transcript(out: Path, seed: int, texts: list[str]) -> Path:
    """Written exactly the way `scripts/evaluate.py` writes a game."""
    transcript = GameTranscript(
        game_id=f"eval-{seed}",
        seed=seed,
        names=dict(NAMES),
        utterances=[_utterance(t) for t in texts] + [_utterance("内緒", kind="wolf_chat")],
    )
    write_transcript_files(transcript, out, seed, partial=False)
    return out / f"transcript-seed{seed}.json"


# Every line lands in the same length bin (20-39 characters once masked), so
# each can be paired; the quiz-building tests cover unequal bins.
AI_TEXTS = [f"マコトの発言{i:02d}は筋が通っていると思う、たぶん。" for i in range(12)]
HUMAN_TEXTS = [f"タロウさんの意見{i:02d}、ちょっと気になるかも。" for i in range(12)]


def _build(tmp_path: Path, *extra: str, items: int = 10) -> Path:
    games = tmp_path / "eval-out"
    _write_transcript(games, 1, AI_TEXTS[:6])
    _write_transcript(games, 2, AI_TEXTS[6:])
    human = tmp_path / "human.txt"
    human.write_text("# 自分たちの村の記録\n" + "\n".join(HUMAN_TEXTS) + "\n", encoding="utf-8")
    names = tmp_path / "names.txt"
    names.write_text("タロウ\n", encoding="utf-8")
    out = tmp_path / "quiz-out"
    code = main(
        [
            "build",
            "--transcripts",
            str(games),
            "--human",
            str(human),
            "--mask-names",
            str(names),
            "--items",
            str(items),
            "--seed",
            "3",
            "--out",
            str(out),
            *extra,
        ]
    )
    assert code == 0
    return out


def test_build_writes_a_sheet_a_key_and_a_template(tmp_path: Path):
    out = _build(tmp_path)
    assert {p.name for p in out.iterdir()} == {
        "quiz-sheet.md",
        "answer-key.json",
        "ratings-template.csv",
    }
    key = json.loads((out / "answer-key.json").read_text(encoding="utf-8"))
    answers = [item["answer"] for item in key["items"]]
    assert answers.count(AI_LABEL) == answers.count(HUMAN_LABEL) == 5
    assert {
        item["origin"].split("#")[0] for item in key["items"] if item["answer"] == AI_LABEL
    } <= {
        "transcript-seed1.json",
        "transcript-seed2.json",
    }

    sheet = (out / "quiz-sheet.md").read_text(encoding="utf-8")
    # Names from both sources are hidden alike; no answer leaks onto the sheet.
    for name in ("マコト", "アカリ", "タロウ"):
        assert name not in sheet
    assert "〈A〉" in sheet
    assert "transcript-seed" not in sheet and "human.txt" not in sheet
    assert "内緒" not in sheet  # private chat is not public speech

    with (out / "ratings-template.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["item"] for row in rows] == [item["item_id"] for item in key["items"]]


def test_the_same_inputs_and_seed_build_the_same_quiz(tmp_path: Path):
    first = _build(tmp_path / "a")
    second = _build(tmp_path / "b")
    assert (first / "quiz-sheet.md").read_text(encoding="utf-8") == (
        second / "quiz-sheet.md"
    ).read_text(encoding="utf-8")


def _write_ratings(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rater", "item", "guess"])
        writer.writerows(rows)
    return path


def test_score_reads_ratings_and_writes_a_report_without_human_text(tmp_path: Path, capsys):
    out = _build(tmp_path)
    key = json.loads((out / "answer-key.json").read_text(encoding="utf-8"))
    flip = {AI_LABEL: HUMAN_LABEL, HUMAN_LABEL: AI_LABEL}
    # r1 gets everything right; r2 everything wrong (in a second file).
    first = _write_ratings(
        tmp_path / "r1.csv", [("r1", item["item_id"], item["answer"]) for item in key["items"]]
    )
    second = _write_ratings(
        tmp_path / "r2.csv",
        [("r2", item["item_id"], flip[item["answer"]]) for item in key["items"]],
    )
    code = main(
        [
            "score",
            "--key",
            str(out / "answer-key.json"),
            "--ratings",
            str(first),
            str(second),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads((out / "score.json").read_text(encoding="utf-8"))
    assert data["judgments"] == 20 and data["correct"] == 10
    assert data["accuracy"] == 0.5
    assert data["target_demonstrated"] is False
    assert [(r["rater"], r["accuracy"]) for r in data["raters"]] == [("r1", 1.0), ("r2", 0.0)]
    report = (out / "score.md").read_text(encoding="utf-8")
    assert "Clopper-Pearson" in report
    for item in key["items"]:
        if item["answer"] == HUMAN_LABEL:
            # Neither the original human line nor its masked form is published.
            assert item["original_text"] not in report
            assert item["text"] not in report
            assert item["text"] not in json.dumps(data, ensure_ascii=False)
    assert "正答率" in capsys.readouterr().out


def test_bad_inputs_fail_with_a_message_not_a_traceback(tmp_path: Path, capsys):
    out = _build(tmp_path)
    bad = _write_ratings(tmp_path / "bad.csv", [("r1", "Q99", "AI"), ("r1", "Q01", "maybe")])
    code = main(["score", "--key", str(out / "answer-key.json"), "--ratings", str(bad)])
    assert code == 2
    err = capsys.readouterr().err
    assert "Q99" in err and "maybe" in err

    missing = main(
        ["build", "--transcripts", str(tmp_path / "eval-out"), "--human", str(tmp_path / "no.txt")]
    )
    assert missing == 2
    assert "no.txt" in capsys.readouterr().err

    not_a_key = tmp_path / "summary.json"
    not_a_key.write_text('{"games": 1}', encoding="utf-8")
    assert main(["score", "--key", str(not_a_key), "--ratings", str(bad)]) == 2

    latin1 = tmp_path / "latin1.txt"
    latin1.write_bytes("caf\xe9\n".encode("latin-1"))
    code = main(["build", "--transcripts", str(tmp_path / "eval-out"), "--human", str(latin1)])
    assert code == 2
    assert "UTF-8" in capsys.readouterr().err


def test_a_rater_judging_one_item_in_two_files_is_refused(tmp_path: Path, capsys):
    out = _build(tmp_path)
    key = json.loads((out / "answer-key.json").read_text(encoding="utf-8"))
    item = key["items"][0]["item_id"]
    one = _write_ratings(tmp_path / "1.csv", [("r1", item, "AI")])
    two = _write_ratings(tmp_path / "2.csv", [("r1", item, "人間")])
    code = main(["score", "--key", str(out / "answer-key.json"), "--ratings", str(one), str(two)])
    assert code == 2
    assert "more than one file" in capsys.readouterr().err


def test_the_default_output_is_inside_the_git_ignored_eval_out():
    """The sheet and the key hold human lines verbatim; by default they must
    land where `.gitignore` already keeps them out of a commit."""
    parser = build_parser()
    build_args = parser.parse_args(["build", "--transcripts", "t.json", "--human", "h.txt"])
    score_args = parser.parse_args(["score", "--key", "k.json", "--ratings", "r.csv"])
    assert build_args.out.parts[0] == "eval-out"
    assert score_args.out == build_args.out
    gitignore = Path(__file__).resolve().parents[3] / ".gitignore"
    assert "backend/eval-out/" in gitignore.read_text(encoding="utf-8").splitlines()
