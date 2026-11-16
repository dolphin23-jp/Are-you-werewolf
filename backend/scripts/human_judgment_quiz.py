#!/usr/bin/env python3
"""Phase D: can raters tell the AI's lines from human ones? Build the quiz, score it.

No network and no API cost: this only reads files.

    # 1. 問題を作る(AI の発言は評価の対戦記録から、人間の発言はオーナーが用意したファイルから)
    python scripts/human_judgment_quiz.py build \\
        --transcripts eval-out/transcript-seed1.json eval-out/transcript-seed2.json \\
        --human human-lines.txt --items 40 --seed 1

    # 2. eval-out/judgment-quiz/quiz-sheet.md を評価者に渡し、
    #    rater,item,guess の CSV で回答してもらう
    # 3. 採点する
    python scripts/human_judgment_quiz.py score \\
        --key eval-out/judgment-quiz/answer-key.json --ratings ratings.csv

`build` writes quiz-sheet.md (for raters: the lines only), answer-key.json
(the answers, the original lines and where each came from -- keep it from
the raters) and ratings-template.csv. `score` writes score.md and
score.json, which hold no human text and are what gets committed to
`backend/artifacts/`.

The human reference file is the owner's to supply -- the repository ships
none, and the quiz sheet and answer key contain it verbatim, so neither is
committed unless the owner has the right to publish that text. One
utterance per line as plain UTF-8 text, or JSONL with a `text` field (plus
optional `speaker` / `names` to mask in that line). `--mask-names` adds
names to hide in every line, e.g. the player names of the human log.

See `app/eval/judgment_quiz.py` for how the quiz is balanced, matched,
anonymized and scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.judgment_quiz import (  # noqa: E402
    QuizInputError,
    Rating,
    SourceLine,
    ai_lines_from_transcript,
    ai_texts_from_key,
    answer_key,
    answers_from_key,
    build_quiz,
    human_lines_from_text,
    names_from_text,
    parse_ratings_csv,
    ratings_template,
    render_score,
    render_sheet,
    score,
)

DEFAULT_ITEMS = 40
# Inside the git-ignored `eval-out/`: the sheet and the key hold the human
# lines verbatim, and a default that git would pick up is one `git add -A`
# away from publishing someone else's words.
DEFAULT_OUT = Path("eval-out/judgment-quiz")
MAX_ITEMS = 2000


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise QuizInputError(f"{path}: not UTF-8 text ({exc.reason})") from exc
    except OSError as exc:
        raise QuizInputError(f"{path}: {exc.strerror or exc}") from exc


def _transcript_paths(paths: Sequence[Path]) -> list[Path]:
    """Files as given; a directory contributes its `transcript-seed*.json`."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("transcript-seed*.json")))
        else:
            found.append(path)
    if not found:
        raise QuizInputError("no transcript files found")
    return found


def load_ai_lines(paths: Sequence[Path], *, include_fallback: bool) -> list[SourceLine]:
    lines: list[SourceLine] = []
    for path in _transcript_paths(paths):
        try:
            data = json.loads(_read_text(path))
        except json.JSONDecodeError as exc:
            raise QuizInputError(f"{path}: not valid JSON ({exc.msg})") from exc
        if not isinstance(data, dict):
            raise QuizInputError(f"{path}: not an evaluation transcript")
        lines.extend(ai_lines_from_transcript(data, path.name, include_fallback=include_fallback))
    return lines


def run_build(args: argparse.Namespace) -> int:
    ai = load_ai_lines(args.transcripts, include_fallback=args.include_fallback)
    human = human_lines_from_text(_read_text(args.human), args.human.name)
    extra_names = names_from_text(_read_text(args.mask_names)) if args.mask_names else ()
    if not ai:
        raise QuizInputError("the transcripts contain no AI discussion lines")
    if not human:
        raise QuizInputError(f"{args.human}: no human lines")
    quiz = build_quiz(ai, human, items=args.items, seed=args.seed, extra_names=extra_names)
    if not quiz.items:
        raise QuizInputError(
            "no AI and human lines share a length bin; supply more lines of similar length"
        )

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "quiz-sheet.md").write_text(render_sheet(quiz), encoding="utf-8")
    (out / "answer-key.json").write_text(
        json.dumps(answer_key(quiz), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "ratings-template.csv").write_text(ratings_template(quiz), encoding="utf-8")

    print(f"==> 問題 {len(quiz.items)}件(AI {quiz.pairs}・人間 {quiz.pairs})", flush=True)
    print(f"    候補: AI {quiz.ai_pool}件 / 人間 {quiz.human_pool}件(名前を伏せ、重複を除いた数)")
    for row in quiz.bins:
        print(
            f"    {row['label']}: AI {row['ai_candidates']} / 人間 {row['human_candidates']}"
            f" -> {row['pairs']}組"
        )
    if len(quiz.items) < args.items - args.items % 2:
        print(
            f"  [--] {args.items}件に届かなかった。"
            "長さの近い人間の発言が足りない帯がある(上の表)。",
            flush=True,
        )
    print(f"==> 評価者に渡す: {out / 'quiz-sheet.md'}")
    print(f"==> 評価者には見せない: {out / 'answer-key.json'}(人間の発言の原文を含む)")
    return 0


def run_score(args: argparse.Namespace) -> int:
    try:
        key = json.loads(_read_text(args.key))
    except json.JSONDecodeError as exc:
        raise QuizInputError(f"{args.key}: not valid JSON ({exc.msg})") from exc
    if not isinstance(key, dict):
        raise QuizInputError(f"{args.key}: not an answer key")
    answers = answers_from_key(key)
    ratings: list[Rating] = []
    blank = 0
    for path in args.ratings:
        more, more_blank = parse_ratings_csv(_read_text(path), answers)
        ratings.extend(more)
        blank += more_blank
    seen: set[tuple[str, str]] = set()
    for rating in ratings:
        if (rating.rater, rating.item) in seen:
            raise QuizInputError(f"{rating.rater} judged {rating.item} in more than one file")
        seen.add((rating.rater, rating.item))
    report = score(ratings, answers, blank_rows=blank, ai_texts=ai_texts_from_key(key))
    rendered = render_score(report)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.md").write_text(rendered, encoding="utf-8")
    (out / "score.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(rendered, flush=True)
    print(f"==> 出力先: {out / 'score.md'} / {out / 'score.json'}")
    return 0


def _items(text: str) -> int:
    value = int(text)
    if not 2 <= value <= MAX_ITEMS:
        raise argparse.ArgumentTypeError(f"must be between 2 and {MAX_ITEMS}, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="問題・回答用紙・答えを作る")
    build.add_argument(
        "--transcripts",
        type=Path,
        nargs="+",
        required=True,
        help="evaluate.py の transcript-seedN.json(ディレクトリなら中の transcript-seed*.json)",
    )
    build.add_argument(
        "--human", type=Path, required=True, help="人間の発言(1行1発言、または text 欄の JSONL)"
    )
    build.add_argument("--mask-names", type=Path, default=None, help="全行で伏せる名前(1行1名)")
    build.add_argument("--items", type=_items, default=DEFAULT_ITEMS, help="問題数(偶数に丸める)")
    build.add_argument("--seed", type=int, default=1)
    build.add_argument(
        "--include-fallback", action="store_true", help="コードが代わりに出した定型の発言も含める"
    )
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.set_defaults(handler=run_build)

    score_cmd = commands.add_parser("score", help="回答を採点する")
    score_cmd.add_argument("--key", type=Path, required=True, help="build が書いた answer-key.json")
    score_cmd.add_argument(
        "--ratings", type=Path, nargs="+", required=True, help="rater,item,guess の CSV"
    )
    score_cmd.add_argument("--out", type=Path, default=DEFAULT_OUT)
    score_cmd.set_defaults(handler=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except QuizInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
