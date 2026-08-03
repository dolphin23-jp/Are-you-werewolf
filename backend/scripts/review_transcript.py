#!/usr/bin/env python3
"""Interactive helper for filling out a HumanTranscriptReview from a v2 transcript.

The release gate refuses to pass without a complete, hand-authored review per
`REVIEW_ITEMS` -- this only makes that authoring less error-prone (surfacing
the transcript sections a reviewer actually needs per question) and never
run by CI or by anything that would let the answers write themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.eval.reasoning_analyzer import (  # noqa: E402
    ReasoningQualityReport,
    ReasoningTranscriptAnalyzer,
)
from app.eval.release_report import REVIEW_ITEMS, HumanTranscriptReview, empty_review  # noqa: E402
from app.eval.transcript import GameTranscript, Utterance  # noqa: E402

InputFn = Callable[[str], str]


def load_transcript(path: Path) -> GameTranscript:
    return GameTranscript.from_dict(json.loads(path.read_text(encoding="utf-8")))


def format_utterance(u: Utterance) -> str:
    tags = []
    if u.public_claim_role:
        tags.append(f"CO={u.public_claim_role}")
    if u.key_point:
        tags.append(f"要点={u.key_point}")
    if u.decision_evidence:
        tags.append(f"根拠={u.decision_evidence}")
    if u.countercase:
        tags.append(f"反論={u.countercase}")
    if u.alternative_target:
        tags.append(f"次善候補={u.alternative_target}")
    if u.public_results:
        tags.append(f"公開結果={u.public_results}")
    suffix = f"  [{', '.join(tags)}]" if tags else ""
    label = f"{u.player_name}({u.player_id}/{u.role}/{u.team})"
    return f"Day{u.day} {u.phase}/{u.kind} {label}: {u.text}{suffix}"


def _print_section(title: str, lines: list[str]) -> None:
    print(f"\n--- {title} ---")
    if not lines:
        print("(該当データなし)")
        return
    for line in lines:
        print(line)


def section_for_item(
    index: int, transcript: GameTranscript, report: ReasoningQualityReport
) -> None:
    """Print exactly what a human needs to read to answer REVIEW_ITEMS[index]."""
    discussion = transcript.by_kind("discussion")
    if index == 0:  # AIごとに発言内容の差があった
        _print_section(REVIEW_ITEMS[0], [format_utterance(u) for u in discussion])
    elif index == 1:  # 差が単なる口調の違いではなかった
        lines = [
            f"personality={u.personality} | {format_utterance(u)}" for u in discussion
        ]
        _print_section(REVIEW_ITEMS[1], lines)
    elif index == 2:  # 人間の主張へ内容面で返答した
        human_id = "p0"
        human_lines = [format_utterance(u) for u in transcript.by_player(human_id)]
        _print_section("人間(p0)の発言", human_lines)
        responses = [
            format_utterance(u) for u in discussion if human_id in u.directed_question_targets
        ]
        _print_section("人間宛と思われる応答", responses)
    elif index == 3:  # 事実訂正後に古い根拠を繰り返さなかった
        lines = [
            f"correction {c.correction_id}: verdict={c.verdict} "
            f"affected={c.affected_seat_ids} retracted={c.retracted_evidence_ids}"
            for c in transcript.correction_audits
        ]
        _print_section("訂正監査記録", lines)
        _print_section(
            "解析結果(古い根拠の使用/公開)",
            [
                f"stale_evidence_attempt_count={report.stale_evidence_attempt_count}",
                f"stale_evidence_publicly_emitted_count="
                f"{report.stale_evidence_publicly_emitted_count}",
            ],
        )
    elif index == 4:  # 多数派への追従だけで議論が決まらなかった
        lines = [
            f"{r.player_id} day{r.day}: decision={r.decision_target} vote={r.vote_target} "
            f"change={r.target_change_classification}"
            for r in transcript.decision_audits
            if r.vote_target is not None
        ]
        _print_section("投票変化の分類", lines)
        _print_section(
            "解析結果",
            [
                f"unexplained_vote_change_count={report.unexplained_vote_change_count}",
                f"unexplained_vote_change_rate={report.unexplained_vote_change_rate:.2f}",
            ],
        )
    elif index == 5:  # 少数意見に具体的根拠があった
        lines = [
            format_utterance(u) for u in discussion if u.alternative_target or u.countercase
        ]
        _print_section(REVIEW_ITEMS[5], lines)
    elif index == 6:  # 人狼の発言が私的情報を漏らしていなかった
        wolves = {pid for pid, team in transcript.teams.items() if team == "werewolf"}
        _print_section(
            "人狼の公開発言", [format_utterance(u) for u in discussion if u.player_id in wolves]
        )
        exposed = sum(
            1
            for r in transcript.decision_audits
            if r.player_id in wolves
            and (
                set(r.public_evidence_ids)
                & (set(r.private_evidence_ids) | set(r.team_private_evidence_ids))
            )
        )
        _print_section(
            "解析結果(private/team_private露出)",
            [
                f"private_evidence_exposed_count={report.private_evidence_exposed_count}",
                f"team_private_evidence_exposed_count="
                f"{report.team_private_evidence_exposed_count}",
                f"人狼決定のうち露出があった件数={exposed}",
            ],
        )
    elif index == 7:  # 能力者が結果を正しく公開した
        lines = [
            f"{r.player_id} day{r.day}: required={r.required_result_ids} "
            f"published={r.published_result_ids} omitted={r.omitted_result_ids} "
            f"duplicate={r.duplicate_result_ids}"
            for r in transcript.result_publication_audits
        ]
        _print_section("結果公開監査記録", lines)
    elif index == 8:  # 騙りの物語が一貫していた、または破綻後に反応した
        fake_roles: dict[str, str] = transcript.deception.get("fake_role_by_player", {})
        _print_section("騙り配役", [f"{pid}: {role}" for pid, role in fake_roles.items()])
        _print_section(
            "騙り担当者の公開発言",
            [format_utterance(u) for u in discussion if u.player_id in fake_roles],
        )
        affected = [
            c for c in transcript.correction_audits if set(c.affected_seat_ids) & set(fake_roles)
        ]
        _print_section(
            "騙り担当者に影響した訂正",
            [f"{c.correction_id}: verdict={c.verdict} affected={c.affected_seat_ids}"
             for c in affected],
        )
    elif index == 9:  # 公開発言と投票が自然につながっていた
        lines = []
        for u in transcript.by_kind("vote"):
            prior = [d for d in discussion if d.player_id == u.player_id]
            last = prior[-1] if prior else None
            line = f"{u.player_id} vote target={u.target}"
            if last is not None:
                line += f" | 直前の発言: {format_utterance(last)}"
            lines.append(line)
        _print_section(REVIEW_ITEMS[9], lines)


def ask_yes_no(prompt: str, input_fn: InputFn = input) -> bool:
    while True:
        raw = input_fn(f"{prompt} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("y か n で答えてください。")


def ask_notes(prompt: str, input_fn: InputFn = input) -> str:
    return input_fn(f"{prompt} について一言(任意、空欄可): ").strip()


def run_review(
    transcript: GameTranscript,
    reviewer: str,
    *,
    existing: HumanTranscriptReview | None = None,
    input_fn: InputFn = input,
    canned_answers: dict[str, bool] | None = None,
) -> HumanTranscriptReview:
    """Fill in a HumanTranscriptReview, resuming from `existing` if given.

    Already-answered items are kept as-is unless `canned_answers` explicitly
    overrides them; unanswered items are taken from `canned_answers` when
    present, otherwise asked interactively via `input_fn`.
    """
    report = ReasoningTranscriptAnalyzer().analyze(transcript)
    base = existing if existing is not None else empty_review(transcript.game_id)
    answers = dict(base.answers)
    notes_lines = [base.notes] if base.notes else []
    for index, item in enumerate(REVIEW_ITEMS):
        if canned_answers is not None and item in canned_answers:
            answers[item] = canned_answers[item]
            continue
        if answers.get(item) is not None:
            continue
        print(f"\n=== [{index + 1}/{len(REVIEW_ITEMS)}] {item} ===")
        section_for_item(index, transcript, report)
        answers[item] = ask_yes_no(item, input_fn)
        note = ask_notes(item, input_fn)
        if note:
            notes_lines.append(f"{item}: {note}")
    return HumanTranscriptReview(
        game_id=transcript.game_id,
        reviewer=reviewer,
        reviewed_at=datetime.now(UTC).isoformat(),
        answers=answers,
        notes="\n".join(notes_lines),
        revision=(base.revision + 1) if existing is not None else base.revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="already-complete レビューへの上書きを許可する"
    )
    parser.add_argument(
        "--answers-file",
        type=Path,
        default=None,
        help="対話式の代わりに {項目文: bool} のJSONで一括回答する(主にテスト/再実行用)",
    )
    args = parser.parse_args()

    transcript = load_transcript(args.transcript)
    if not transcript.game_id:
        raise SystemExit(
            "transcript.game_id が空です。live_ab_reasoning_check.py の出力か確認してください。"
        )

    args.review_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.review_dir / f"{transcript.game_id}.json"

    existing = HumanTranscriptReview.from_json(out_path) if out_path.exists() else None
    if existing is not None and existing.complete and not args.force:
        raise SystemExit(
            f"{out_path} は完了済みレビューです。上書きするには --force を指定してください。"
        )

    canned_answers = None
    if args.answers_file is not None:
        canned_answers = json.loads(args.answers_file.read_text(encoding="utf-8"))

    review = run_review(transcript, args.reviewer, existing=existing, canned_answers=canned_answers)
    review.write_json(out_path)
    print(f"\n{out_path} に保存しました。complete={review.complete}")


if __name__ == "__main__":
    main()
