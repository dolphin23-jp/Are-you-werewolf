"""Renders evaluation output for humans.

Two artefacts, because the questions have different shapes: a summary
report for the measurable metrics, and a readable per-game transcript so
you can judge Japanese quality with your own eyes rather than trusting a
judge model.
"""

from __future__ import annotations

from typing import Any

from app.eval.analyzers import AnalysisResult
from app.eval.transcript import GameTranscript

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

_CHECK_LABELS = {
    "result_claim_without_role": "役職に無い占い/霊媒結果の主張",
    "co_role_mismatch": "役職と異なるCO",
    "co_role_changed": "CO役職の変更(明確な矛盾)",
    "treats_dead_player_as_active": "死亡済みプレイヤーを現在の行動対象として扱う",
    "vote_contradicts_stated_intent": "思考メモと投票先の不一致",
    "wolf_named_teammate_with_wolf_word": "人狼が仲間を明示的に狼視",
    "wolf_memo_suspects_teammate": "人狼の非公開メモで仲間を疑い先に指定",
    "wolf_voted_teammate": "人狼の仲間投票(戦略観測)",
    "assigned_faker_never_claimed": "騙り担当がCOしなかった",
    "lurker_broke_cover": "潜伏担当がCOした",
    "meta_phrase_leaked": "メタ発言の漏れ",
    "low_japanese_ratio": "日本語比率が低い発言",
    "self_treated_as_other_player": "自分自身を他プレイヤーとして扱う",
    "claimed_p0_identity": "p0ではないプレイヤーがp0本人を主張",
    "true_role_result_conflict": "真役職が実際と逆の判定結果を主張",
}


def render_report(
    *,
    games: list[tuple[GameTranscript, AnalysisResult]],
    metrics_summary: dict[str, Any],
    judge_summary: dict[str, Any] | None,
    provider: str,
    model: str,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# AI人狼プレイヤー 評価レポート\n")
    add(f"- プロバイダ: `{provider}` / モデル: `{model}`")
    add(f"- 評価ゲーム数: {len(games)}")
    if provider == "MockProvider":
        add("")
        add(
            "> **注意: モックプロバイダでの実行です。** 定型文を返すだけなので、"
            "日本語の質・人格・推理に関する数値には意味がありません。"
            "ハーネス自体の動作確認用です。"
        )
    add("")

    add("## 1. 動作指標(JSON成功率・応答時間・コスト)\n")
    add(_render_metrics(metrics_summary))

    add("\n## 2. ルールベース検査(役職整合性・矛盾・人狼の連携)\n")
    add(_render_findings(games))

    add("\n## 3. 発言の形式指標\n")
    add(_render_speech_stats(games))

    add("\n## 4. 日本語の自然さ・人格の維持\n")
    if judge_summary is None:
        add(
            "LLM判定は未実行です(`--judge` で有効化)。\n\n"
            "`transcript-*.md` に読みやすい形式の全発言を出力しているので、"
            "目視での確認も可能です。"
        )
    else:
        add(_render_judge(judge_summary))

    return "\n".join(lines) + "\n"


def _render_metrics(summary: dict[str, Any]) -> str:
    if not summary or summary.get("total_calls", 0) == 0:
        return "(LLM呼び出しの記録がありません)"

    rows = [
        "| 指標 | 値 |",
        "|---|---|",
        f"| 総呼び出し数 | {summary['total_calls']} |",
        f"| **JSON成功率** | {summary['success_rate']:.1%} |",
        f"| うち strict schema で成功 | {summary['strict_schema_rate']:.1%} |",
        f"| リトライが発生した呼び出し | {summary['retry_calls']} |",
        f"| 議論生成スキップ | {summary.get('discussion_skips', 0)} / "
        f"{summary.get('discussion_generation_attempts', 0)} "
        f"({summary.get('discussion_skip_rate', 0.0):.1%}) |",
        f"| 応答時間 平均 | {summary['latency_seconds']['mean']:.3f} 秒 |",
        f"| 応答時間 p50 / p95 | "
        f"{summary['latency_seconds']['p50']:.3f} / {summary['latency_seconds']['p95']:.3f} 秒 |",
        f"| 応答時間 最大 | {summary['latency_seconds']['max']:.3f} 秒 |",
    ]

    tokens = summary.get("tokens")
    if tokens:
        rows.append(f"| 入力トークン | {tokens['prompt']:,} |")
        rows.append(f"| 出力トークン | {tokens['completion']:,} |")
        if summary.get("pricing_supplied"):
            rows.append(f"| **推定コスト** | {summary['estimated_cost']:.6f} |")
        else:
            rows.append("| 推定コスト | 単価未指定 (`--price-in` / `--price-out` で算出) |")
    else:
        rows.append("| トークン/コスト | エンドポイントが usage を返していません |")

    out = ["\n".join(rows)]

    by_schema = summary.get("by_schema") or {}
    if by_schema:
        out.append("\n呼び出し種別ごと:\n")
        out.append("| 種別 | 回数 | 成功率 | 平均応答 |")
        out.append("|---|---|---|---|")
        for schema, data in by_schema.items():
            out.append(
                f"| {schema} | {data['calls']} | {data['success_rate']:.1%} | "
                f"{data['mean_latency_seconds']:.3f} 秒 |"
            )

    errors = summary.get("errors") or []
    if errors:
        out.append("\n主なエラー:\n")
        for entry in errors:
            out.append(f"- `{entry['error']}` ({entry['count']}件)")

    return "\n".join(out)


def _render_findings(games: list[tuple[GameTranscript, AnalysisResult]]) -> str:
    totals: dict[str, dict[str, Any]] = {}
    for _transcript, analysis in games:
        for finding in analysis.findings:
            entry = totals.setdefault(
                finding.check, {"count": 0, "severity": finding.severity, "examples": []}
            )
            entry["count"] += 1
            if len(entry["examples"]) < 3:
                entry["examples"].append(finding)

    if not totals:
        return "検出された問題はありません。"

    ranked = sorted(
        totals.items(),
        key=lambda kv: (_SEVERITY_ORDER.get(kv[1]["severity"], 9), -kv[1]["count"]),
    )

    out = ["| 重要度 | 検査項目 | 件数 |", "|---|---|---|"]
    for check, entry in ranked:
        label = _CHECK_LABELS.get(check, check)
        out.append(f"| {entry['severity']} | {label} | {entry['count']} |")

    out.append("\n### 具体例\n")
    for check, entry in ranked:
        label = _CHECK_LABELS.get(check, check)
        out.append(f"**{label}** ({entry['severity']})\n")
        for finding in entry["examples"]:
            out.append(f"- {finding.day}日目 `{finding.player_id}`: {finding.detail}")
            if finding.text:
                out.append(f"  > {finding.text}")
        out.append("")

    return "\n".join(out)


def _render_speech_stats(games: list[tuple[GameTranscript, AnalysisResult]]) -> str:
    collected: list[dict[str, Any]] = [
        stats for _t, a in games if (stats := a.stats.get("speech")) is not None
    ]
    if not collected:
        return "(発言がありません)"

    total = sum(s["utterances"] for s in collected)
    mean_len = sum(s["mean_length"] * s["utterances"] for s in collected) / max(total, 1)
    over = sum(s["over_length_limit"] for s in collected)
    fallbacks = sum(s["fallback_lines"] for s in collected)
    repeats = sum(s["verbatim_repeat_players"] for s in collected)

    def weighted(key: str) -> float:
        return float(sum(s[key] * s["utterances"] for s in collected) / max(total, 1))

    return "\n".join(
        [
            "| 指標 | 値 |",
            "|---|---|",
            f"| 発言数 | {total} |",
            f"| 平均文字数 | {mean_len:.1f} |",
            f"| 人格別の発言長上限超過 | {over} ({over / max(total, 1):.1%}) |",
            f"| フォールバック定型文 | {fallbacks} ({fallbacks / max(total, 1):.1%}) |",
            f"| 同一発言の繰り返し | {repeats} |",
            f"| クロスプレイヤー論点重複率 | {weighted('cross_player_topic_overlap_rate'):.1%} |",
            f"| クロスプレイヤー平均Jaccard | {weighted('cross_player_mean_jaccard'):.3f} |",
            f"| 発言回数ジニ係数 | {weighted('speech_count_gini'):.3f} |",
            f"| 返信率 | {weighted('reply_rate'):.1%} |",
            f"| 未回答質問残存率 | {weighted('unanswered_question_rate'):.1%} |",
            f"| 発言長の分散 | {weighted('length_variance'):.1f} |",
        ]
    )


def _render_judge(summary: dict[str, Any]) -> str:
    if not summary.get("judged_players"):
        return "LLM判定を実行しましたが、有効なスコアが得られませんでした。"

    return "\n".join(
        [
            "> 判定は同一モデルによる自己採点になり得るため、絶対値より"
            "**プロンプト変更前後の相対比較**に使ってください。",
            "",
            "| 指標 | 平均 | 最低 |",
            "|---|---|---|",
            f"| 日本語の自然さ | {summary['naturalness_mean']} | {summary['naturalness_min']} |",
            f"| 人格の維持 | {summary['persona_consistency_mean']} | "
            f"{summary['persona_consistency_min']} |",
            "",
            f"(5段階評価・高いほど良い / 評価対象 {summary['judged_players']} 人)",
        ]
    )


def render_transcript(transcript: GameTranscript) -> str:
    """Readable per-game log, with each speaker's hidden role and persona
    shown inline so Japanese quality and persona drift can be eyeballed."""
    lines = [f"# 対戦記録 (seed={transcript.seed})\n"]
    lines.append("## 配役\n")
    lines.append("| プレイヤー | 役職 | 人格 | 騙り役 |")
    lines.append("|---|---|---|---|")
    fake_roles = transcript.deception.get("fake_role_by_player", {})
    lurkers = set(transcript.deception.get("lurking_player_ids", []))
    for pid, name in transcript.names.items():
        deception = fake_roles.get(pid)
        label = f"偽{deception}" if deception else ("潜伏" if pid in lurkers else "-")
        lines.append(
            f"| {name} (`{pid}`) | {transcript.roles.get(pid, '?')} | "
            f"{transcript.personalities.get(pid, '-')} | {label} |"
        )

    lines.append(f"\n人狼陣営の方針: {transcript.deception.get('wolf_pattern_label', '-')}\n")

    current_day = None
    for u in transcript.utterances:
        if u.day != current_day:
            current_day = u.day
            lines.append(f"\n## {u.day}日目\n")
        speaker = f"{u.player_name}({u.role}/{u.personality})"
        if u.kind == "discussion":
            flag = " [フォールバック]" if u.used_fallback else ""
            lines.append(f"- **{speaker}**{flag}: {u.text}")
        elif u.kind == "vote":
            lines.append(f"- _{speaker} → 投票: {u.target}_ ({u.text})")
        elif u.kind == "night_action":
            lines.append(f"- _{speaker} → 夜行動: {u.target}_ ({u.text})")
        elif u.kind in ("wolf_chat", "freemason_chat"):
            lines.append(f"- `[{u.kind}]` {speaker}: {u.text}")
        elif u.kind == "summary":
            lines.append(f"- _要約_: {u.text}")

    lines.append("\n## システム記録\n")
    names = transcript.names
    for death in transcript.final_state.get("death_records", []):
        pid = death["player_id"]
        lines.append(
            f"- {death['day']}日目: {names.get(pid, pid)}({pid})が"
            f"{death['cause']}で死亡"
        )
    guards = {
        (record["day"], record["target_id"])
        for record in transcript.final_state.get("guard_records", [])
    }
    for attack in transcript.final_state.get("attack_records", []):
        target = attack["target_id"]
        guarded = (attack["day"], target) in guards
        outcome = "護衛され失敗" if guarded else ("成功" if attack["succeeded"] else "失敗")
        lines.append(
            f"- {attack['day']}日目夜: {names.get(attack['wolf_id'], attack['wolf_id'])}が"
            f"{names.get(target, target)}({target})を襲撃 → {outcome}"
        )

    winner = transcript.final_state.get("winner")
    reason = transcript.final_state.get("victory_reason", "")
    lines.append(f"\n## 結果\n\n勝者: {winner} / {reason}")
    return "\n".join(lines) + "\n"
