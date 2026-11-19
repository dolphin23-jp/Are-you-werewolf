"""Renders evaluation output for humans.

Two artefacts, because the questions have different shapes: a summary
report for the measurable metrics, and a readable per-game transcript so
you can judge Japanese quality with your own eyes rather than trusting a
judge model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from app.eval.analyzers import AnalysisResult
from app.eval.grey_reading import GREY_READING_HEADING, render_grey_reading
from app.eval.public_record import describe_record, undeclared_records
from app.eval.reasoning_analyzer import AllyVoteKind, VoteChangeKind, audit_ballots
from app.eval.release_gate import HARD_FAILURE_FIELDS
from app.eval.table_tells import TABLE_TELLS_HEADING, render_table_tells
from app.eval.transcript import GameTranscript
from app.eval.transcript_render import team_label

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Shown at the top of a report written after the spending cap stopped the run.
# Short and fixed, so a workflow or a person can grep for it.
BUDGET_STOP_BANNER = "費用上限に達したため中断"

# What the cap stopped: the games themselves, or the LLM judge that runs after
# every game has finished.
BudgetStopPhase = Literal["game", "judge"]

# Section headings other tooling and the docs refer to by name.
CALL_HEALTH_HEADING = "呼び出し種別ごとの健全性"
HARD_CORRECTNESS_HEADING = "ハード正しさ（すべて0が合格）"

# Providers that never reach a model. Their numbers measure the harness only.
_OFFLINE_PROVIDERS = frozenset({"MockProvider", "ScenarioProvider"})

# Everything the code did *instead of* the model, by the event names the AI
# layer records (`MetricsCollector.record_event`). Listed even when zero: a
# missing row reads as "not measured", a zero row as "did not happen".
EVENT_LABELS: dict[str, str] = {
    "brief_fallback": "簡易契約(発言だけ)への降格",
    "vote_code_resolved": "コードが決めた投票",
    "night_code_resolved": "コードが決めた夜行動",
    "wolf_chat_fallback_line": "狼チャットの定型文",
    "obligation_posted_by_code": "コードが出した公開義務の文",
    "speech_discarded_after_phase_end": "議論の終了後に届いて捨てた発言",
    "discussion_stalled": "連続失敗で止まった議論",
    "designation_question_unanswered": "本指定で出た、答える番の無い質問",
    "night_not_resolved": "夜を解決できなかった(襲撃が無いまま)",
    "night_code_bite": "人狼の襲撃をコードが代わりに決めた夜",
    "summary_without_speaker": "まとめ役が見つからずまとめを飛ばした日",
    "directed_question_target_unnamed": "本文が相手を名指していないので捨てた名指しの質問",
    "directed_question_on_code_line": "コードが書いた行の名指しの質問(捨てた)",
}

# `release_gate.HARD_FAILURE_FIELDS`, in words. Any non-zero count fails the
# release gate, whatever else the run looks like.
HARD_FAILURE_LABELS: dict[str, str] = {
    "private_evidence_exposed_count": "自分だけの根拠を公開発言に出した",
    "team_private_evidence_exposed_count": "陣営内だけの根拠を公開発言に出した",
    "dead_target_selection_count": "死亡者を対象に選んだ",
    "public_fact_flip_count": "同じ公開結果の白黒を言い換えた",
    "missing_required_result_count": "出すべき能力結果を出さなかった",
    "duplicate_required_result_count": "同じ能力結果を重ねて出した",
    "displayed_target_mismatch_count": "表示した対象と決めた対象が違った",
    "stale_evidence_publicly_emitted_count": "訂正済みの古い根拠を公開発言に出した",
    "unexplained_vote_change_count": (
        "卓で名指しした吊り先から、新しい関連の公開情報も引いた論もなく変えた投票"
    ),
    "unplanned_wolf_ally_vote_count": (
        "計画のない人狼の仲間投票（指定に従わない、指定がない、または処刑を決めた票）"
    ),
    "duplicate_night_action_count": "同じ夜の行動の重複",
    "undeclared_record_count": "本人が宣言していないCO・結果が公開の記録に載った",
}

# The validation-log repairs `scripts/live_reasoning_check.py` reports. The
# code caught each before it reached the table, so they are context for the
# hard counts, not part of the verdict.
VALIDATION_LABELS: dict[str, str] = {
    "dead_target_selections": "死亡者・選べない相手を対象にした",
    "public_fact_flips": "公開結果と逆の白黒を書いた",
    "vote_history_misreads": "投票の記録を読み違えた",
    "vote_plan_mismatches": "メモ(v3)・発言(v2)の吊り先と違う相手に投票した",
}

_PRICING_SOURCE_LABELS = {
    "table": "単価表(pricing.py)から自動計算",
    "arguments": "`--price-in` / `--price-out` から計算",
}

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
    "true_role_result_without_record": "本物の占い/霊媒が、実際には出していない結果を公開した扱い",
    "undeclared_spoken_result": "宣言のない結果が発言文から公開の記録に載った",
}


def render_report(
    *,
    games: list[tuple[GameTranscript, AnalysisResult]],
    metrics_summary: dict[str, Any],
    judge_summary: dict[str, Any] | None,
    provider: str,
    model: str,
    engine: str | None = None,
    effort_by_kind: Mapping[str, str] | None = None,
    commit: str | None = None,
    per_game: Sequence[Mapping[str, Any]] | None = None,
    budget: Mapping[str, Any] | None = None,
    budget_stop: str | None = None,
    budget_stop_phase: BudgetStopPhase = "game",
) -> str:
    """The whole run on one page.

    `per_game` is the list `summary.json` carries under `per_game`; without it
    the per-game and hard-correctness sections say they were not measured
    rather than disappear. `budget_stop` is the `BudgetExceeded` message when
    the spending cap ended the run: the report then says so first, because
    every number below it covers only what was played before the stop.

    `budget_stop_phase` says what the cap stopped. "judge" means every game
    was played without reaching it and only the LLM judge afterwards did:
    the report must not then send the reader looking for interrupted games,
    nor tell them to pass a `--judge` they already passed.
    """
    lines: list[str] = []
    add = lines.append
    judge_stopped = budget_stop is not None and budget_stop_phase == "judge"

    add("# AI人狼プレイヤー 評価レポート\n")
    if judge_stopped:
        add(
            f"> **{BUDGET_STOP_BANNER}。** 対戦は上限に届く前にすべて終わり、そのあとの"
            "LLM判定(`--judge`)で上限に達したため、上限に届く判定の呼び出しは送らずに止めました"
            "(それまでに送った判定の呼び出しは費用に含まれます)。"
            "対戦の数値はそろっています。LLM判定の結果はありません(6章)。\n>\n"
            f"> `{budget_stop}`\n"
        )
    elif budget_stop is not None:
        add(
            f"> **{BUDGET_STOP_BANNER}。** 上限に届く呼び出しは送らずに止めました。"
            "以下の数値は止まるまでに行った分だけです。途中で止まった試合の対戦記録は"
            "「途中経過」として残っています(試合ごとの表で勝者が「(中断)」の試合)。\n>\n"
            f"> `{budget_stop}`\n"
        )
    add(f"- プロバイダ: `{provider}` / モデル: `{model}`")
    if engine is not None:
        add(f"- 推理エンジン: `{engine}`")
    if effort_by_kind:
        add("- 思考量(effort): " + " / ".join(f"{k} `{v}`" for k, v in effort_by_kind.items()))
    if commit:
        add(f"- コミット: `{commit}`")
    add(f"- 評価ゲーム数: {len(games)}")
    if budget is not None and budget.get("max_estimated_cost") is not None:
        spent = budget.get("estimated_cost")
        spent_text = f"${spent:.4f}" if isinstance(spent, int | float) else "不明"
        add(f"- 費用上限: ${budget['max_estimated_cost']:.2f}(使用 {spent_text})")
    if provider in _OFFLINE_PROVIDERS:
        add("")
        add(
            "> **注意: モックプロバイダでの実行です。** 定型文を返すだけなので、"
            "日本語の質・人格・推理に関する数値には意味がありません。"
            "ハーネス自体の動作確認用です。"
        )
    add("")

    add("## 1. 動作指標(JSON成功率・応答時間・コスト)\n")
    add(_render_metrics(metrics_summary))

    add(f"\n## 2. {CALL_HEALTH_HEADING}\n")
    add(_render_call_health(metrics_summary, per_game))

    add(f"\n## 3. {HARD_CORRECTNESS_HEADING}\n")
    add(_render_hard_correctness(per_game))
    undeclared = _render_undeclared_records(games)
    if undeclared:
        add(f"\n{undeclared}")
    wolf_chat_plans = _render_wolf_chat_plans(games)
    if wolf_chat_plans:
        add(f"\n{wolf_chat_plans}")
    ballot_lists = _render_ballot_lists(games)
    if ballot_lists:
        add(f"\n{ballot_lists}")

    add("\n## 4. ルールベース検査(役職整合性・矛盾・人狼の連携)\n")
    add(_render_findings(games))

    add("\n## 5. 発言の形式指標\n")
    add(_render_speech_stats(games))

    add("\n## 6. 日本語の自然さ・人格の維持\n")
    if judge_summary is None and judge_stopped:
        add(
            "LLM判定は費用上限に達したため中断しました(結果なし)。判定も含めるには、"
            "`--max-cost` を上げて実行し直してください。\n\n"
            "`transcript-*.md` に読みやすい形式の全発言を出力しているので、"
            "目視での確認も可能です。"
        )
    elif judge_summary is None:
        add(
            "LLM判定は未実行です(`--judge` で有効化)。\n\n"
            "`transcript-*.md` に読みやすい形式の全発言を出力しているので、"
            "目視での確認も可能です。"
        )
    else:
        add(_render_judge(judge_summary))

    transcripts = [transcript for transcript, _analysis in games]
    table_tells = render_table_tells(transcripts)
    if table_tells:
        add(f"\n## 7. {TABLE_TELLS_HEADING}\n")
        add(table_tells)
    grey_reading = render_grey_reading(transcripts)
    if grey_reading:
        add(f"\n## 8. {GREY_READING_HEADING}\n")
        add(grey_reading)

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

    wall = summary.get("total_game_wall_time") or 0.0
    if wall > 0:
        rows.append(f"| 対戦の所要時間(合計) | {format_duration(wall)} |")
        # Every seat is an AI and pacing is off here, so this is how long the
        # models took to produce one public line -- not what a human at the
        # table would wait, which adds pacing and their own pauses.
        rows.append(
            "| 1発言あたりの生成時間(全席AI・ペース配分と人間の間なし) | "
            f"{summary.get('time_per_public_utterance', 0.0):.1f} 秒 |"
        )

    tokens = summary.get("tokens")
    if tokens:
        rows.append(f"| 入力トークン | {tokens['prompt']:,} |")
        rows.append(f"| 出力トークン | {tokens['completion']:,} |")
    else:
        rows.append("| トークン | エンドポイントが usage を返していません |")
    rows.append(f"| **推定コスト** | {_cost_text(summary)} |")

    return "\n".join(rows)


def format_duration(seconds: float) -> str:
    """mm:ss under an hour, h:mm:ss after -- the same shape as the live log."""
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _cost_text(summary: Mapping[str, Any]) -> str:
    """The estimated cost with where its prices came from.

    An unpriced run says so instead of printing 0, which would read as free.
    """
    if not summary.get("tokens"):
        return "不明(usage が無い)"
    if not summary.get("pricing_supplied") or summary.get("estimated_cost") is None:
        return "単価未指定 (`--price-in` / `--price-out` で算出)"
    source = _PRICING_SOURCE_LABELS.get(str(summary.get("pricing_source")), "出典不明")
    return f"${summary['estimated_cost']:.4f}({source})"


def _short_cost(metrics: Mapping[str, Any] | None) -> str:
    if not metrics or not metrics.get("pricing_supplied"):
        return "—"
    cost = metrics.get("estimated_cost")
    return f"${cost:.2f}" if isinstance(cost, int | float) else "—"


def _render_call_health(
    summary: Mapping[str, Any], per_game: Sequence[Mapping[str, Any]] | None
) -> str:
    """Whether the run was carried by the model or by the safety nets.

    A 100% success rate says little on its own: a refused turn re-served by a
    fallback model, a turn degraded to the brief contract, or a vote the code
    resolved all count as successes. Each of those is counted here by name.
    """
    out: list[str] = []
    if not summary or summary.get("total_calls", 0) == 0:
        out.append("(LLM呼び出しの記録がありません)")
    else:
        by_schema = summary.get("by_schema") or {}
        out.append("| 種別 | 回数 | 成功率 | 平均応答 | effort |")
        out.append("|---|---|---|---|---|")
        for schema, data in by_schema.items():
            out.append(
                f"| {schema} | {data['calls']} | {data['success_rate']:.1%} | "
                f"{data['mean_latency_seconds']:.1f} 秒 | {data.get('effort') or '—'} |"
            )

    events: Mapping[str, int] = summary.get("events") or {}
    out.append("\nコードによる代行(0 が理想):\n")
    out.append("| 出来事 | 件数 |")
    out.append("|---|---|")
    for name, label in EVENT_LABELS.items():
        out.append(f"| {label} (`{name}`) | {events.get(name, 0)} |")
    for name, count in events.items():
        if name not in EVENT_LABELS:
            out.append(f"| `{name}` | {count} |")

    if summary and summary.get("total_calls", 0):
        refusals: Mapping[str, int] = summary.get("refusals") or {}
        served: Mapping[str, int] = summary.get("served_model_counts") or {}
        out.append("\n| 指標 | 値 |")
        out.append("|---|---|")
        out.append(
            "| 拒否 | "
            + (
                ", ".join(f"{category} {count}" for category, count in refusals.items())
                if refusals
                else "0"
            )
            + " |"
        )
        out.append(
            f"| 途中切れ(出力上限) | {summary.get('truncated_calls', 0)}"
            f"(失敗のうち {summary.get('truncated_failure_rate', 0.0):.0%}) |"
        )
        out.append(
            "| 応答したモデル | "
            + (
                " / ".join(f"`{model}` {count}" for model, count in served.items())
                if served
                else "記録なし"
            )
            + " |"
        )
        fallback_calls = summary.get("fallback_served_calls", 0)
        out.append(f"| フォールバックのモデルが答えた呼び出し | {fallback_calls} |")
        out.append(f"| キャッシュ読みの割合 | {summary.get('cache_read_share', 0.0):.1%} |")
        out.append(f"| 推定コスト | {_cost_text(summary)} |")

        errors = summary.get("errors") or []
        if errors:
            out.append("\n主なエラー:\n")
            for entry in errors:
                out.append(f"- `{entry['error']}` ({entry['count']}件)")

    out.append("\n試合ごと:\n")
    if not per_game:
        out.append("(試合ごとの記録はありません)")
    else:
        out.append("| seed | 勝者 | 発言/日 | 所要 | 費用 | ハード |")
        out.append("|---|---|---|---|---|---|")
        for game in per_game:
            days = int(game.get("days") or 0)
            utterances = int(game.get("public_utterances") or 0)
            per_day = f"{utterances / days:.1f}" if days else "—"
            winner = (
                team_label(game["winner"])
                if game.get("winner")
                else ("(中断)" if not game.get("finished") else "—")
            )
            hard = game.get("hard_failures")
            hard_text = "—" if hard is None else str(sum(int(v) for v in hard.values()))
            out.append(
                f"| {game.get('seed')} | {winner} | {per_day} | "
                f"{format_duration(float(game.get('wall_time_seconds') or 0.0))} | "
                f"{_short_cost(game.get('metrics'))} | {hard_text} |"
            )
        out.append(
            "\n「ハード」は次の節の項目の合計(0 が合格)。`—` は計測していない"
            "(legacy エンジン)という意味です。"
        )
    return "\n".join(out)


def _render_hard_correctness(per_game: Sequence[Mapping[str, Any]] | None) -> str:
    """The release gate's hard failures, per game. Any non-zero fails the run.

    Computed from the transcript's audit records the same way the release
    gate reads them, so a number here and a gate verdict can never disagree.
    """
    measured = [game for game in per_game or () if game.get("hard_failures") is not None]
    if not measured:
        return (
            "計測していません。監査記録(判断・公開・夜行動)を残すのは v2 と v3 だけで、"
            "legacy にはありません。"
        )

    out = ["| 項目 | 合計 | 0 でない試合(seed) |", "|---|---|---|"]
    failing: list[str] = []
    for field in HARD_FAILURE_FIELDS:
        total = sum(int(game["hard_failures"].get(field, 0)) for game in measured)
        seeds = [
            str(game.get("seed"))
            for game in measured
            if int(game["hard_failures"].get(field, 0)) > 0
        ]
        if total:
            failing.append(field)
        mark = f"**{total}**" if total else "0"
        out.append(
            f"| {HARD_FAILURE_LABELS.get(field, field)} | {mark} | {', '.join(seeds) or '—'} |"
        )
    unfinished = [str(game.get("seed")) for game in measured if not game.get("finished", True)]
    verdict = (
        f"**不合格**: {len(failing)}項目が 0 ではありません(`{'`, `'.join(failing)}`)。"
        if failing
        else "**合格**: すべて 0 です。"
    )
    out.append(f"\n{verdict}")

    repairs = [game.get("validation") or {} for game in measured]
    if any(repairs):
        out.append(
            "\n参考: コードが差し戻した誤り(公開記録には届いていないので合否には数えない。"
            "多ければ、正しさをモデルではなく検証が支えているということ)\n"
        )
        out.append("| 項目 | 合計 |")
        out.append("|---|---|")
        for key, label in VALIDATION_LABELS.items():
            out.append(f"| {label} | {sum(int(r.get(key, 0)) for r in repairs)} |")
    if unfinished:
        out.append(
            f"\n途中で止まった試合(seed {', '.join(unfinished)})は、"
            "止まるまでの分だけを数えています。"
        )
    return "\n".join(out)


UNDECLARED_RECORDS_HEADING = "### 本人が宣言していないのに公開の記録に載ったCO・結果（引用つき）"


def _render_undeclared_records(games: list[tuple[GameTranscript, AnalysisResult]]) -> str:
    """Every CO, result and partner naming on the record that its own line did not declare.

    The gate counts the ones that changed what the record says
    (`undeclared_record_count`); one that only repeated what already stood is
    listed after them and not counted. Each is quoted with the line it was
    registered from, so a reader checks the reading rather than a count.
    """
    audited = [
        (transcript, undeclared_records(transcript))
        for transcript, _analysis in games
        if (transcript.final_state or {}).get("chat_log")
    ]
    if not audited:
        return ""
    counted: list[str] = []
    repeated: list[str] = []
    for transcript, records in audited:
        for record in records:
            row = (
                f"| {_seed(transcript)} | {record.day} | {record.source_message_id or '—'} | "
                f"{describe_record(record, transcript.names)} | {_quote(record.text)} |"
            )
            (counted if record.changed_record else repeated).append(row)
    out = [
        UNDECLARED_RECORDS_HEADING,
        "",
        "公開の記録（COの一覧・公開結果・相方の名指し）に載ったもののうち、載った発言そのものが"
        "それを宣言していないものです。宣言とは、その発言の構造化出力（`public_claim_role`・"
        "`public_results`）か、本人のこととして言い切った本文（「占い師CO。」「霊媒COします」"
        "「俺は本物の共有だ」「0日目夜はミオを占って白」）です。他人のCOの受け答え"
        "（「共有CO二人は了解」）や、他人の結果への言及（「リク黒が外れたのは…」）は宣言では"
        "ありません。卓は記録を事実として読むので、合否に数えます"
        "（`undeclared_record_count`）。",
        "",
    ]
    if counted:
        out += [
            "| seed | 日 | 発言 | 記録 | 記録の元になった発言 |",
            "|---|---|---|---|---|",
            *counted,
        ]
    else:
        out.append("(ありません)")
    if repeated:
        out += [
            "",
            "すでに記録にあったものを繰り返しただけで、記録を変えなかったもの（合否に数えません）:",
            "",
            "| seed | 日 | 発言 | 記録 | 記録の元になった発言 |",
            "|---|---|---|---|---|",
            *repeated,
        ]
    return "\n".join(out)


WOLF_CHAT_PLANS_HEADING = "### 人狼チャットの文から「計画あり」と読んだ仲間への投票"


def _render_wolf_chat_plans(games: list[tuple[GameTranscript, AnalysisResult]]) -> str:
    """Every ballot on a partner that counted as planned because of a wolf-chat line.

    Whether a line planned that ballot is read from its wording by a pattern
    rule (`coordinator._wolf_chat_ballot_plan`), not established as a fact. It
    is conservative, and a misreading still makes a betrayal pass the gate as
    planned -- so each such ballot is listed with the line it rests on, and a
    reader checks it instead of trusting a count. A ballot planned by a public
    mark rests on the table's own record and is not listed.
    """
    rows: list[str] = []
    for transcript, _analysis in games:
        names = transcript.names
        for plan in transcript.wolf_ally_vote_plans:
            if plan.basis != "wolf_chat":
                continue
            voter = names.get(plan.voter_id, plan.voter_id)
            target = names.get(plan.target_id, plan.target_id)
            sentence = plan.story_reason.replace("|", "｜").replace("\n", " ")
            round_note = f"(決選{plan.round_number}回目)" if plan.round_number > 1 else ""
            rows.append(
                f"| {transcript.seed if transcript.seed is not None else '—'} "
                f"| {plan.day}{round_note} | {voter} → {target} | {sentence} |"
            )
    if not rows:
        return ""
    return "\n".join(
        [
            WOLF_CHAT_PLANS_HEADING,
            "",
            "上の「計画のない人狼の仲間投票」に数えなかった投票のうち、人狼チャットの1文を"
            "コードが「仲間に投票する計画」と読んだものです。読み取りは文の形だけによるので、"
            "誤読なら実際には計画のない投票です。引用した文を確かめてください"
            "(公開の黒や偽の確定を根拠にした投票は、卓の記録に基づくので載せていません)。",
            "",
            "| seed | 日 | 投票者 → 仲間 | 計画と読んだ文 |",
            "|---|---|---|---|",
            *rows,
        ]
    )


BALLOT_LISTS_HEADING = "### 合否に数えない投票（引用つき）"
UNEXPLAINED_LIST_HEADING = "### 合否に数えた、理由の見えない変更（引用つき）"

# The non-hard changes with a cause the table could see, in words.
_JUSTIFIED_LABELS: dict[VoteChangeKind, str] = {
    VoteChangeKind.PUBLIC_ARGUMENT: "後の発言者の論を理由に挙げた",
    VoteChangeKind.NEW_PUBLIC_EVIDENCE: "発言後に、どちらかの吊り先に関わる結果が出た",
    VoteChangeKind.CORRECTION_ACCEPTED: "発言後に事実の訂正があった",
    VoteChangeKind.TARGET_BECAME_INVALID: "吊り先が死亡した",
    VoteChangeKind.RUNOFF_RESTRICTION: "決選の候補から外れた",
    VoteChangeKind.GREY_RANDOM_POLICY: (
        "発言後に進行役がグレランを宣言し、グレー外からグレーへ移した"
    ),
    VoteChangeKind.FINAL_DESIGNATION: "発言後の進行役の最後の指定どおりに入れた",
}


def _quote(text: str) -> str:
    """A line as a table cell: whole, on one line, without breaking the table."""
    flat = " / ".join(part.strip() for part in text.splitlines() if part.strip())
    return flat.replace("|", "｜") or "—"


def _render_ballot_lists(games: list[tuple[GameTranscript, AnalysisResult]]) -> str:
    """Every ballot the gate fails or does not judge, with the words it rests on.

    The gate measures a seat's public stance against its vote. A ballot with
    no basis was compared with nothing; a change off a memo target the seat
    never said aloud, or said only in its own result about that seat or in a
    question to it, left no stance the table saw; a change with a cause the
    table could see is justified. None of them fails the gate, and each is
    listed so a reader can check the reading rather than trust a count. The
    unexplained changes the gate does fail come first, quoted the same way.
    """
    audited = [
        (transcript, audit_ballots(transcript))
        for transcript, _analysis in games
        if any(audit.vote_target is not None for audit in transcript.decision_audits)
    ]
    if not audited:
        return ""
    out = [
        BALLOT_LISTS_HEADING,
        "",
        "票を比べる基準は、その日の最後の手番のメモの吊り先です。ハードな失敗になるのは、"
        "その名前を自分の公開発言で言っていて、発言後にどちらの吊り先にも関わる新しい"
        "公開情報がなく、理由が後の発言者の論を挙げておらず、進行役のグレラン宣言で"
        "グレー外からグレーへ移したのでも、発言後の進行役の最後の指定どおりに入れたのでもない"
        "変更だけです。",
        "",
        "比較できた票／全票（その日の手番がない票と、メモの吊り先が空の票は比較できません）:",
        "",
        "| seed | 比較できた票／全票 | 比較できない | 卓で言っていない吊り先からの変更 "
        "| 結果の発表・本人への質問でだけ触れた吊り先からの変更 |",
        "|---|---|---|---|---|",
    ]
    for transcript, verdicts in audited:
        no_basis = sum(v.kind is VoteChangeKind.NO_BASIS for v in verdicts)
        unspoken = sum(v.kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN for v in verdicts)
        excused = sum(
            v.kind is VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED for v in verdicts
        )
        out.append(
            f"| {_seed(transcript)} | {len(verdicts) - no_basis} / {len(verdicts)} | "
            f"{no_basis} | {unspoken} | {excused} |"
        )

    unexplained_rows: list[str] = []
    unspoken_rows: list[str] = []
    excused_rows: list[str] = []
    justified_rows: list[str] = []
    designated_rows: list[str] = []
    mismatch_rows: list[str] = []
    no_basis_lines: list[str] = []
    for transcript, verdicts in audited:
        names = transcript.names
        no_basis_by_day: dict[int, list[str]] = {}
        for verdict in verdicts:
            voter = names.get(verdict.player_id, verdict.player_id)
            basis = names.get(verdict.basis or "", verdict.basis or "—")
            target = names.get(verdict.vote_target or "", verdict.vote_target or "—")
            day = _day_label(verdict.day, verdict.round_number)
            if verdict.reason_names_other:
                others = "、".join(names.get(seat, seat) for seat in verdict.reason_names_other)
                mismatch_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} → {target} | {others} | "
                    f"{_quote(verdict.reason)} |"
                )
            if verdict.ally_kind is AllyVoteKind.DESIGNATED:
                tally = "・".join(
                    f"{names.get(seat, seat)} {count}票" for seat, count in verdict.tally
                )
                designated_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} → {target} | "
                    f"{_quote(verdict.designation_line)} | {tally} | {_quote(verdict.reason)} |"
                )
            if verdict.kind is VoteChangeKind.UNEXPLAINED:
                last_line = verdict.own_lines[-1] if verdict.own_lines else ""
                unexplained_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} | {basis} → {target} | "
                    f"{_quote(last_line)} | {_quote(verdict.reason)} |"
                )
            elif verdict.kind is VoteChangeKind.MEMO_TARGET_UNSPOKEN:
                last_line = verdict.own_lines[-1] if verdict.own_lines else ""
                unspoken_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} | {basis} → {target} | "
                    f"{_quote(last_line)} | {_quote(verdict.reason)} |"
                )
            elif verdict.kind is VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED:
                sentences = " ／ ".join(
                    f"「{sentence}」（{excuse}）" for sentence, excuse in verdict.excused_sentences
                )
                excused_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} | {basis} → {target} | "
                    f"{_quote(sentences)} | {_quote(verdict.reason)} |"
                )
            elif verdict.kind in _JUSTIFIED_LABELS:
                justified_rows.append(
                    f"| {_seed(transcript)} | {day} | {voter} | {basis} → {target} | "
                    f"{_JUSTIFIED_LABELS[verdict.kind]} | {_quote(verdict.why)} |"
                )
            elif verdict.kind is VoteChangeKind.NO_BASIS:
                runoff = (
                    f"、決選{verdict.round_number}回目" if verdict.round_number > 1 else ""
                )
                no_basis_by_day.setdefault(verdict.day, []).append(
                    f"{voter}→{target}（{verdict.why}{runoff}）"
                )
        for day_number, items in sorted(no_basis_by_day.items()):
            no_basis_lines.append(
                f"- seed {_seed(transcript)} {day_number}日目: {'、'.join(items)}"
            )

    # The failures first, under their own heading, quoted like the rest: a
    # count alone asked the reader to trust the reading the lists below let
    # them check (run 14 counted seven and quoted none).
    failed = [
        "",
        UNEXPLAINED_LIST_HEADING,
        "",
        "その日の自分の発言でメモの吊り先の名前を出し、発言後に関わる公開情報も、"
        "理由が挙げる後の発言者の論も、グレー外からグレーへ移す進行役のグレラン宣言も、"
        "票の相手を名指した進行役の最後の指定も"
        "ないまま、別の相手に入れた票です（`unexplained_vote_change_count`）。",
        "",
    ]
    if unexplained_rows:
        failed += [
            "| seed | 日 | 投票者 | メモの吊り先 → 票 | その日の自分の最後の発言 | 票の理由 |",
            "|---|---|---|---|---|---|",
            *unexplained_rows,
        ]
    else:
        failed.append("(ありません)")
    out = [*failed[1:], "", *out]
    out += [
        "",
        "#### 卓で言っていないメモの吊り先からの変更（`memo_target_unspoken`）",
        "",
        "メモの吊り先がその日の自分の発言に一度も出ていないまま、別の相手に入れた票です。"
        "卓には離れる立場が見えていないので、合否に数えません。",
        "",
    ]
    if unspoken_rows:
        out += [
            "| seed | 日 | 投票者 | メモの吊り先 → 票 | その日の自分の最後の発言 | 票の理由 |",
            "|---|---|---|---|---|---|",
            *unspoken_rows,
        ]
    else:
        out.append("(ありません)")
    out += [
        "",
        "#### 結果の発表・本人への質問でだけ触れたメモの吊り先からの変更"
        "（`memo_target_only_reported_or_asked`）",
        "",
        "メモの吊り先の名前がその日の自分の発言に出ているが、どの文も、その席についての"
        "自分の能力結果の発表（結果が登録された発言の中で、名前の後に白・黒などの判定を"
        "述べ、吊り・投票・指定の語を含まない文）か、その席への質問（問いかけ・回答待ち・"
        "回答の要求で、名前が呼びかけか回答を求められる側として出る文）だったものです。"
        "文の形だけで読み、意図は読みません。卓には離れる立場が見えていないので、"
        "合否に数えません。読み違いがないか確かめられるよう、該当する文をすべて引きます。",
        "",
    ]
    if excused_rows:
        out += [
            "| seed | 日 | 投票者 | メモの吊り先 → 票 | 名前が出た文（読み） | 票の理由 |",
            "|---|---|---|---|---|---|",
            *excused_rows,
        ]
    else:
        out.append("(ありません)")
    out += [
        "",
        "#### 本指定に従った、結果を変えない仲間への投票",
        "",
        "人狼が計画なしに仲間へ入れた票のうち、その日の最後の指定どおりで、この票を除いて"
        "数え直しても処刑される人が変わらないものです。合否に数えません"
        "（指定と違う票、指定のない日の票、処刑を決めた票はハードな失敗のままです）。",
        "",
    ]
    if designated_rows:
        out += [
            "| seed | 日 | 投票者 → 仲間 | その日の最後の指定 | 集計 | 票の理由 |",
            "|---|---|---|---|---|---|",
            *designated_rows,
        ]
    else:
        out.append("(ありません)")
    out += [
        "",
        "#### 理由が別の相手への投票を述べている票",
        "",
        "理由に「Xに入れる／Xに投票」とあり、X が票の相手でないものです（ほかの席の票を"
        "述べた文は除いています）。票はモデルが決めたもので、理由から決め直すことは"
        "しません。合否に数えません。",
        "",
    ]
    if mismatch_rows:
        out += [
            "| seed | 日 | 投票者 → 票 | 理由が挙げた相手 | 票の理由 |",
            "|---|---|---|---|---|",
            *mismatch_rows,
        ]
    else:
        out.append("(ありません)")
    out += ["", "#### 卓に見える理由のある変更", ""]
    if justified_rows:
        out += [
            "| seed | 日 | 投票者 | 基準 → 票 | 理由の区分 | 記録 |",
            "|---|---|---|---|---|---|",
            *justified_rows,
        ]
    else:
        out.append("(ありません)")
    out += [
        "",
        "#### 比較できない票（`no_basis`）",
        "",
        "その日に手番がなかった席の票と、メモの吊り先が空のまま投じた票です。"
        "「変更なし」ではありません。",
        "",
        *(no_basis_lines or ["(ありません)"]),
    ]
    return "\n".join(out)


def _seed(transcript: GameTranscript) -> str:
    return str(transcript.seed) if transcript.seed is not None else "—"


def _day_label(day: int, round_number: int) -> str:
    return f"{day}(決選{round_number}回目)" if round_number > 1 else str(day)


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
    # Lengths are over the lines with a part the model wrote (all of them in a
    # transcript recorded before code-written prefixes were told apart).
    measured = sum(s.get("length_lines", s["utterances"]) for s in collected)
    mean_len = (
        sum(s["mean_length"] * s.get("length_lines", s["utterances"]) for s in collected)
        / max(measured, 1)
    )
    over = sum(s["over_length_limit"] for s in collected)
    fallbacks = sum(s["fallback_lines"] for s in collected)
    repeats = sum(s["verbatim_repeat_players"] for s in collected)

    def weighted(key: str) -> float:
        return float(sum(s[key] * s["utterances"] for s in collected) / max(total, 1))

    def weighted_or_dash(key: str) -> str:
        """A rate some transcripts cannot give (recorded before the field existed)."""
        if any(s.get(key) is None for s in collected):
            return "—"
        return f"{weighted(key):.1%}"

    stance = _stance_rows(games)
    examples = _adjacent_example_lines(collected)
    return "\n".join(
        [
            "| 指標 | 値 |",
            "|---|---|",
            f"| 発言数 | {total} |",
            f"| 平均文字数(モデルが書いた部分) | {mean_len:.1f} |",
            f"| 発言長上限超過(モデルが書いた部分) | {over} ({over / max(measured, 1):.1%}) |",
            f"| フォールバック定型文 | {fallbacks} ({fallbacks / max(total, 1):.1%}) |",
            f"| 同一発言の繰り返し | {repeats} |",
            f"| クロスプレイヤー論点重複率 | {weighted('cross_player_topic_overlap_rate'):.1%} |",
            f"| クロスプレイヤー平均Jaccard | {weighted('cross_player_mean_jaccard'):.3f} |",
            f"| 直前の発言との重複率(Jaccard 0.25以上) | "
            f"{weighted('adjacent_overlap_rate'):.1%} |",
            f"| 直前の発言との平均Jaccard | {weighted('adjacent_mean_jaccard'):.3f} |",
            f"| 発言回数ジニ係数 | {weighted('speech_count_gini'):.3f} |",
            f"| 返信率(返信先の欄が埋まった公開発言の割合) | {weighted('reply_rate'):.1%} |",
            f"| 未回答質問残存率 | {weighted('unanswered_question_rate'):.1%} |",
            f"| うち相手に手番が来なかった質問 | "
            f"{weighted_or_dash('unanswered_never_served_rate')} |",
            f"| うち手番は来たが返信リンクのない質問(本文で答えたか、答えなかった) | "
            f"{weighted_or_dash('unanswered_served_without_reply_link_rate')} |",
            f"| 発言長の分散 | {weighted('length_variance'):.1f} |",
            f"| 定型句を含むAI発言率(プロンプト由来とコードが書いた文の合計) | "
            f"{weighted('template_phrase_rate'):.1%} |",
            f"| うちプロンプトの語彙 | {weighted('prompt_template_phrase_rate'):.1%} |",
            f"| うちコードが文頭に足したCO・結果の文 | "
            f"{sum(s['code_fact_sentence_lines'] for s in collected)}行 "
            f"({weighted('code_fact_sentence_rate'):.1%}) |",
            f"| ID表記(pN)を含むAI発言率 | {weighted('id_mention_rate'):.1%} |",
            f"| 1日あたり公開発言数 | {weighted('public_messages_per_day'):.1f} |",
            f"| 公開発言の平均文字数(AI発言はモデルが書いた部分) | "
            f"{weighted('public_mean_length'):.1f} |",
            *stance,
            *examples,
        ]
    )


def _adjacent_example_lines(collected: list[dict[str, Any]]) -> list[str]:
    """The strongest echoes, so the overlap rate can be checked against the lines."""
    pairs = sorted(
        (example for stats in collected for example in stats.get("adjacent_overlap_examples", [])),
        key=lambda example: example["jaccard"],
        reverse=True,
    )[:3]
    if not pairs:
        return []
    lines = ["", "直前の発言との重複が大きい組(先頭80字):", ""]
    for pair in pairs:
        lines.append(
            f"- {pair['day']}日目 `{pair['first']}` → `{pair['second']}` "
            f"(Jaccard {pair['jaccard']:.2f})"
        )
        lines.append(f"  > {pair['first_text']}")
        lines.append(f"  > {pair['second_text']}")
    return lines


def _stance_rows(games: list[tuple[GameTranscript, AnalysisResult]]) -> list[str]:
    """Rows for how often a seat moved its candidate, summed over the games.

    The denominator is the pairs of turns where a change *could* be seen: both
    turns state a candidate and the old one is still alive. A pair with a blank
    side is reported apart, because "did not move" is not observable there --
    the first Claude game's 19 "acknowledged without change" were 11 such pairs,
    3 misread words and 5 reasoned partial concessions.
    """
    collected = [s for _t, a in games if (s := a.stats.get("stance")) is not None]
    if not collected:
        return []

    def total(key: str) -> int:
        return sum(s.get(key) or 0 for s in collected)

    turns = total("follow_up_turns")
    moved = total("voluntary_changes")
    replying = (
        "—"
        if any(s.get("voluntary_changes_replying") is None for s in collected)
        else str(total("voluntary_changes_replying"))
    )
    return [
        "| 吊り先の自発的な変更(両方の手番に吊り先があり、元の吊り先が生きていた対) | "
        f"{moved} / {total('observable_follow_ups')}対(全後続手番 {turns}) |",
        f"| うち同じ日のうちの変更 | {total('voluntary_changes_same_day')} |",
        f"| うち相手の発言に返信しての変更 | {replying} |",
        f"| うち認める言葉(確かに・認める・撤回…)を伴う変更 | "
        f"{total('voluntary_changes_acknowledged')} |",
        f"| 吊り先の死亡による変更 | {total('forced_changes')} |",
        f"| 認める言葉を言いながら同じ吊り先のまま | {total('acknowledged_without_change')} |",
        f"| うち逆接と理由を添えた部分的な譲歩 | "
        f"{total('acknowledged_without_change_with_reason')} |",
        f"| うち理由のない空約束 | {total('acknowledged_without_change_no_reason')} |",
        f"| 吊り先が片方でも空欄の対(変化を観測できない) | "
        f"{total('target_unstated_follow_ups')} |",
        f"| うち認める言葉を伴う | {total('target_unstated_acknowledged')} |",
        f"| うち空欄から吊り先を置いた(認める言葉を伴う) | "
        f"{total('target_adopted_acknowledged')} |",
    ]


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


def render_transcript(transcript: GameTranscript, *, partial: bool = False) -> str:
    """Readable per-game log, with each speaker's hidden role and persona
    shown inline so Japanese quality and persona drift can be eyeballed.

    Written in full -- private notes, vote reasons, night results, audits --
    by `app.eval.transcript_render`; nothing is shortened.

    `partial` marks a snapshot taken while the game is still being played, so a
    run that was cancelled or timed out leaves a file that says what it is
    instead of announcing a winner of None."""
    from app.eval.transcript_render import render_transcript_full

    return render_transcript_full(transcript, partial=partial)
