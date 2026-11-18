"""`render_report`: the sections a live run is judged by.

Built from hand-written summaries rather than a played game, so each test
pins one reading of the numbers -- what a zero means, what "not measured"
looks like, which source a cost came from.
"""

from __future__ import annotations

from typing import Any

from app.eval.release_gate import HARD_FAILURE_FIELDS
from app.eval.report import (
    BUDGET_STOP_BANNER,
    CALL_HEALTH_HEADING,
    EVENT_LABELS,
    HARD_CORRECTNESS_HEADING,
    HARD_FAILURE_LABELS,
    format_duration,
    render_report,
)


def _metrics(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_calls": 10,
        "success_rate": 0.9,
        "strict_schema_rate": 0.9,
        "retry_calls": 1,
        "discussion_skips": 0,
        "discussion_generation_attempts": 4,
        "discussion_skip_rate": 0.0,
        "latency_seconds": {"mean": 2.0, "p50": 1.5, "p95": 4.0, "max": 5.0},
        "total_game_wall_time": 600.0,
        "time_per_public_utterance": 12.5,
        "tokens": {"prompt": 50_000, "completion": 5_000, "total": 55_000},
        "estimated_cost": 0.3,
        "pricing_supplied": True,
        "pricing_source": "table",
        "by_schema": {
            "ChatDiscussionOutput": {
                "calls": 4,
                "success_rate": 1.0,
                "mean_latency_seconds": 3.0,
                "effort": "low",
            },
            "VoteOutput": {
                "calls": 6,
                "success_rate": 5 / 6,
                "mean_latency_seconds": 1.2,
                "effort": "medium",
            },
        },
        "events": {"vote_code_resolved": 2, "brand_new_event": 1},
        "refusals": {"cyber": 1},
        "truncated_calls": 0,
        "truncated_failure_rate": 0.0,
        "served_model_counts": {"claude-opus-5-5": 9, "claude-opus-5": 1},
        "fallback_served_calls": 1,
        "cache_read_share": 0.25,
        "errors": [{"error": "model refused (category=cyber)", "count": 1}],
    }
    summary.update(overrides)
    return summary


def _game(seed: int, *, hard: dict[str, int] | None, **overrides: Any) -> dict[str, Any]:
    game: dict[str, Any] = {
        "seed": seed,
        "winner": "village",
        "finished": True,
        "wall_time_seconds": 3725.0,
        "public_utterances": 60,
        "days": 5,
        "metrics": {"pricing_supplied": True, "estimated_cost": 7.5},
        "hard_failures": hard,
        "validation": {"dead_target_selections": 2},
    }
    game.update(overrides)
    return game


def _zero() -> dict[str, int]:
    return {field: 0 for field in HARD_FAILURE_FIELDS}


def _render(**overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "games": [],
        "metrics_summary": _metrics(),
        "judge_summary": None,
        "provider": "AnthropicClaudeProvider",
        "model": "claude-opus-5-5",
    }
    kwargs.update(overrides)
    return render_report(**kwargs)


def _section(report: str, heading: str) -> str:
    return report.split(heading, 1)[1].split("\n## ", 1)[0]


def test_the_header_names_what_was_measured() -> None:
    report = _render(
        engine="v3",
        effort_by_kind={"discussion": "low", "vote": "medium"},
        commit="abc123",
        budget={"max_estimated_cost": 15.0, "estimated_cost": 3.25},
    )

    head = report.split("## 1.")[0]
    assert "`claude-opus-5-5`" in head
    assert "`v3`" in head
    assert "discussion `low`" in head and "vote `medium`" in head
    assert "`abc123`" in head
    assert "$15.00" in head and "$3.2500" in head
    assert "モックプロバイダ" not in head
    assert BUDGET_STOP_BANNER not in report


def test_a_budget_stop_is_the_first_thing_the_report_says() -> None:
    report = _render(budget_stop="estimated cost budget would be exceeded (15.1 > 15.0)")

    assert report.index(BUDGET_STOP_BANNER) < report.index("- プロバイダ")
    assert "15.1 > 15.0" in report
    # A stop during the games keeps saying where the partial games are.
    assert "途中経過" in report.split("- プロバイダ")[0]


def test_a_stop_in_the_judge_says_the_games_finished_and_the_judge_did_not() -> None:
    report = _render(
        budget_stop="estimated cost budget would be exceeded (20.1 > 20.0)",
        budget_stop_phase="judge",
    )

    head = report.split("- プロバイダ")[0]
    assert BUDGET_STOP_BANNER in head  # still greppable by workflows
    assert "20.1 > 20.0" in head
    # No game was cut short, so the reader is not sent looking for one.
    assert "途中経過" not in head
    assert "LLM判定" in head
    judge = _section(report, "## 6.")
    assert "費用上限" in judge
    # The flag was passed; telling the owner to pass it would be wrong.
    assert "`--judge` で有効化" not in judge


def test_without_a_judge_stop_section_6_still_points_at_the_flag() -> None:
    assert "`--judge` で有効化" in _section(_render(), "## 6.")


def test_offline_doubles_are_flagged_as_such() -> None:
    assert "モックプロバイダ" in _render(provider="MockProvider")
    assert "モックプロバイダ" in _render(provider="ScenarioProvider")


def test_the_per_utterance_time_says_it_is_the_all_ai_harness_figure() -> None:
    report = _render()

    row = next(line for line in report.splitlines() if "1発言あたりの生成時間" in line)
    assert "全席AI" in row and "ペース配分" in row and "12.5 秒" in row
    assert "10:00" in report  # total wall time


def test_cost_says_where_its_prices_came_from() -> None:
    assert "$0.3000(単価表" in _render()
    assert "--price-in" in _render(metrics_summary=_metrics(pricing_source="arguments"))
    unpriced = _render(metrics_summary=_metrics(pricing_supplied=False, pricing_source=None))
    assert "単価未指定" in unpriced
    no_usage = _render(metrics_summary=_metrics(tokens=None))
    assert "usage が無い" in no_usage


def test_call_health_counts_every_safety_net_by_name() -> None:
    section = _section(_render(per_game=[_game(1, hard=_zero())]), CALL_HEALTH_HEADING)

    assert "| ChatDiscussionOutput | 4 | 100.0% | 3.0 秒 | low |" in section
    assert "| VoteOutput | 6 | 83.3% | 1.2 秒 | medium |" in section
    # Every known event is listed, zero included, and an unknown one too.
    for name in EVENT_LABELS:
        assert f"(`{name}`)" in section
    assert "(`vote_code_resolved`) | 2 |" in section
    assert "(`brief_fallback`) | 0 |" in section
    assert "`brand_new_event` | 1 |" in section
    assert "cyber 1" in section
    assert "`claude-opus-5` 1" in section
    assert "フォールバックのモデルが答えた呼び出し | 1 |" in section
    assert "25.0%" in section  # cache read share
    assert "model refused (category=cyber)" in section


def test_the_per_game_table_reads_on_a_phone() -> None:
    games = [
        _game(1, hard=_zero()),
        _game(2, hard={**_zero(), "dead_target_selection_count": 2}, winner=None, finished=False),
        _game(3, hard=None, metrics={"pricing_supplied": False}),
    ]
    section = _section(_render(per_game=games), CALL_HEALTH_HEADING)

    table = [line for line in section.splitlines() if line.startswith("| ") and "|" in line[2:]]
    header = next(line for line in table if line.startswith("| seed"))
    assert header.count("|") - 1 <= 6  # few columns: readable on a phone
    assert "| 1 | 村人陣営 | 12.0 | 1:02:05 | $7.50 | 0 |" in section
    assert "| 2 | (中断) | 12.0 | 1:02:05 | $7.50 | 2 |" in section
    assert "| 3 | 村人陣営 | 12.0 | 1:02:05 | — | — |" in section


def test_hard_correctness_passes_only_when_every_count_is_zero() -> None:
    passing = _section(_render(per_game=[_game(1, hard=_zero())]), HARD_CORRECTNESS_HEADING)
    failing = _section(
        _render(
            per_game=[
                _game(1, hard=_zero()),
                _game(4, hard={**_zero(), "unplanned_wolf_ally_vote_count": 1}),
            ]
        ),
        HARD_CORRECTNESS_HEADING,
    )

    for field in HARD_FAILURE_FIELDS:
        assert HARD_FAILURE_LABELS[field] in passing
    assert "**合格**" in passing and "不合格" not in passing
    assert "**不合格**" in failing and "`unplanned_wolf_ally_vote_count`" in failing
    row = next(
        line
        for line in failing.splitlines()
        if HARD_FAILURE_LABELS["unplanned_wolf_ally_vote_count"] in line
    )
    assert row.endswith("| **1** | 4 |")
    # The validation repairs are context, not part of the verdict.
    assert "差し戻した誤り" in passing and "| 2 |" in passing


def test_hard_correctness_is_not_measured_for_legacy_rather_than_passed() -> None:
    section = _section(_render(per_game=[_game(1, hard=None)]), HARD_CORRECTNESS_HEADING)

    assert "計測していません" in section
    assert "合格" not in section


def test_an_unfinished_game_is_called_out_in_the_hard_section() -> None:
    section = _section(
        _render(per_game=[_game(7, hard=_zero(), finished=False)]), HARD_CORRECTNESS_HEADING
    )

    assert "seed 7" in section and "止まるまで" in section


def test_the_existing_sections_are_kept_in_order() -> None:
    report = _render(per_game=[])
    headings = [line for line in report.splitlines() if line.startswith("## ")]

    assert [h.split(" ", 2)[2].split("(")[0] for h in headings] == [
        "動作指標",
        CALL_HEALTH_HEADING,
        HARD_CORRECTNESS_HEADING,
        "ルールベース検査",
        "発言の形式指標",
        "日本語の自然さ・人格の維持",
    ]


def test_no_calls_still_shows_the_event_counters() -> None:
    report = _render(metrics_summary={"total_calls": 0, "events": {"vote_code_resolved": 3}})

    section = _section(report, CALL_HEALTH_HEADING)
    assert "記録がありません" in section
    assert "(`vote_code_resolved`) | 3 |" in section


def test_format_duration() -> None:
    assert format_duration(59.6) == "01:00"
    assert format_duration(3725) == "1:02:05"


# -- ballots on a partner that a wolf-chat line was read as planning --


def _plan(basis: str, sentence: str, *, round_number: int = 1) -> Any:
    from app.eval.transcript import WolfAllyVotePlan

    return WolfAllyVotePlan(
        plan_id=f"g:3:{round_number}:p1:p2",
        voter_id="p1",
        target_id="p2",
        day=3,
        round_number=round_number,
        public_evidence_ids=(),
        story_reason=sentence,
        strategy_reason="",
        source_event_id="m9",
        basis=basis,
    )


def _played(*plans: Any) -> list[Any]:
    from app.eval.analyzers import AnalysisResult
    from app.eval.transcript import GameTranscript

    transcript = GameTranscript(seed=2, names={"p1": "ハルト", "p2": "ダイキ"})
    transcript.wolf_ally_vote_plans.extend(plans)
    return [(transcript, AnalysisResult())]


def test_a_ballot_planned_by_a_wolf_chat_line_is_listed_with_that_line() -> None:
    """The plan is a reading of wording, not a fact: a misread line would let a
    betrayal pass the gate as planned, so the report shows the line it rests on."""
    from app.eval.report import WOLF_CHAT_PLANS_HEADING

    report = _render(
        games=_played(
            _plan("wolf_chat", "明日はダイキ吊りで決まり|仕方ない"),
            _plan("wolf_chat", "ダイキに入れよう", round_number=2),
        ),
        per_game=[_game(2, hard=_zero())],
    )

    section = _section(report, HARD_CORRECTNESS_HEADING)
    assert WOLF_CHAT_PLANS_HEADING in section
    assert "| 2 | 3 | ハルト → ダイキ | 明日はダイキ吊りで決まり｜仕方ない |" in section
    assert "| 2 | 3(決選2回目) | ハルト → ダイキ | ダイキに入れよう |" in section


def test_a_ballot_planned_by_a_public_mark_is_not_listed() -> None:
    from app.eval.report import WOLF_CHAT_PLANS_HEADING

    report = _render(
        games=_played(_plan("public_mark", "公開の黒")),
        per_game=[_game(2, hard=_zero())],
    )

    assert WOLF_CHAT_PLANS_HEADING not in report
