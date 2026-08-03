#!/usr/bin/env python3
"""Budgeted, resumable real-provider qualification (never run by CI).

Written as a paired legacy-vs-v2 harness. It no longer runs the pair by
default: `--engines` is v2 only, because the legacy arm exists to answer a
comparison question that has been answered, and re-running it costs a full
game's spend and roughly 25 minutes of wall time per seed for an engine that
is not being shipped. The legacy path is kept, not deleted -- pass
`--engines legacy v2` to get the old behaviour back.

The release gate moved with it. It used to require `minimum_live_pairs`, the
number of seeds that ran both arms, which a v2-only run can never satisfy; an
unsatisfiable criterion is not a strict gate but a silent one. It now requires
`minimum_live_games` distinct v2 seeds instead. The bar itself is unchanged at
two, and nothing else was loosened -- in particular PASS still needs a complete
human transcript review (`--review-dir`), which no amount of machine evidence
substitutes for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget  # noqa: E402
from app.eval.reasoning_analyzer import (  # noqa: E402
    ReasoningQualityReport,
    ReasoningTranscriptAnalyzer,
)
from app.eval.release_gate import ReleaseGate  # noqa: E402
from app.eval.release_report import HumanTranscriptReview  # noqa: E402
from app.eval.transcript import GameTranscript  # noqa: E402
from scripts.live_reasoning_check import run  # noqa: E402


async def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    existing: dict[str, Any] = {}
    aggregate_path = args.output_dir / "aggregate.json"
    if args.resume and aggregate_path.exists():
        existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
    budget_state = existing.get("budget", {})
    requests = int(budget_state.get("used_requests", 0))
    budget = EvaluationBudget(
        max_requests=args.max_http_requests,
        used_requests=requests,
        max_estimated_cost=args.max_estimated_cost,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        prompt_tokens=int(budget_state.get("prompt_tokens", 0)),
        completion_tokens=int(budget_state.get("completion_tokens", 0)),
    )
    spent = budget.estimated_cost or 0.0
    # Formal qualification order: v2 smoke, its legacy pair, then remaining pairs.
    schedule = [(args.seeds[0], "v2"), (args.seeds[0], "legacy")]
    schedule.extend(
        (seed, engine) for seed in args.seeds[1 : args.max_games] for engine in ("legacy", "v2")
    )
    for seed, engine in schedule:
        if engine not in args.engines:
            continue
        path = args.output_dir / f"{seed}-{engine}.json"
        transcript = args.output_dir / f"{seed}-{engine}-transcript.json"
        if args.resume and path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                report = await run(seed, transcript, engine, budget)
            except BudgetExceeded as exc:
                report = {
                    "seed": seed,
                    "engine": engine,
                    "status": "budget_exhausted",
                    "error": str(exc),
                }
            if transcript.exists():
                restored = GameTranscript.from_dict(
                    json.loads(transcript.read_text(encoding="utf-8"))
                )
                report["reasoning_quality"] = (
                    ReasoningTranscriptAnalyzer().analyze(restored).to_dict()
                )
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(report)
        requests = budget.used_requests
        spent = budget.estimated_cost or 0.0
        _save_aggregate(args.output_dir, rows, budget, args.review_dir)
        hard = report.get("reasoning_quality", {})
        if engine == "v2" and any(
            hard.get(field, 0)
            for field in (
                "private_evidence_exposed_count",
                "team_private_evidence_exposed_count",
                "dead_target_selection_count",
                "missing_required_result_count",
                "displayed_target_mismatch_count",
                "stale_evidence_publicly_emitted_count",
                "unexplained_vote_change_count",
                "unplanned_wolf_ally_vote_count",
                "duplicate_night_action_count",
            )
        ):
            return rows
        if requests >= args.max_http_requests or spent >= args.max_estimated_cost:
            return rows
    return rows


def _save_aggregate(
    directory: Path,
    rows: list[dict[str, Any]],
    budget: EvaluationBudget,
    review_dir: Path | None = None,
) -> None:
    # Seeds, not rows: a resumed run that replays the same seed has not added
    # evidence, and the gate's bar is about coverage.
    v2_seeds = {
        row["seed"]
        for row in rows
        if row.get("engine") == "v2" and row.get("status") not in ("budget_exhausted", "failed")
    }
    live_games = len(v2_seeds)
    reports = [row.get("reasoning_quality", {}) for row in rows if row.get("engine") == "v2"]
    combined = (
        ReasoningQualityReport(
            **{
                field: sum(report.get(field, 0) for report in reports)
                for field in ReasoningQualityReport.__dataclass_fields__
            }
        )
        if reports
        else ReasoningQualityReport()
    )
    v2_game_ids = {
        str(row["game_id"]) for row in rows if row.get("engine") == "v2" and row.get("game_id")
    }
    reviews: list[HumanTranscriptReview] = []
    if review_dir is not None:
        for game_id in v2_game_ids:
            path = review_dir / f"{game_id}.json"
            if path.exists():
                reviews.append(HumanTranscriptReview.from_json(path))
    review_complete = (
        bool(v2_game_ids) and {r.game_id for r in reviews if r.complete} == v2_game_ids
    )
    operational_complete = (
        budget.pricing_supplied and bool(rows) and all("operational_metrics" in row for row in rows)
    )
    gate = ReleaseGate.from_toml(
        Path(__file__).resolve().parent.parent / "config" / "reasoning_release_gate.toml"
    ).evaluate(
        combined,
        _combined_operational(rows),
        live_games=live_games,
        human_review_complete=review_complete,
        operational_complete=operational_complete,
    )
    payload = {
        "games": rows,
        "stages": _stage_status(rows),
        "http_requests": budget.used_requests,
        "estimated_cost": budget.estimated_cost,
        "budget": budget.snapshot(),
        "release_decision": gate.decision.value,
        "release_reasons": gate.reasons,
    }
    (directory / "aggregate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "report.md").write_text(
        f"# Live A/B qualification\n\nGames: {len(rows)}\n\n"
        f"HTTP requests: {budget.used_requests}\n\n"
        f"Estimated cost: {budget.estimated_cost}\n\n"
        f"ReleaseDecision: {gate.decision.value.upper()}\n"
        f"\nReasons: {', '.join(gate.reasons)}\n",
        encoding="utf-8",
    )


def _stage_status(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    first_seed = rows[0].get("seed") if rows else None
    stage_a = [r for r in rows if r.get("seed") == first_seed and r.get("engine") == "v2"]
    stage_b = [r for r in rows if r.get("seed") == first_seed]
    stage_c = [r for r in rows if r.get("seed") != first_seed]

    def status(items: list[dict[str, Any]], expected: int) -> dict[str, Any]:
        failures = [r for r in items if r.get("status") in {"budget_exhausted", "failed"}]
        return {
            "status": "failed" if failures else ("passed" if len(items) >= expected else "pending"),
            "game_ids": [r.get("game_id") for r in items if r.get("game_id")],
            "stop_reason": failures[0].get("error", "") if failures else "",
            "updated_at": datetime.now(UTC).isoformat(),
        }

    return {
        "stage_a_v2_smoke": status(stage_a, 1),
        "stage_b_paired_smoke": status(stage_b, 2),
        "stage_c_small_evaluation": status(stage_c, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    # v2 only by default. The legacy arm was the A side of a comparison that has
    # served its purpose, and every legacy game is a full game's spend and about
    # 25 minutes of wall time to re-measure an engine nobody is shipping. The
    # path is kept, not deleted: pass `--engines legacy v2` for a paired run.
    parser.add_argument("--engines", nargs="+", choices=("legacy", "v2"), default=["v2"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-http-requests", type=int, default=4000)
    parser.add_argument("--max-estimated-cost", type=float, default=20.0)
    parser.add_argument("--max-games", type=int, default=3)
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=(float(value) if (value := os.getenv("LLM_INPUT_PRICE_PER_MILLION")) else None),
    )
    parser.add_argument(
        "--output-price-per-million",
        type=float,
        default=(float(value) if (value := os.getenv("LLM_OUTPUT_PRICE_PER_MILLION")) else None),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-dir", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(execute(args))


def _combined_operational(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [row.get("operational_metrics", {}) for row in rows]
    return {
        "complete_failure_rate": max(
            (
                float(item.get("complete_failures", 0))
                / max(float(item.get("total_calls", 0)), 1.0)
                for item in metrics
            ),
            default=0.0,
        ),
        "discussion_skip_rate": max(
            (float(item.get("discussion_skip_rate", 0.0)) for item in metrics), default=0.0
        ),
    }


if __name__ == "__main__":
    main()
