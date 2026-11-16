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

The schedule is built from `--engines`, so `--engines v3` (or any single
engine) plays that engine's games; the gate measures every requested engine
except the legacy comparison arm. The provider comes from
`WEREWOLF_LLM_PROVIDER` as in `scripts/live_reasoning_check.py`; with
`anthropic` the provider prices its own calls and `--max-estimated-cost` caps
them before they are sent, and a resumed run carries the dollars already
spent forward:

    WEREWOLF_LLM_PROVIDER=anthropic WEREWOLF_ANTHROPIC_API_KEY=... \\
        python scripts/live_ab_reasoning_check.py --engines v3 --seeds 11 12 \\
            --max-estimated-cost 40 --output-dir live-ab-out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget  # noqa: E402
from app.eval.reasoning_analyzer import (  # noqa: E402
    ReasoningQualityReport,
    ReasoningTranscriptAnalyzer,
)
from app.eval.release_gate import HARD_FAILURE_FIELDS, ReleaseGate  # noqa: E402
from app.eval.release_report import HumanTranscriptReview  # noqa: E402
from app.eval.transcript import GameTranscript  # noqa: E402
from scripts.live_reasoning_check import run  # noqa: E402

COMPARISON_ENGINE = "legacy"


def candidate_engines(engines: Sequence[str]) -> list[str]:
    """The engines being qualified: every requested one except the legacy
    comparison arm, in the order given, without repeats."""
    return [e for e in dict.fromkeys(engines) if e != COMPARISON_ENGINE]


def build_schedule(
    seeds: Sequence[int], engines: Sequence[str], max_games: int
) -> list[tuple[int, str]]:
    """`(seed, engine)` pairs in qualification order, for exactly the requested engines.

    The order is the formal one the paired run always used -- on the first
    seed the candidate's smoke game comes before its legacy pair, so a broken
    candidate stops the run before the comparison arm is paid for; on later
    seeds legacy comes first -- but it is now built from `engines` rather than
    filtered from a hardcoded v2/legacy list, which silently scheduled nothing
    for `--engines v3`.
    """
    candidates = candidate_engines(engines)
    with_legacy = COMPARISON_ENGINE in engines
    schedule: list[tuple[int, str]] = []
    for index, seed in enumerate(seeds[: max(1, max_games)]):
        if index == 0:
            order = [*candidates, *([COMPARISON_ENGINE] if with_legacy else [])]
        else:
            order = [*([COMPARISON_ENGINE] if with_legacy else []), *candidates]
        schedule.extend((seed, engine) for engine in order)
    return schedule


def restore_budget(args: argparse.Namespace, budget_state: dict[str, Any]) -> EvaluationBudget:
    """The run's budget, carrying over what a resumed run already spent.

    Both accounting styles are restored: luna's token counts, and the settled
    dollars a self-pricing provider (anthropic) reports. Without the latter a
    resumed anthropic run would start its cap from zero again. `settled_attempts`
    is not in the snapshot; one is enough to make `estimated_cost` read the
    restored dollars rather than recompute from token prices it never had.
    """
    spent_cost = float(budget_state.get("spent_cost") or 0.0)
    return EvaluationBudget(
        max_requests=args.max_http_requests,
        used_requests=int(budget_state.get("used_requests", 0)),
        max_estimated_cost=args.max_estimated_cost,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        prompt_tokens=int(budget_state.get("prompt_tokens", 0)),
        completion_tokens=int(budget_state.get("completion_tokens", 0)),
        spent_cost=spent_cost,
        settled_attempts=1 if spent_cost > 0 else 0,
    )


async def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    existing: dict[str, Any] = {}
    aggregate_path = args.output_dir / "aggregate.json"
    if args.resume and aggregate_path.exists():
        existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
    budget = restore_budget(args, existing.get("budget", {}))
    requests = budget.used_requests
    spent = budget.estimated_cost or 0.0
    candidates = candidate_engines(args.engines)
    for seed, engine in build_schedule(args.seeds, args.engines, args.max_games):
        path = args.output_dir / f"{seed}-{engine}.json"
        transcript = args.output_dir / f"{seed}-{engine}-transcript.json"
        stopped_by_budget = False
        if args.resume and path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                report = await run(seed, transcript, engine, budget)
            except BudgetExceeded as exc:
                stopped_by_budget = True
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
        _save_aggregate(args.output_dir, rows, budget, args.review_dir, candidates=candidates)
        hard = report.get("reasoning_quality", {})
        # The release gate's own list: any of these fails it, so another paid
        # game cannot change the verdict.
        if engine in candidates and any(hard.get(field, 0) for field in HARD_FAILURE_FIELDS):
            return rows
        # A self-pricing provider refuses the request that would cross the
        # cap, so `spent` stays just under it and the check below never
        # fires: stop on the refusal itself, or every later game would start
        # only to be cut off. (A row reloaded on resume does not stop the run,
        # as before.)
        if stopped_by_budget:
            return rows
        if requests >= args.max_http_requests or spent >= args.max_estimated_cost:
            return rows
    return rows


def _save_aggregate(
    directory: Path,
    rows: list[dict[str, Any]],
    budget: EvaluationBudget,
    review_dir: Path | None = None,
    *,
    candidates: Sequence[str] = ("v2",),
) -> None:
    # Seeds, not rows: a resumed run that replays the same seed has not added
    # evidence, and the gate's bar is about coverage.
    candidate_rows = [row for row in rows if row.get("engine") in candidates]
    candidate_seeds = {
        row["seed"]
        for row in candidate_rows
        if row.get("status") not in ("budget_exhausted", "failed")
    }
    live_games = len(candidate_seeds)
    reports = [row.get("reasoning_quality", {}) for row in candidate_rows]
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
    candidate_game_ids = {str(row["game_id"]) for row in candidate_rows if row.get("game_id")}
    reviews: list[HumanTranscriptReview] = []
    if review_dir is not None:
        for game_id in candidate_game_ids:
            path = review_dir / f"{game_id}.json"
            if path.exists():
                reviews.append(HumanTranscriptReview.from_json(path))
    review_complete = (
        bool(candidate_game_ids)
        and {r.game_id for r in reviews if r.complete} == candidate_game_ids
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
        "engines": list(candidates),
        "models": sorted({str(row["model"]) for row in rows if row.get("model")}),
        "games": rows,
        "stages": _stage_status(rows, candidates),
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
        f"Qualified engines: {', '.join(candidates)}\n\n"
        f"Models: {', '.join(payload['models']) or 'unknown'}\n\n"
        f"HTTP requests: {budget.used_requests}\n\n"
        f"Estimated cost: {budget.estimated_cost}\n\n"
        f"ReleaseDecision: {gate.decision.value.upper()}\n"
        f"\nReasons: {', '.join(gate.reasons)}\n",
        encoding="utf-8",
    )


def _stage_status(
    rows: list[dict[str, Any]], candidates: Sequence[str] = ("v2",)
) -> dict[str, dict[str, Any]]:
    # The key names are kept from the v2-only days so existing readers of
    # aggregate.json still find them; stage A is the candidate engines' smoke.
    first_seed = rows[0].get("seed") if rows else None
    stage_a = [r for r in rows if r.get("seed") == first_seed and r.get("engine") in candidates]
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    # v2 only by default. The legacy arm was the A side of a comparison that has
    # served its purpose, and every legacy game is a full game's spend and about
    # 25 minutes of wall time to re-measure an engine nobody is shipping. The
    # path is kept, not deleted: pass `--engines legacy v2` for a paired run.
    parser.add_argument("--engines", nargs="+", choices=("legacy", "v2", "v3"), default=["v2"])
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
    args = parser.parse_args(argv)
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
