#!/usr/bin/env python3
"""Budgeted, resumable paired real-provider qualification (never run by CI)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.eval.reasoning_analyzer import (  # noqa: E402
    ReasoningQualityReport,
    ReasoningTranscriptAnalyzer,
)
from app.eval.release_gate import ReleaseGate  # noqa: E402
from app.eval.transcript import GameTranscript  # noqa: E402
from scripts.live_reasoning_check import run  # noqa: E402


def _cost(report: dict[str, Any], input_price: float, output_price: float) -> float:
    tokens = report.get("tokens") or {}
    return (
        tokens.get("prompt", 0) * input_price + tokens.get("completion", 0) * output_price
    ) / 1_000_000


async def execute(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    spent = 0.0
    requests = 0
    for seed in args.seeds[: args.max_games]:
        for engine in args.engines:
            path = args.output_dir / f"{seed}-{engine}.json"
            transcript = args.output_dir / f"{seed}-{engine}-transcript.json"
            if args.resume and path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
            else:
                report = await run(seed, transcript, engine)
                restored = GameTranscript.from_dict(
                    json.loads(transcript.read_text(encoding="utf-8"))
                )
                report["reasoning_quality"] = (
                    ReasoningTranscriptAnalyzer().analyze(restored).to_dict()
                )
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(report)
            requests += int(report.get("http_requests", report.get("llm_requests", 0)))
            spent += _cost(report, args.input_price_per_million, args.output_price_per_million)
            _save_aggregate(args.output_dir, rows, requests, spent)
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
    directory: Path, rows: list[dict[str, Any]], requests: int, cost: float
) -> None:
    legacy_seeds = {row["seed"] for row in rows if row.get("engine") == "legacy"}
    v2_seeds = {row["seed"] for row in rows if row.get("engine") == "v2"}
    live_pairs = len(legacy_seeds & v2_seeds)
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
    gate = ReleaseGate.from_toml(
        Path(__file__).resolve().parent.parent / "config" / "reasoning_release_gate.toml"
    ).evaluate(combined, {}, live_pairs=live_pairs)
    payload = {
        "games": rows,
        "http_requests": requests,
        "estimated_cost": cost,
        "release_decision": gate.decision.value,
        "release_reasons": gate.reasons,
    }
    (directory / "aggregate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "report.md").write_text(
        f"# Live A/B qualification\n\nGames: {len(rows)}\n\nHTTP requests: {requests}\n\n"
        f"Estimated cost: {cost:.4f}\n\nReleaseDecision: {gate.decision.value.upper()}\n"
        f"\nReasons: {', '.join(gate.reasons)}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--engines", nargs="+", choices=("legacy", "v2"), default=["legacy", "v2"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-http-requests", type=int, default=4000)
    parser.add_argument("--max-estimated-cost", type=float, default=20.0)
    parser.add_argument("--max-games", type=int, default=3)
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        default=float(os.getenv("LLM_INPUT_PRICE_PER_MILLION", "0")),
    )
    parser.add_argument(
        "--output-price-per-million",
        type=float,
        default=float(os.getenv("LLM_OUTPUT_PRICE_PER_MILLION", "0")),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
