#!/usr/bin/env python3
"""Run reproducible, network-free seeded legacy/v2 campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compare_engines import play  # noqa: E402


def wilson_interval(wins: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    p = wins / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def parse_seeds(value: str) -> list[int]:
    if ":" not in value:
        return [int(item) for item in value.split(",")]
    start, end = (int(part) for part in value.split(":", 1))
    return list(range(start, end + 1))


async def campaign(seeds: list[int], engines: list[str]) -> dict[str, Any]:
    rows = [(await play(engine, seed)).as_dict() for seed in seeds for engine in engines]
    aggregate: dict[str, Any] = {"games": rows, "game_count": len(rows), "engines": {}}
    for engine in engines:
        subset = [row for row in rows if row["engine"] == engine]
        winner_counts: dict[str, int] = {}
        for row in subset:
            winner = row.get("winner") or "none"
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        aggregate["engines"][engine] = {
            "games": len(subset),
            "completed_games": sum(row.get("winner") is not None for row in subset),
            "aborted_games": sum(row.get("winner") is None for row in subset),
            "logical_calls": sum(row["llm_requests"] for row in subset),
            "http_requests": sum(row["http_requests"] for row in subset),
            "public_utterances": sum(row.get("public_utterances", 0) for row in subset),
            "executions": [item for row in subset for item in row.get("executions", [])],
            "reasoning_quality": {
                key: sum(row.get("reasoning_quality", {}).get(key, 0) for row in subset)
                for key in (subset[0].get("reasoning_quality", {}) if subset else {})
            },
            "sample_size_warning": len(subset) < 100,
            "role_survival": _role_survival(subset),
            "wins_by_team": {
                team: {
                    "wins": wins,
                    "trials": len(subset),
                    "ratio": wins / len(subset),
                    "wilson_95": list(wilson_interval(wins, len(subset))),
                }
                for team, wins in winner_counts.items()
            },
        }
    legacy = aggregate["engines"].get("legacy")
    v2 = aggregate["engines"].get("v2")
    if legacy and v2:
        aggregate["comparison"] = {
            "logical_call_reduction": _reduction(
                legacy["logical_calls"], v2["logical_calls"]
            ),
            "http_request_reduction": _reduction(
                legacy["http_requests"], v2["http_requests"]
            ),
        }
    return aggregate


def _reduction(before: int, after: int) -> float | None:
    return (before - after) / before if before else None


def _role_survival(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_role: dict[str, list[int]] = {}
    for row in rows:
        if row.get("winner") is None:
            continue
        for role, days in row.get("role_survival_days", {}).items():
            by_role.setdefault(role, []).extend(days)
    return {
        role: {
            "observations": len(days),
            "mean": statistics.fmean(days),
            "median": statistics.median(days),
            "min": min(days),
            "max": max(days),
        }
        for role, days in by_role.items()
        if days
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock",), default="mock")
    parser.add_argument("--seeds", default="1:10")
    parser.add_argument("--engines", nargs="+", choices=("legacy", "v2"), default=["legacy", "v2"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(campaign(parse_seeds(args.seeds), args.engines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
