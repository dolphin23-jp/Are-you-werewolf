#!/usr/bin/env python3
"""Run reproducible, network-free seeded legacy/v2 campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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
            "logical_calls": sum(row["llm_requests"] for row in subset),
            "http_requests": sum(row["http_requests"] for row in subset),
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
    return aggregate


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
