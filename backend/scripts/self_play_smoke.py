"""Run strategy-free self-play episodes as a Phase-0 environment smoke test.

Usage from backend/:
    python scripts/self_play_smoke.py --games 100 --seed 1
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from app.engine.game import PlayerSpec
from app.training.runner import RandomEpisodeRunner


def _player_specs() -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--discussion-ticks", type=int, default=8)
    args = parser.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive")

    runner = RandomEpisodeRunner(
        _player_specs(),
        max_discussion_ticks=args.discussion_ticks,
    )
    outcomes: Counter[str] = Counter()
    total_days = 0
    total_events = 0
    started = time.perf_counter()

    for offset in range(args.games):
        result = runner.run(args.seed + offset)
        outcome = "draw" if result.is_draw else result.winner.value
        outcomes[outcome] += 1
        total_days += result.days
        total_events += result.semantic_event_count

    elapsed = time.perf_counter() - started
    games_per_second = args.games / elapsed if elapsed > 0 else float("inf")
    print(f"games={args.games} elapsed={elapsed:.3f}s games_per_second={games_per_second:.2f}")
    print(f"mean_days={total_days / args.games:.2f}")
    print(f"mean_semantic_events={total_events / args.games:.2f}")
    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}={count} ({count / args.games:.1%})")


if __name__ == "__main__":
    main()
