"""Which reasoning engine a game runs, resolved in exactly one place.

Five callers used to repeat `ReasoningRuntime(...) if engine == "v2" else None`.
Adding a third engine to each of them by hand is how one of them ends up
running the wrong one, so the mapping lives here and the callers ask.

    legacy  no runtime; the model decides and speaks with no fact ledger
    v2      runtime present; code decides, the model words the decision
    v3      runtime present; code keeps facts, logic, validation and the
            speaking order, the model decides and speaks
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.metrics import MetricsCollector
from app.ai.reasoning.runtime import ReasoningRuntime
from app.engine.state import GameState

ENGINES = ("legacy", "v2", "v3")


def uses_reasoning_runtime(engine: str) -> bool:
    return engine in ("v2", "v3")


def model_decides(engine: str) -> bool:
    """Whether the model, not the belief engine, picks targets and votes."""
    return engine in ("legacy", "v3")


def build_reasoning_runtime(
    engine: str,
    state: GameState,
    ai_player_ids: Sequence[str],
    *,
    seed: int | None = None,
    metrics: MetricsCollector | None = None,
) -> ReasoningRuntime | None:
    if engine not in ENGINES:
        raise ValueError(f"unknown reasoning engine {engine!r}; expected one of {ENGINES}")
    if not uses_reasoning_runtime(engine):
        return None
    return ReasoningRuntime(state, ai_player_ids, seed=seed, metrics=metrics)


__all__ = ["ENGINES", "build_reasoning_runtime", "model_decides", "uses_reasoning_runtime"]
