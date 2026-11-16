"""Record the exact prompts a game sends, without sending them anywhere.

Measurement that replays prompts against a real model (the latency sweep,
`scripts/latency_sweep.py`) is only meaningful if the prompts are the ones a
game really produces: the same system prompt, the same board memo, the same
schema and token budget, on the same kind of turn. Hand-written stand-ins
would measure a different workload. `CapturingProvider` therefore sits in
front of any `LLMProvider`, notes each request as it passes, and delegates
unchanged; `capture_offline_game` drives one complete all-AI game through it
with the offline `ScenarioProvider`, so the capture costs nothing and touches
no network.

The scenario double answers with structurally populated output (replies,
directed questions, claims, results), so the game's later prompts carry the
pending-question layer, the "since you last spoke" digest and the claim
ledger a live game builds up. The *text* of those answers is canned, which is
fine for timing: what the sweep measures is how long a model takes to answer
a prompt of this shape and size, not whether the canned history is good play.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from app.ai.coordinator import AICoordinator
from app.ai.engine_mode import build_reasoning_runtime, model_decides
from app.ai.metrics import MetricsCollector
from app.ai.provider.base import LLMProvider, Message, SchemaT
from app.ai.provider.scenario import ScenarioProvider
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase

# What a capture records about where the game was when the call was made.
# Either part may be unknown (None) when no context function was given.
GameContext = tuple[int | None, str | None]


@dataclass(frozen=True)
class CapturedPrompt:
    """One request exactly as the AI layer handed it to the provider.

    `response_schema` is the Pydantic class itself, so a replay can build the
    same structured-output format and validate the answer the same way the
    provider does; `schema` is its name, which is what survives
    serialization (and what the provider's effort routing keys on).
    """

    index: int
    response_schema: type[BaseModel]
    system: str
    messages: tuple[Message, ...]
    max_tokens: int
    day: int | None = None
    phase: str | None = None

    @property
    def schema(self) -> str:
        return self.response_schema.__name__

    @property
    def input_chars(self) -> int:
        """Characters of prompt text, for sizing and rough cost estimates."""
        return len(self.system) + sum(len(m.content) for m in self.messages)

    def wire_messages(self) -> list[dict[str, str]]:
        """The `messages` list in the shape the Messages API takes."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "schema": self.schema,
            "system": self.system,
            "messages": self.wire_messages(),
            "max_tokens": self.max_tokens,
            "day": self.day,
            "phase": self.phase,
        }


class CapturingProvider:
    """An `LLMProvider` that records every request, then delegates it.

    `context`, when given, is asked for `(day, phase)` at the moment of each
    call; a provider has no view of the game, so whoever owns the game state
    supplies it. The inner provider's `_metrics` and `model` are forwarded
    for the same reason `BudgetedProvider` forwards them: the coordinator
    finds the metrics collector by duck typing on the provider it is given,
    and a wrapper that hid it would silently drop every recorded event.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        context: Callable[[], GameContext] | None = None,
    ) -> None:
        self.inner = inner
        self._context = context
        self.captured: list[CapturedPrompt] = []

    @property
    def _metrics(self) -> object | None:
        return getattr(self.inner, "_metrics", None)

    @property
    def model(self) -> str | None:
        return getattr(self.inner, "model", None)

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        day, phase = self._context() if self._context is not None else (None, None)
        self.captured.append(
            CapturedPrompt(
                index=len(self.captured),
                response_schema=response_schema,
                system=system,
                # Copied: the caller owns the list and may keep appending to it.
                messages=tuple(Message(role=m.role, content=m.content) for m in messages),
                max_tokens=max_tokens,
                day=day,
                phase=phase,
            )
        )
        return await self.inner.generate_structured(
            system=system,
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )


# The same seventeen names `scripts/evaluate.py` seats, so a captured prompt
# reads exactly like one from an evaluation game of the same seed.
CAPTURE_PLAYER_NAMES: tuple[str, ...] = (
    "マコト",
    "アカリ",
    "ハルト",
    "ユイ",
    "ソウタ",
    "ミオ",
    "レン",
    "サクラ",
    "カイト",
    "ノゾミ",
    "リク",
    "ツムギ",
    "ダイキ",
    "ホノカ",
    "シオン",
    "アオイ",
    "ケント",
)

# A game that has not ended after this many phase steps is stuck, not long.
_MAX_PHASE_STEPS = 200


@dataclass
class CaptureResult:
    """What one offline capture game produced."""

    seed: int
    engine: str
    prompts: list[CapturedPrompt] = field(default_factory=list)
    winner: str | None = None
    days: int = 0
    finished: bool = False

    def schema_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for prompt in self.prompts:
            counts[prompt.schema] = counts.get(prompt.schema, 0) + 1
        return dict(sorted(counts.items()))


async def capture_offline_game(
    seed: int,
    *,
    engine: str = "v3",
    inner: LLMProvider | None = None,
    metrics: MetricsCollector | None = None,
) -> CaptureResult:
    """Play one all-AI game offline and return every prompt it sent.

    The phase loop is the one `scripts/evaluate.py` runs (copied rather than
    imported: that script is a CLI, and its loop also carries live-log and
    partial-transcript plumbing a capture has no use for). The reasoning
    runtime and the model-decides flag come from `app.ai.engine_mode`, the
    same place the API and the evaluator resolve them, so a v3 capture sends
    v3 prompts -- the chat wire schema included.

    `inner` defaults to `ScenarioProvider(seed=seed)`; it must be offline,
    because every call the game makes goes to it.
    """
    metrics = metrics if metrics is not None else MetricsCollector()
    specs = [
        PlayerSpec(player_id=f"p{i}", name=name, is_human=False)
        for i, name in enumerate(CAPTURE_PLAYER_NAMES)
    ]
    controller = GameController(session_id=f"capture-{seed}", player_specs=specs, seed=seed)
    ai_ids = [spec.player_id for spec in specs]

    def context() -> GameContext:
        state = controller.state
        return state.day, str(state.phase)

    provider = CapturingProvider(
        inner if inner is not None else ScenarioProvider(seed=seed, metrics=metrics),
        context=context,
    )
    coordinator = AICoordinator(
        controller.state,
        ai_ids,
        provider,
        seed=seed,
        reasoning=build_reasoning_runtime(engine, controller.state, ai_ids, seed=seed),
        model_decides=model_decides(engine),
        pacing_scale=0.0,
    )
    session = SimpleNamespace(
        controller=controller,
        # No seat is human; the coordinator still reads the attribute.
        human_id="p0",
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )

    controller.start_game()
    for _ in range(_MAX_PHASE_STEPS):
        phase = controller.state.phase
        if phase == Phase.GAME_OVER:
            break
        if phase == Phase.NIGHT:
            await coordinator.run_night_phase(session)
        elif phase == Phase.DAWN:
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            await coordinator.generate_all_votes(session)
        elif phase == Phase.VOTE_RESULT:
            controller.start_night()
        else:
            raise RuntimeError(f"unexpected phase {phase}")

    state = controller.state
    winner = getattr(state, "winner", None)
    return CaptureResult(
        seed=seed,
        engine=engine,
        prompts=list(provider.captured),
        winner=str(winner) if winner is not None else None,
        days=state.day,
        finished=state.phase == Phase.GAME_OVER,
    )


__all__ = [
    "CAPTURE_PLAYER_NAMES",
    "CaptureResult",
    "CapturedPrompt",
    "CapturingProvider",
    "GameContext",
    "capture_offline_game",
]
