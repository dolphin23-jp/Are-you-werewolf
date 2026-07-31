"""Provider-agnostic LLM interface. Only one real implementation exists
today (`luna_openai.py`, OpenAI-compatible), but every AI-layer module talks
to this interface, not to a concrete SDK -- a second provider is addable
without touching `coordinator.py` / `context.py` / `player_agent.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


class LLMProvider(Protocol):
    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        """Return a validated instance of `response_schema`, or None if the
        provider could not produce a valid response after its own internal
        retries. Callers (player_agent.py) must handle None gracefully."""
        ...
