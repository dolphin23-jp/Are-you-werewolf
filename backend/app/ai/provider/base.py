"""Provider-agnostic LLM interface. Two real implementations exist --
`anthropic_claude.py` (the Claude API, the one in use) and `luna_openai.py`
(OpenAI-compatible, kept dormant but selectable) -- plus the offline `mock`
and `scenario` doubles. Every AI-layer module talks to this interface, not to
a concrete SDK, which is what let the Anthropic provider arrive without
touching `coordinator.py` / `context.py` / `player_agent.py`.

`temperature` stays in the signature for the providers that sample with it;
a provider whose API has no sampling parameters (Anthropic) accepts and
ignores it rather than breaking the contract."""

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
