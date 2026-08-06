"""Ensure structured-generation prompts satisfy JSON-object endpoint requirements.

Some OpenAI-compatible endpoints reject ``response_format={"type": "json_object"}``
unless the message text explicitly contains the word ``json``. The structured
provider uses JSON-object mode as its compatibility fallback, so callers that
only say "structured object" can otherwise fail before the model is invoked.
"""

from __future__ import annotations

from app.ai.provider.base import LLMProvider, Message, SchemaT

_JSON_INSTRUCTION = (
    "Return the final answer as one valid JSON object matching the requested schema."
)


class JsonInstructionProvider:
    """Decorator that adds an explicit JSON instruction when one is absent."""

    def __init__(self, delegate: LLMProvider) -> None:
        self._delegate = delegate

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        all_text = "\n".join([system, *(message.content for message in messages)]).lower()
        if "json" not in all_text:
            system = f"{system.rstrip()}\n\n{_JSON_INSTRUCTION}"
        return await self._delegate.generate_structured(
            system=system,
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
