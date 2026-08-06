"""Make JSON-object fallback prompts self-describing.

Some OpenAI-compatible endpoints reject ``response_format={"type": "json_object"}``
unless the message text explicitly contains the word ``json``. More importantly,
JSON-object mode does not transmit the Pydantic response schema to the model.
Without an explicit contract in the prompt, the endpoint can return syntactically
valid JSON that cannot be validated as the requested response type.
"""

from __future__ import annotations

import json

from app.ai.provider.base import LLMProvider, Message, SchemaT

_SCHEMA_MARKER = "BEGIN_RESPONSE_JSON_SCHEMA"


class JsonInstructionProvider:
    """Decorator that embeds the exact response schema in the system prompt."""

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
        if _SCHEMA_MARKER not in system:
            schema_json = json.dumps(
                response_schema.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            instruction = (
                "Return exactly one valid JSON object and no surrounding prose or "
                "Markdown. The object must validate against the JSON Schema below. "
                "Include every required field, use the exact field names and enum "
                "values, and do not add undeclared fields.\n"
                f"{_SCHEMA_MARKER}\n{schema_json}\nEND_RESPONSE_JSON_SCHEMA"
            )
            system = f"{system.rstrip()}\n\n{instruction}"

        return await self._delegate.generate_structured(
            system=system,
            messages=messages,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
