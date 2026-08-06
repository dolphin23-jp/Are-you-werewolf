from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.ai.provider.base import Message
from app.ai.provider.json_instruction import JsonInstructionProvider


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: str


class _RecordingProvider:
    def __init__(self) -> None:
        self.system: str | None = None
        self.messages: list[Message] | None = None

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[_Response],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> _Response | None:
        self.system = system
        self.messages = messages
        return response_schema(value="ok", confidence="high")


@pytest.mark.asyncio
async def test_embeds_exact_json_schema_in_system_prompt() -> None:
    delegate = _RecordingProvider()
    provider = JsonInstructionProvider(delegate)

    result = await provider.generate_structured(
        system="Return only the requested structured object.",
        messages=[Message(role="user", content="Classify the candidates.")],
        response_schema=_Response,
    )

    assert result == _Response(value="ok", confidence="high")
    assert delegate.system is not None
    assert "BEGIN_RESPONSE_JSON_SCHEMA" in delegate.system
    assert "exactly one valid JSON object" in delegate.system

    schema_text = delegate.system.split("BEGIN_RESPONSE_JSON_SCHEMA\n", 1)[1].split(
        "\nEND_RESPONSE_JSON_SCHEMA", 1
    )[0]
    embedded: dict[str, Any] = json.loads(schema_text)
    assert embedded == _Response.model_json_schema()
    assert embedded["additionalProperties"] is False
    assert set(embedded["required"]) == {"value", "confidence"}


@pytest.mark.asyncio
async def test_schema_instruction_is_not_duplicated() -> None:
    delegate = _RecordingProvider()
    provider = JsonInstructionProvider(delegate)
    system = (
        "Return one valid JSON object.\n"
        "BEGIN_RESPONSE_JSON_SCHEMA\n{}\nEND_RESPONSE_JSON_SCHEMA"
    )

    await provider.generate_structured(
        system=system,
        messages=[Message(role="user", content="Classify the candidates.")],
        response_schema=_Response,
    )

    assert delegate.system == system
    assert delegate.system.count("BEGIN_RESPONSE_JSON_SCHEMA") == 1
