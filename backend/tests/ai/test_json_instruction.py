from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.ai.provider.base import Message, SchemaT
from app.ai.provider.json_instruction import JsonInstructionProvider


class _Response(BaseModel):
    value: str


class _RecordingProvider:
    def __init__(self) -> None:
        self.system: str | None = None
        self.messages: list[Message] | None = None

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        self.system = system
        self.messages = messages
        return response_schema.model_validate({"value": "ok"})


@pytest.mark.asyncio
async def test_adds_json_instruction_when_messages_do_not_contain_json() -> None:
    delegate = _RecordingProvider()
    provider = JsonInstructionProvider(delegate)

    result = await provider.generate_structured(
        system="Return only the requested structured object.",
        messages=[Message(role="user", content="Classify the candidates.")],
        response_schema=_Response,
    )

    assert result == _Response(value="ok")
    assert delegate.system is not None
    assert "json" in delegate.system.lower()


@pytest.mark.asyncio
async def test_does_not_duplicate_existing_json_instruction() -> None:
    delegate = _RecordingProvider()
    provider = JsonInstructionProvider(delegate)
    system = "Return one valid JSON object."

    await provider.generate_structured(
        system=system,
        messages=[Message(role="user", content="Classify the candidates.")],
        response_schema=_Response,
    )

    assert delegate.system == system
