"""LunaOpenAIProvider tests: verify the strict-schema -> json_object ->
permissive-parse fallback chain, without any real network calls (the
client's `chat.completions.create` is monkeypatched)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.provider.base import Message
from app.ai.provider.luna_openai import LunaOpenAIProvider
from app.ai.schemas import VoteOutput


def _fake_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_provider() -> LunaOpenAIProvider:
    return LunaOpenAIProvider(
        api_key="sk-test", base_url="https://example.invalid/v1", model="gpt-5.6-luna"
    )


@pytest.mark.asyncio
async def test_strict_schema_success():
    provider = _make_provider()
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return _fake_response('{"vote_target": "p1", "reason": "ok"}')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result == VoteOutput(vote_target="p1", reason="ok")
    assert calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_falls_back_to_json_object_mode_when_strict_schema_call_raises():
    provider = _make_provider()
    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        if kwargs["response_format"]["type"] == "json_schema":
            raise RuntimeError("endpoint doesn't support json_schema")
        return _fake_response('{"vote_target": "p2", "reason": "fallback"}')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result == VoteOutput(vote_target="p2", reason="fallback")
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_permissive_parser_extracts_fenced_json_block():
    provider = _make_provider()

    async def fake_create(**kwargs):
        if kwargs["response_format"]["type"] == "json_schema":
            return _fake_response("not valid json at all")
        return _fake_response('Here you go:\n```json\n{"vote_target": "p3", "reason": "x"}\n```')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result == VoteOutput(vote_target="p3", reason="x")


@pytest.mark.asyncio
async def test_returns_none_on_total_failure():
    provider = _make_provider()

    async def fake_create(**kwargs):
        return _fake_response("completely unparseable garbage")

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result is None
