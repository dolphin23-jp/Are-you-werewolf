"""LunaOpenAIProvider tests: verify the strict-schema -> json_object ->
permissive-parse fallback chain, without any real network calls (the
client's `chat.completions.create` is monkeypatched)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.metrics import MetricsCollector
from app.ai.provider.base import Message
from app.ai.provider.luna_openai import LunaOpenAIProvider
from app.ai.schemas import VoteOutput


def _fake_response(
    content: str | None, prompt_tokens: int = 0, completion_tokens: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


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
    assert len(calls) == 1
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
    provider = LunaOpenAIProvider(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        model="gpt-5.6-luna",
        max_retries=1,
    )
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return _fake_response("completely unparseable garbage")

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result is None
    # Parse failures are not HTTP failures and must not trigger API retries.
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_retry_ceiling_is_shared_by_both_response_modes(monkeypatch):
    provider = LunaOpenAIProvider(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        model="gpt-5.6-luna",
        max_retries=1,
    )
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("temporary")

    monkeypatch.setattr("app.ai.provider.luna_openai._is_retryable", lambda exc: True)
    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[], response_schema=VoteOutput
    )

    assert result is None
    assert len(calls) == 2 * (provider._max_retries + 1)


@pytest.mark.asyncio
async def test_metrics_count_requests_and_sum_usage_across_fallback():
    metrics = MetricsCollector()
    provider = LunaOpenAIProvider(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        model="gpt-5.6-luna",
        metrics=metrics,
    )
    calls: list[dict] = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema":
            return _fake_response("invalid", 10, 2)
        return _fake_response('{"vote_target":"p1","reason":"ok"}', 7, 3)

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    result = await provider.generate_structured(
        system="sys", messages=[], response_schema=VoteOutput
    )

    assert result is not None
    summary = metrics.summary()
    assert summary["total_calls"] == 1
    assert summary["http_requests"] == len(calls) == 2
    assert summary["strict_schema_failures"] == 1
    assert summary["json_object_successes"] == 1
    assert summary["tokens"] == {"prompt": 17, "completion": 5, "total": 22}
