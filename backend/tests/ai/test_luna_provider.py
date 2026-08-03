"""LunaOpenAIProvider tests: verify the strict-schema -> json_object ->
permissive-parse fallback chain, without any real network calls (the
client's `chat.completions.create` is monkeypatched)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from app.ai.metrics import MetricsCollector
from app.ai.provider.base import Message
from app.ai.provider.luna_openai import LunaOpenAIProvider
from app.ai.schemas import VoteOutput


def _fake_response(
    content: str | None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
    reasoning_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
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


def _status_error(message: str, status_code: int = 400) -> APIStatusError:
    """An APIStatusError shaped like the one a real endpoint raises when it
    refuses the json_schema contract."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(message, response=response, body=None)


@pytest.mark.asyncio
async def test_strict_schema_rejection_is_remembered_and_not_retried():
    """The seed-11 live run spent 125 of 253 requests re-probing a contract the
    endpoint had already refused on the first call."""
    provider = _make_provider()
    formats: list[str] = []

    async def fake_create(**kwargs):
        formats.append(kwargs["response_format"]["type"])
        if kwargs["response_format"]["type"] == "json_schema":
            raise _status_error(
                "Invalid schema for response_format 'VoteOutput': "
                "'additionalProperties' is required to be supplied and to be false"
            )
        return _fake_response('{"vote_target":"p1","reason":"ok"}')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    for _ in range(3):
        assert await provider.generate_structured(
            system="sys", messages=[], response_schema=VoteOutput
        ) == VoteOutput(vote_target="p1", reason="ok")

    assert provider.strict_schema_supported is False
    # Probed once, then never again: three logical calls, four HTTP requests.
    assert formats == ["json_schema", "json_object", "json_object", "json_object"]


@pytest.mark.asyncio
async def test_malformed_strict_response_does_not_disable_strict_schema():
    """A model having a bad turn must not cost the process its strict path."""
    provider = _make_provider()
    formats: list[str] = []

    async def fake_create(**kwargs):
        formats.append(kwargs["response_format"]["type"])
        if kwargs["response_format"]["type"] == "json_schema":
            return _fake_response("not json")
        return _fake_response('{"vote_target":"p1","reason":"ok"}')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    for _ in range(2):
        await provider.generate_structured(system="sys", messages=[], response_schema=VoteOutput)

    assert provider.strict_schema_supported is None
    assert formats == ["json_schema", "json_object", "json_schema", "json_object"]


@pytest.mark.asyncio
async def test_unrelated_bad_request_does_not_disable_strict_schema():
    """A 400 that says nothing about the response format (an over-long prompt,
    say) breaks json_object too and is not evidence about strict support."""
    provider = _make_provider()

    async def fake_create(**kwargs):
        if kwargs["response_format"]["type"] == "json_schema":
            raise _status_error("This model's maximum context length is 8192 tokens")
        return _fake_response('{"vote_target":"p1","reason":"ok"}')

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    await provider.generate_structured(system="sys", messages=[], response_schema=VoteOutput)

    assert provider.strict_schema_supported is None


@pytest.mark.asyncio
async def test_truncated_response_is_reported_as_truncation_not_bad_json():
    """46% of the seed-11 calls failed as 'not parseable as JSON' with no way to
    tell an output-budget problem from a prompting one."""
    metrics = MetricsCollector()
    provider = LunaOpenAIProvider(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        model="gpt-5.6-luna",
        metrics=metrics,
    )

    async def fake_create(**kwargs):
        # Cut off mid-object, exactly as an exhausted completion budget leaves it.
        return _fake_response(
            '{"vote_target":"p1","rea',
            prompt_tokens=2700,
            completion_tokens=800,
            finish_reason="length",
            reasoning_tokens=760,
        )

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    assert (
        await provider.generate_structured(system="sys", messages=[], response_schema=VoteOutput)
        is None
    )

    summary = metrics.summary()
    assert summary["finish_reason_counts"] == {"length": 1}
    assert summary["truncated_calls"] == 1
    assert summary["truncated_failure_rate"] == 1.0
    assert summary["reasoning_tokens"] == 1520
    assert "cut off before the JSON closed" in summary["errors"][0]["error"]
