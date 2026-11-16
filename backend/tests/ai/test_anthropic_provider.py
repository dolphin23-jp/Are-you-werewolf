"""AnthropicClaudeProvider tests, fully offline.

Every test drives the real `anthropic.AsyncAnthropic` client over an
`httpx2.MockTransport`, not a monkeypatched `create()`: request building,
header handling, response parsing and the SDK's own error classes are all
exercised for real, so a wrong parameter name or a status the SDK maps to an
unexpected exception fails here instead of on the first paid request.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic._constants import ANTHROPIC_API_VERSION
from pydantic import BaseModel

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider import anthropic_claude
from app.ai.provider.anthropic_claude import FALLBACK_BETA, AnthropicClaudeProvider
from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget
from app.ai.provider.pricing import cost_usd, price_for
from app.ai.schemas import (
    BriefDiscussionOutput,
    DiscussionOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)

API_KEY = "test-key"
BASE_URL = "https://example.invalid"
VOTE_JSON = '{"vote_target": "p1", "reason": "ok"}'
USER = [Message(role="user", content="誰に投票しますか")]

Handler = Callable[[httpx2.Request], Any]


class ChatDiscussionOutput(BaseModel):
    """Stand-in for the v3 chat schema: effort routing is by class name."""

    public_message: str


class _Server:
    """A scripted Messages API: each request is answered by the next item."""

    def __init__(self, *responses: httpx2.Response | Handler) -> None:
        self._responses = list(responses)
        self.requests: list[httpx2.Request] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("more requests were sent than the test scripted")
        item = self._responses.pop(0)
        if isinstance(item, httpx2.Response):
            return item
        result = item(request)
        if asyncio.iscoroutine(result):
            result = await result
        assert isinstance(result, httpx2.Response)
        return result

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def _message(
    text: str | None = VOTE_JSON,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
    request_id: str = "req_ok",
) -> httpx2.Response:
    # Opus 5.5 always thinks, and by default its thinking block comes first
    # with empty text -- the parser must skip it and take the text block.
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 100, "output_tokens": 50},
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    return httpx2.Response(200, headers={"request-id": request_id}, json=body)


def _error(
    status: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers={"request-id": f"req_{status}", **(headers or {})},
        json={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _client(server: _Server) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(server.handler)
        ),
    )


class _Sleeps(list[float]):
    async def __call__(self, seconds: float) -> None:
        self.append(seconds)


def _provider(
    server: _Server,
    *,
    metrics: MetricsCollector | None = None,
    sleeps: _Sleeps | None = None,
    **overrides: Any,
) -> AnthropicClaudeProvider:
    kwargs: dict[str, Any] = {
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "model": "claude-opus-5-5",
        "max_tokens_floor": 12000,
        "metrics": metrics,
        "client": _client(server),
        "sleep": sleeps if sleeps is not None else _Sleeps(),
    }
    kwargs.update(overrides)
    return AnthropicClaudeProvider(**kwargs)


async def _vote(provider: AnthropicClaudeProvider, **kwargs: Any) -> VoteOutput | None:
    return await provider.generate_structured(
        system="あなたは人狼ゲームの村人です。", messages=USER, response_schema=VoteOutput, **kwargs
    )


def _only_record(metrics: MetricsCollector) -> CallRecord:
    assert len(metrics.records) == 1
    return metrics.records[0]


# --- the request on the wire --------------------------------------------------


async def test_request_shape_matches_the_api_contract():
    server = _Server(_message())
    provider = _provider(server, fallbacks=False)

    result = await _vote(provider, max_tokens=800, temperature=0.3)

    assert result == VoteOutput(vote_target="p1", reason="ok")
    (body,) = server.bodies
    # No sampling parameters exist on this API (400 if sent) and thinking is
    # always on for Opus 5.5 -- none of them may appear.
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in body
    assert body["model"] == "claude-opus-5-5"
    # The caller's budget was sized for a non-thinking model; the floor wins.
    assert body["max_tokens"] == 12000
    # System prompt at top level, never smuggled in as a message.
    assert body["system"] == "あなたは人狼ゲームの村人です。"
    assert body["messages"] == [{"role": "user", "content": "誰に投票しますか"}]
    output_format = body["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False
    assert output_format["schema"]["required"] == ["vote_target"]
    assert body["output_config"]["effort"] == "medium"
    assert "fallbacks" not in body
    assert FALLBACK_BETA not in server.requests[0].headers.get("anthropic-beta", "")


async def test_a_larger_caller_budget_than_the_floor_is_kept():
    server = _Server(_message())
    provider = _provider(server, max_tokens_floor=12000)
    await _vote(provider, max_tokens=20000)
    assert server.bodies[0]["max_tokens"] == 20000


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        (ChatDiscussionOutput, "low"),
        (DiscussionOutput, "low"),
        (BriefDiscussionOutput, "low"),
        (VoteOutput, "high"),
        (NightActionOutput, "xhigh"),
        (WolfChatOutput, "max"),
        (SummaryOutput, "medium"),
    ],
)
async def test_effort_is_routed_by_call_kind(schema: type[BaseModel], expected: str):
    server = _Server(_message('{"x": 1}'))
    provider = _provider(
        server,
        effort_by_kind={
            "discussion": "low",
            "vote": "high",
            "night": "xhigh",
            "chat": "max",
            "default": "medium",
        },
    )
    await provider.generate_structured(system="s", messages=USER, response_schema=schema)
    assert server.bodies[0]["output_config"]["effort"] == expected
    assert provider.effort_for(schema) == expected


def test_an_invalid_effort_is_refused_at_construction():
    with pytest.raises(ValueError, match="effort"):
        _provider(_Server(), effort_by_kind={"vote": "extreme"})


async def test_fallbacks_send_the_beta_header_and_the_scalar_form_only_when_enabled():
    server = _Server(_message(), _message())
    await _vote(_provider(server, fallbacks=True))
    await _vote(_provider(server, fallbacks=False))

    enabled, disabled = server.requests
    assert FALLBACK_BETA in enabled.headers["anthropic-beta"]
    assert json.loads(enabled.content)["fallbacks"] == "default"
    assert FALLBACK_BETA not in disabled.headers.get("anthropic-beta", "")
    assert "fallbacks" not in json.loads(disabled.content)


async def test_a_rejected_fallback_beta_is_switched_off_instead_of_failing_every_call():
    server = _Server(
        _error(400, "fallbacks: 'default' is not available for this organization"),
        _message(),
        _message(),
    )
    provider = _provider(server, fallbacks=True)

    assert await _vote(provider) is not None
    assert await _vote(provider) is not None
    assert provider.fallbacks_enabled is False
    first, retried, later = server.bodies
    assert first["fallbacks"] == "default"
    assert "fallbacks" not in retried and "fallbacks" not in later


async def test_calls_in_flight_when_the_fallback_beta_is_rejected_are_all_resent():
    """A parallel batch (votes, night actions) is all built with `fallbacks`
    before any answer arrives. Whichever 400 comes back first switches the
    feature off; the others must still be re-sent without it rather than
    fail because the shared switch already reads off."""
    in_flight = 3
    arrived: list[httpx2.Request] = []
    all_arrived = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if "fallbacks" not in json.loads(request.content):
            return _message()
        arrived.append(request)
        if len(arrived) == in_flight:
            all_arrived.set()
        await asyncio.wait_for(all_arrived.wait(), timeout=5)
        return _error(400, "fallbacks: 'default' is not available for this organization")

    server = _Server(*(respond for _ in range(2 * in_flight)))
    metrics = MetricsCollector()
    provider = _provider(server, metrics=metrics, fallbacks=True)

    results = await asyncio.gather(*(_vote(provider) for _ in range(in_flight)))

    assert all(result is not None for result in results)
    assert provider.fallbacks_enabled is False
    assert len(server.requests) == 2 * in_flight
    assert all(record.path is ParsePath.STRICT_SCHEMA for record in metrics.records)
    # A rejection of the beta is not a transient-error retry.
    assert all(record.attempt == 0 for record in metrics.records)


async def test_a_request_with_fallbacks_reserves_a_second_attempt():
    """With fallbacks on, one call can bill the requested model's attempt
    (which may decline after its whole `max_tokens`) and then the fallback
    model's attempt on the same input. The reservation must cover both, or
    a cap can be crossed by a call that passed the check."""
    budget = EvaluationBudget(max_requests=10, max_estimated_cost=50.0)
    reserved_in_flight: list[float] = []

    def rejected(request: httpx2.Request) -> httpx2.Response:
        reserved_in_flight.append(budget.reserved_cost)
        return _error(400, "fallbacks: 'default' is not available for this organization")

    def served(request: httpx2.Request) -> httpx2.Response:
        reserved_in_flight.append(budget.reserved_cost)
        return _message()

    # The same call, same prompt and same prices, sent once with fallbacks
    # and then without them: only the number of attempts it can bill differs.
    server = _Server(rejected, served)
    provider = _provider(server, fallbacks=True)
    provider.set_request_budget(budget)

    assert await _vote(provider) is not None

    with_fallback, without = reserved_in_flight
    assert "fallbacks" in server.bodies[0] and "fallbacks" not in server.bodies[1]
    assert without > 0.0
    assert with_fallback == pytest.approx(2 * without)
    assert budget.reserved_cost == pytest.approx(0.0)


# --- credentials and endpoint are ours, never the environment's --------------


def _capture_built_client(monkeypatch: pytest.MonkeyPatch, server: _Server) -> dict[str, Any]:
    """Let the provider construct its own SDK client, but over the mock
    transport, and record what it passed."""
    captured: dict[str, Any] = {}
    real = anthropic.AsyncAnthropic

    def build(**kwargs: Any) -> anthropic.AsyncAnthropic:
        captured.update(kwargs)
        return real(
            **kwargs,
            http_client=anthropic.DefaultAsyncHttpxClient(
                transport=httpx2.MockTransport(server.handler)
            ),
        )

    monkeypatch.setattr(anthropic_claude, "AsyncAnthropic", build)
    return captured


async def test_key_and_base_url_are_explicit_even_when_ambient_variables_exist(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-auth-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS", "X-Ambient-Header: leaked\nAuthorization: Bearer ambient"
    )
    server = _Server(_message())
    captured = _capture_built_client(monkeypatch, server)

    provider = AnthropicClaudeProvider(
        api_key=API_KEY, base_url=BASE_URL, model="claude-opus-5-5", timeout_seconds=42.0
    )
    assert await _vote(provider) is not None

    assert captured["api_key"] == API_KEY
    assert captured["base_url"] == BASE_URL
    # Retries are ours alone, so every HTTP request is counted and budgeted.
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 42.0
    (request,) = server.requests
    assert request.url.host == "example.invalid"
    # SDK 2.x sends an API key as `Authorization: Bearer`; older ones used
    # `x-api-key`. Either way it must be our key and only our key.
    sent = request.headers.get("x-api-key") or request.headers["authorization"]
    assert sent in (API_KEY, f"Bearer {API_KEY}")
    assert "X-Ambient-Header" not in request.headers
    all_header_values = " ".join(request.headers.values())
    assert "ambient" not in all_header_values


async def test_ambient_headers_never_strip_the_sdks_own_required_headers(
    monkeypatch: pytest.MonkeyPatch,
):
    """Another tool's `ANTHROPIC_CUSTOM_HEADERS` may name a header the SDK
    sends itself. Cancelling that name must leave the SDK's own value in
    place: a request with no `anthropic-version` or with a non-JSON
    `Content-Type` is a 400 on every call of every game."""
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "anthropic-version: 2023-06-01\nContent-Type: text/plain\nACCEPT: text/html\n"
        "X-Ambient-Header: leaked",
    )
    server = _Server(_message())
    _capture_built_client(monkeypatch, server)

    provider = AnthropicClaudeProvider(api_key=API_KEY, base_url=BASE_URL, model="claude-opus-5-5")
    assert await _vote(provider) is not None

    (request,) = server.requests
    assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "application/json"
    # A name the SDK does not send itself is still cancelled outright.
    assert "X-Ambient-Header" not in request.headers


def test_an_empty_key_is_refused_rather_than_failing_every_call_with_401():
    with pytest.raises(ValueError, match="api_key"):
        AnthropicClaudeProvider(api_key="", base_url=BASE_URL, model="claude-opus-5-5")


# --- responses ----------------------------------------------------------------


async def test_thinking_then_text_content_is_parsed_and_usage_recorded():
    metrics = MetricsCollector()
    usage = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 300,
        "output_tokens": 50,
        "output_tokens_details": {"thinking_tokens": 30},
    }
    server = _Server(_message(usage=usage, request_id="req_usage"))
    result = await _vote(_provider(server, metrics=metrics))

    assert result == VoteOutput(vote_target="p1", reason="ok")
    record = _only_record(metrics)
    assert record.path is ParsePath.STRICT_SCHEMA
    # prompt_tokens is the total input; the cache fields say how it split.
    assert record.prompt_tokens == 420
    assert record.cache_creation_tokens == 20
    assert record.cache_read_tokens == 300
    assert record.completion_tokens == 50
    assert record.reasoning_tokens == 30
    assert record.stop_reason == "end_turn"
    assert record.finish_reason == "end_turn"
    assert record.served_model == "claude-opus-5-5"
    assert record.requested_model == "claude-opus-5-5"
    assert record.request_id == "req_usage"
    assert record.effort == "medium"
    assert record.http_requests == 1 and record.attempt == 0
    assert record.error is None
    price, _ = price_for("claude-opus-5-5")
    assert record.estimated_cost == pytest.approx(
        cost_usd(
            price,
            input_tokens=100,
            cache_creation_tokens=20,
            cache_read_tokens=300,
            output_tokens=50,
        )
    )


def _iteration(
    kind: str,
    *,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": kind,
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output_tokens,
    }
    if model is not None:
        entry["model"] = model
    return entry


async def test_a_fallback_served_turn_bills_every_attempt_at_its_own_model():
    """The shape the API returns when Opus 5.5 declines part-way and a
    fallback model serves the turn: top-level `usage` is only the serving
    attempt, `usage.iterations` lists both, and the declined attempt's 8000
    output tokens are billed too. Reading top-level usage alone priced this
    turn at about a fifth of its bill."""
    metrics = MetricsCollector()
    usage = {
        "input_tokens": 5000,
        "output_tokens": 600,
        "iterations": [
            # The requested model's declined attempt; the API may leave its
            # model unnamed, which means the requested one.
            _iteration("message", input_tokens=5000, output_tokens=8000),
            # Fallback attempts are repriced credit-style: cache reads count.
            _iteration(
                "fallback_message",
                model="claude-opus-5",
                input_tokens=1000,
                cache_read=4000,
                output_tokens=600,
            ),
        ],
    }
    server = _Server(_message(model="claude-opus-5", usage=usage))
    provider = _provider(server, metrics=metrics)
    budget = EvaluationBudget(max_requests=10, max_estimated_cost=5.0)
    provider.set_request_budget(budget)

    assert await _vote(provider) is not None

    declined = cost_usd(price_for("claude-opus-5-5")[0], input_tokens=5000, output_tokens=8000)
    served = cost_usd(
        price_for("claude-opus-5")[0], input_tokens=1000, cache_read_tokens=4000, output_tokens=600
    )
    record = _only_record(metrics)
    assert record.served_model == "claude-opus-5"
    assert record.estimated_cost == pytest.approx(declined + served)
    assert budget.spent_cost == pytest.approx(declined + served)
    # Tokens are counted the same way as the cost, attempt by attempt.
    assert record.prompt_tokens == 10000
    assert record.cache_read_tokens == 4000
    assert record.completion_tokens == 8600
    assert metrics.summary()["fallback_served_calls"] == 1


async def test_without_iterations_the_top_level_usage_is_priced_at_the_served_model():
    """A response with no per-attempt breakdown has only top-level usage to go
    on, and the model that served it is the one that billed it."""
    metrics = MetricsCollector()
    server = _Server(_message(model="claude-opus-5"))
    await _vote(_provider(server, metrics=metrics))

    record = _only_record(metrics)
    assert record.served_model == "claude-opus-5"
    assert record.estimated_cost == pytest.approx(
        cost_usd(price_for("claude-opus-5")[0], input_tokens=100, output_tokens=50)
    )
    assert record.prompt_tokens == 100 and record.completion_tokens == 50


async def test_a_single_attempt_iterations_list_matches_top_level_usage():
    """Iterations can also accompany an ordinary response; they must not be
    counted on top of the top-level figures they repeat."""
    metrics = MetricsCollector()
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "iterations": [_iteration("message", input_tokens=100, output_tokens=50)],
    }
    await _vote(_provider(_Server(_message(usage=usage)), metrics=metrics))

    record = _only_record(metrics)
    assert record.estimated_cost == pytest.approx(
        cost_usd(price_for("claude-opus-5-5")[0], input_tokens=100, output_tokens=50)
    )
    assert record.prompt_tokens == 100 and record.completion_tokens == 50


async def test_max_tokens_stop_is_a_truncation_and_is_not_retried():
    metrics = MetricsCollector()
    server = _Server(_message('{"vote_target": "p', stop_reason="max_tokens"))
    sleeps = _Sleeps()

    assert await _vote(_provider(server, metrics=metrics, sleeps=sleeps)) is None

    assert len(server.requests) == 1 and sleeps == []
    record = _only_record(metrics)
    assert record.finish_reason == "length"
    assert record.truncated is True
    assert record.stop_reason == "max_tokens"
    assert record.path is ParsePath.FAILED


async def test_refusal_is_recorded_with_its_category_and_sent_exactly_once(
    caplog: pytest.LogCaptureFixture,
):
    metrics = MetricsCollector()
    server = _Server(
        _message(
            None,
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "reasoning_extraction"},
            request_id="req_refused",
        )
    )
    with caplog.at_level("WARNING", logger="app.ai.provider.anthropic"):
        assert await _vote(_provider(server, metrics=metrics)) is None

    assert len(server.requests) == 1
    record = _only_record(metrics)
    assert record.finish_reason == "refusal"
    assert record.refusal_category == "reasoning_extraction"
    assert metrics.summary()["refusals"] == {"reasoning_extraction": 1}
    # The failure is logged with what an operator needs, and no prompt text.
    (line,) = [r.getMessage() for r in caplog.records]
    assert "VoteOutput" in line and "reasoning_extraction" in line and "req_refused" in line
    assert "誰に投票しますか" not in line and API_KEY not in line


async def test_an_end_turn_text_that_fails_validation_is_salvaged_permissively():
    metrics = MetricsCollector()
    server = _Server(_message(f"```json\n{VOTE_JSON}\n```"))
    result = await _vote(_provider(server, metrics=metrics))
    assert result == VoteOutput(vote_target="p1", reason="ok")
    assert _only_record(metrics).path is ParsePath.PERMISSIVE


async def test_an_unparseable_text_is_a_failure():
    metrics = MetricsCollector()
    server = _Server(_message("投票先は決められません"))
    assert await _vote(_provider(server, metrics=metrics)) is None
    record = _only_record(metrics)
    assert record.path is ParsePath.FAILED
    assert record.error == "strict schema response did not validate"


# --- retries --------------------------------------------------------------------


async def test_rate_limit_waits_for_retry_after_then_succeeds():
    metrics = MetricsCollector()
    sleeps = _Sleeps()
    server = _Server(
        _error(429, "rate limited", error_type="rate_limit_error", headers={"retry-after": "7"}),
        _message(),
    )
    result = await _vote(_provider(server, metrics=metrics, sleeps=sleeps))

    assert result is not None
    assert sleeps == [7.0]
    record = _only_record(metrics)
    assert record.http_requests == 2 and record.attempt == 1
    # A 429 is a rejection before inference: free. Only the response is priced.
    assert record.estimated_cost == pytest.approx(
        cost_usd(price_for("claude-opus-5-5")[0], input_tokens=100, output_tokens=50)
    )


async def test_retry_after_is_capped():
    sleeps = _Sleeps()
    server = _Server(
        _error(429, "slow down", error_type="rate_limit_error", headers={"retry-after": "3600"}),
        _message(),
    )
    await _vote(_provider(server, sleeps=sleeps))
    assert sleeps == [30.0]


async def test_overloaded_529_is_retried_with_backoff():
    sleeps = _Sleeps()
    server = _Server(_error(529, "Overloaded", error_type="overloaded_error"), _message())
    assert await _vote(_provider(server, sleeps=sleeps)) is not None
    assert sleeps == [1.0]
    assert len(server.requests) == 2


async def test_timeouts_and_connection_errors_are_retried():
    def timeout(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    def disconnect(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    sleeps = _Sleeps()
    server = _Server(timeout, disconnect, _message())
    assert await _vote(_provider(server, sleeps=sleeps, max_retries=2)) is not None
    assert sleeps == [1.0, 2.0]


async def test_retries_exhausted_returns_none_with_the_last_error():
    metrics = MetricsCollector()
    sleeps = _Sleeps()
    server = _Server(*(_error(500, "boom", error_type="api_error") for _ in range(3)))

    assert await _vote(_provider(server, metrics=metrics, sleeps=sleeps, max_retries=2)) is None

    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0]
    record = _only_record(metrics)
    assert record.http_requests == 3 and record.attempt == 2
    assert record.error is not None and record.error.startswith("InternalServerError: ")
    assert len(record.error) <= 200
    assert record.request_id == "req_500"
    # No response ever arrived: usage is unknown, not zero.
    assert record.prompt_tokens is None and record.completion_tokens is None


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_client_errors_are_never_retried(status: int):
    sleeps = _Sleeps()
    server = _Server(_error(status, "no"))
    assert await _vote(_provider(server, sleeps=sleeps)) is None
    assert len(server.requests) == 1 and sleeps == []


async def test_a_monthly_spend_limit_429_is_not_retried():
    sleeps = _Sleeps()
    response = httpx2.Response(
        429,
        headers={"retry-after": "1"},
        json={
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "spend limit reached",
                "details": {"error_code": "enforced_spend_limit_reached"},
            },
        },
    )
    server = _Server(response)
    assert await _vote(_provider(server, sleeps=sleeps)) is None
    assert len(server.requests) == 1 and sleeps == []


# --- grammar rejection memory -----------------------------------------------------


async def test_a_schema_the_grammar_rejects_is_remembered_for_that_schema_only():
    metrics = MetricsCollector()
    sleeps = _Sleeps()
    server = _Server(
        _error(400, "Schema is too complex for compilation"),
        _message(VOTE_JSON),  # degraded retry within the same call
        _message(f"投票します。{VOTE_JSON}"),  # a later VoteOutput call
        _message('{"target": "p3", "reason": "guard"}'),  # another schema
    )
    provider = _provider(server, metrics=metrics, sleeps=sleeps)

    assert await _vote(provider) == VoteOutput(vote_target="p1", reason="ok")
    assert provider.grammar_rejected_schemas == frozenset({"VoteOutput"})
    assert await _vote(provider) == VoteOutput(vote_target="p1", reason="ok")
    night = await provider.generate_structured(
        system="s", messages=USER, response_schema=NightActionOutput
    )
    assert night == NightActionOutput(target="p3", reason="guard")

    rejected, degraded, later, other = server.bodies
    assert "format" in rejected["output_config"]
    for body in (degraded, later):
        # No grammar, effort kept, and the schema reaches the model as text.
        assert "format" not in body["output_config"]
        assert body["output_config"]["effort"] == "medium"
        assert "BEGIN_RESPONSE_JSON_SCHEMA" in body["system"]
        assert '"vote_target"' in body["system"]
    assert other["output_config"]["format"]["type"] == "json_schema"
    assert "BEGIN_RESPONSE_JSON_SCHEMA" not in other["system"]
    # The switch to the degraded path is not a transient-error retry.
    assert sleeps == []

    first, second, third = metrics.records
    assert first.path is ParsePath.JSON_OBJECT and first.http_requests == 2
    assert second.path is ParsePath.PERMISSIVE and second.http_requests == 1
    assert third.path is ParsePath.STRICT_SCHEMA


async def test_an_unrelated_bad_request_is_not_remembered():
    server = _Server(_error(400, "max_tokens: 999999 > 128000, the maximum allowed"), _message())
    provider = _provider(server)

    assert await _vote(provider) is None
    assert provider.grammar_rejected_schemas == frozenset()
    assert await _vote(provider) is not None
    assert "format" in server.bodies[1]["output_config"]


async def test_a_system_prompt_already_carrying_the_schema_is_not_given_it_twice():
    server = _Server(_error(400, "Schema is too complex for compilation"), _message())
    provider = _provider(server)
    system = "s\nBEGIN_RESPONSE_JSON_SCHEMA\n{}\nEND_RESPONSE_JSON_SCHEMA"
    await provider.generate_structured(system=system, messages=USER, response_schema=VoteOutput)
    assert server.bodies[1]["system"] == system


# --- budget -----------------------------------------------------------------------


async def test_budget_exceeded_propagates_and_nothing_is_sent():
    server = _Server()
    provider = _provider(server, max_concurrency=1)
    provider.set_request_budget(EvaluationBudget(max_requests=0))

    with pytest.raises(BudgetExceeded):
        await _vote(provider)
    assert server.requests == []

    # The limiter slot was released: with max_concurrency=1 a leaked slot
    # would hang the next call instead of letting it through.
    ok_server = _Server(_message())
    other = _provider(ok_server, max_concurrency=1)
    assert await asyncio.wait_for(_vote(other), timeout=5) is not None


async def test_the_cost_cap_refuses_a_request_whose_worst_case_does_not_fit():
    server = _Server()
    provider = _provider(server)
    budget = EvaluationBudget(max_requests=10, max_estimated_cost=0.01)
    provider.set_request_budget(budget)

    # 12000 output tokens at $20/MTok alone is $0.24 -- over the cap.
    with pytest.raises(BudgetExceeded, match="estimated cost"):
        await _vote(provider)
    assert server.requests == []
    assert budget.used_requests == 0 and budget.reserved_cost == 0.0


async def test_budget_exceeded_on_a_retry_records_what_was_already_spent():
    metrics = MetricsCollector()
    server = _Server(_error(529, "Overloaded", error_type="overloaded_error"))
    provider = _provider(server, metrics=metrics)
    provider.set_request_budget(EvaluationBudget(max_requests=1))

    with pytest.raises(BudgetExceeded):
        await _vote(provider)

    record = _only_record(metrics)
    assert record.http_requests == 1
    assert record.error is not None and record.error.startswith("BudgetExceeded")


async def test_budget_and_metrics_agree_on_the_cost_across_attempts():
    metrics = MetricsCollector()
    server = _Server(_error(529, "Overloaded", error_type="overloaded_error"), _message())
    provider = _provider(server, metrics=metrics)
    budget = EvaluationBudget(max_requests=10, max_estimated_cost=5.0)
    provider.set_request_budget(budget)

    assert await _vote(provider) is not None

    record = _only_record(metrics)
    response_cost = cost_usd(price_for("claude-opus-5-5")[0], input_tokens=100, output_tokens=50)
    # The 529 attempt may have been partly processed, so it is charged its
    # estimated input on top of the response's real cost.
    assert record.estimated_cost is not None and record.estimated_cost > response_cost
    assert budget.spent_cost == pytest.approx(record.estimated_cost)
    assert budget.reserved_cost == pytest.approx(0.0)
    assert budget.used_requests == 2
    assert budget.estimated_cost == pytest.approx(record.estimated_cost)


async def test_a_cancelled_call_releases_its_reservation():
    started = asyncio.Event()

    async def hang(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    server = _Server(hang)
    provider = _provider(server)
    budget = EvaluationBudget(max_requests=10, max_estimated_cost=5.0)
    provider.set_request_budget(budget)

    task = asyncio.create_task(_vote(provider))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert budget.reserved_cost == pytest.approx(0.0)
    # Abandoned mid-flight: charged its estimated input, not nothing.
    assert budget.spent_cost > 0.0


# --- the process-wide limiter -----------------------------------------------------


async def test_the_concurrency_limit_is_shared_by_every_instance_on_a_loop():
    in_flight = 0
    peak = 0

    async def slow(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _message()

    server = _Server(*(slow for _ in range(4)))
    first = _provider(server, max_concurrency=1)
    second = _provider(server, max_concurrency=1)

    results = await asyncio.gather(_vote(first), _vote(second), _vote(first), _vote(second))

    assert all(result is not None for result in results)
    assert peak == 1


def test_one_provider_works_across_event_loops():
    """A single module-level semaphore would raise 'bound to a different
    event loop' on the second asyncio.run; per-loop limiters do not."""
    server = _Server(_message(), _message())
    provider = _provider(server)
    assert asyncio.run(_vote(provider)) is not None
    assert asyncio.run(_vote(provider)) is not None
