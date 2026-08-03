"""OpenAI-compatible client for gpt-5.6-luna.

"OpenAI-compatible" is a weaker promise than it sounds: the first real
call to gpt-5.6-luna was rejected outright for sending `max_tokens`
(it requires `max_completion_tokens`). So this client negotiates rather
than assumes -- see `app/ai/dialect.py` -- and is defensive throughout:

  - Optional request parameters are learned from the endpoint's own
    rejections on the first call and reused thereafter, so neither the
    operator nor this file has to hardcode a model generation's quirks.
  - Uses `openai.AsyncOpenAI` exclusively (never the sync client), so a
    real API call can never block the event loop -- this was a real bug in
    the prior Claude-based implementation.
  - Tries strict JSON-schema structured output first
    (`response_format={"type": "json_schema", ...}`); if the endpoint
    rejects or ignores that, falls back to `{"type": "json_object"}` mode
    with the same permissive 3-stage JSON extraction the prior
    implementation used (raw parse -> fenced ```json block -> first-`{`-
    to-last-`}` slice), so an unexpected response shape degrades instead
    of throwing.
  - A semaphore bounds concurrent in-flight requests
    (`LUNA_MAX_CONCURRENCY`), since the coordinator fires up to 16
    concurrent calls during voting/night phases.
  - `generate_structured` returns None (never raises) on total failure;
    `player_agent.py` is responsible for the retry/fallback-line/
    invalid-target safety nets built on top of this.
  - A reasoning model bills its private thinking against the same
    completion budget as the visible answer, so a budget sized without
    that in mind runs out mid-object instead of producing a short answer.
    The largest `reasoning_tokens` seen on any call is remembered and
    added to every later request's budget (`_padded_max_tokens`); a call
    that still truncates is retried once at double the budget before it
    is counted as failed.

When a `MetricsCollector` is attached, every logical call records which
parse path actually worked, its real HTTP request count, the wall-clock
latency, and token usage summed across all responses. See `app/ai/metrics.py`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import ValidationError

from app.ai.dialect import EndpointDialect
from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.budget import EvaluationBudget

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# Headroom reserved for the visible JSON on top of the learned reasoning
# overhead below -- the overhead alone covers only what the thinking used
# last time, not the structural cost of the answer that has to follow it.
_REASONING_BUDGET_MARGIN = 200


@dataclass
class _Attempt:
    result: Any = None
    path: ParsePath = ParsePath.FAILED
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    http_requests: int = 0
    finish_reason: str | None = None
    reasoning_tokens: int = 0
    # The endpoint refused the json_schema contract itself, as opposed to
    # answering it badly. Permanent for this endpoint, so it is worth caching.
    strict_unsupported: bool = False


# This is the single retry boundary. One logical generation uses strict schema
# and, only if that fails, json_object. Therefore its absolute HTTP ceiling is
# 2 * (max_retries + 1). Agent code never retries the same contract.
DEFAULT_MAX_HTTP_RETRIES = 2


class LunaOpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_concurrency: int = 6,
        timeout_seconds: float = 30.0,
        max_retries: int = DEFAULT_MAX_HTTP_RETRIES,
        metrics: MetricsCollector | None = None,
    ) -> None:
        # SDK retries are disabled: doing them here makes every actual request
        # countable and prevents a hidden second retry layer.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0
        )
        self._model = model
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._metrics = metrics
        # Learned from the endpoint's own rejections on the first call, then
        # reused for the rest of the process.
        self._dialect = EndpointDialect()
        self._request_budget: EvaluationBudget | None = None
        # None until the endpoint has told us. False is sticky and only set on
        # an outright rejection of the json_schema contract, never on a call
        # that merely came back malformed -- a model having a bad turn must not
        # cost the whole process its strict-schema path.
        self._strict_schema_supported: bool | None = None
        # The largest `reasoning_tokens` this endpoint has spent on any call so
        # far. A reasoning model bills its private thinking against the same
        # completion budget as the visible answer, so a budget sized without
        # this ran out mid-object on the seed-11 live run in 46% of calls, all
        # of them finish_reason=length. Grows monotonically and is applied to
        # every subsequent request before it is sent, so later turns do not
        # have to fail first to find out how much room they need.
        self._reasoning_overhead: int = 0

    @property
    def strict_schema_supported(self) -> bool | None:
        return self._strict_schema_supported

    @property
    def reasoning_overhead(self) -> int:
        return self._reasoning_overhead

    def _padded_max_tokens(self, max_tokens: int) -> int:
        if self._reasoning_overhead == 0:
            return max_tokens
        return max_tokens + self._reasoning_overhead + _REASONING_BUDGET_MARGIN

    def _learn_reasoning_overhead(self, reasoning_tokens: int) -> None:
        if reasoning_tokens > self._reasoning_overhead:
            self._reasoning_overhead = reasoning_tokens

    def set_request_budget(self, budget: EvaluationBudget) -> None:
        """Attach the manual evaluation budget at the actual HTTP boundary."""
        self._request_budget = budget

    @property
    def dialect(self) -> EndpointDialect:
        return self._dialect

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        openai_messages = [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

        prompt_tokens = 0
        completion_tokens = 0
        last_error: str | None = None
        http_requests = 0
        finish_reason: str | None = None
        reasoning_tokens = 0
        started = asyncio.get_running_loop().time()
        # One request per logical call is spent probing strict schema, so an
        # endpoint that has already rejected it outright must not be asked
        # again -- on the seed-11 live run that was 125 certain-fail 400s out
        # of 253 requests, doubling both the request count and the latency.
        modes = 1 if self._strict_schema_supported is False else 2
        padded_max_tokens = self._padded_max_tokens(max_tokens)

        async with self._semaphore:
            if self._strict_schema_supported is not False:
                strict = await self._try_strict_schema(
                    openai_messages, response_schema, padded_max_tokens, temperature
                )
                prompt_tokens += strict.prompt_tokens
                completion_tokens += strict.completion_tokens
                http_requests += strict.http_requests
                last_error = strict.error
                finish_reason = strict.finish_reason or finish_reason
                reasoning_tokens += strict.reasoning_tokens
                self._learn_reasoning_overhead(strict.reasoning_tokens)
                if strict.strict_unsupported:
                    self._strict_schema_supported = False
                if strict.result is not None:
                    self._strict_schema_supported = True
                    self._record(
                        response_schema,
                        strict.path,
                        started,
                        max(0, http_requests - 1),
                        http_requests,
                        prompt_tokens,
                        completion_tokens,
                        None,
                        strict.finish_reason,
                        strict.reasoning_tokens,
                    )
                    return strict.result  # type: ignore[no-any-return]

            fallback = await self._try_json_object_mode(
                openai_messages, response_schema, padded_max_tokens, temperature
            )
            prompt_tokens += fallback.prompt_tokens
            completion_tokens += fallback.completion_tokens
            http_requests += fallback.http_requests
            last_error = fallback.error or last_error
            finish_reason = fallback.finish_reason or finish_reason
            reasoning_tokens += fallback.reasoning_tokens
            if fallback.result is not None:
                self._record(
                    response_schema,
                    fallback.path,
                    started,
                    max(0, http_requests - modes),
                    http_requests,
                    prompt_tokens,
                    completion_tokens,
                    None,
                    fallback.finish_reason,
                    fallback.reasoning_tokens,
                )
                return fallback.result  # type: ignore[no-any-return]

        self._record(
            response_schema,
            ParsePath.FAILED,
            started,
            max(0, http_requests - modes),
            http_requests,
            prompt_tokens,
            completion_tokens,
            last_error,
            finish_reason,
            reasoning_tokens,
        )
        return None

    def _record(
        self,
        response_schema: type[SchemaT],
        path: ParsePath,
        started: float,
        attempt: int,
        http_requests: int,
        prompt_tokens: int,
        completion_tokens: int,
        error: str | None,
        finish_reason: str | None = None,
        reasoning_tokens: int = 0,
    ) -> None:
        if self._metrics is None:
            return
        elapsed = asyncio.get_running_loop().time() - started
        self._metrics.record(
            CallRecord(
                schema=response_schema.__name__,
                path=path,
                latency_seconds=elapsed,
                attempt=attempt,
                http_requests=http_requests,
                # None (not 0) when the endpoint reported no usage, so the
                # summary can say "unknown" instead of implying zero spend.
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
                error=error,
                finish_reason=finish_reason,
                reasoning_tokens=reasoning_tokens or None,
            )
        )

    async def _create(
        self,
        openai_messages: list[dict[str, str]],
        response_format: dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Run one response mode with the sole HTTP/API retry budget.

        Dialect negotiation consumes the same bounded budget rather than
        creating another retry layer.
        """
        requests = 0
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": openai_messages,
                "response_format": response_format,
            }
            self._dialect.apply(kwargs, max_tokens=max_tokens, temperature=temperature)
            try:
                if self._request_budget is not None:
                    self._request_budget.claim_request()
                requests += 1
                return await self._client.chat.completions.create(**kwargs), requests
            except Exception as exc:
                last_error = exc
                adapted = self._dialect.adapt(exc)
                if not adapted and not _is_retryable(exc):
                    raise _RequestFailure(exc, requests) from exc
        assert last_error is not None
        raise _RequestFailure(last_error, requests)

    async def _try_strict_schema(
        self,
        openai_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens: int,
        temperature: float,
    ) -> _Attempt:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "schema": response_schema.model_json_schema(),
                "strict": True,
            },
        }
        try:
            response, requests = await self._create(
                openai_messages, response_format, max_tokens, temperature
            )
        except Exception as exc:
            root = _root_error(exc)
            return _Attempt(
                error=_describe(root),
                http_requests=_request_count(exc),
                strict_unsupported=_rejects_strict_schema(root),
            )

        prompt_tokens, completion_tokens = _usage(response)
        if self._request_budget is not None:
            self._request_budget.record_usage(prompt_tokens, completion_tokens)
        content = response.choices[0].message.content if response.choices else None
        parsed = _parse_strict(content, response_schema)
        return _Attempt(
            result=parsed,
            path=ParsePath.STRICT_SCHEMA if parsed is not None else ParsePath.FAILED,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=None if parsed is not None else "strict schema response did not validate",
            http_requests=requests,
            finish_reason=_finish_reason(response),
            reasoning_tokens=_reasoning_tokens(response),
        )

    async def _try_json_object_mode(
        self,
        openai_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens: int,
        temperature: float,
    ) -> _Attempt:
        attempt = await self._json_object_attempt(
            openai_messages, response_schema, max_tokens, temperature
        )
        self._learn_reasoning_overhead(attempt.reasoning_tokens)
        if attempt.result is not None or attempt.finish_reason != "length":
            return attempt
        # The budget just learned from was exhausted by reasoning before the
        # answer got a chance to run -- padding did not carry over from an
        # earlier call because there was nothing to learn from yet, or this
        # turn's own thinking simply ran longer than any turn before it. One
        # retry at double the budget recovers the turn instead of leaving it
        # as a silent skip; the budget is not raised again beyond that, so a
        # model that never finishes is reported as failed rather than chased.
        retry = await self._json_object_attempt(
            openai_messages, response_schema, max_tokens * 2, temperature
        )
        self._learn_reasoning_overhead(retry.reasoning_tokens)
        return _Attempt(
            result=retry.result,
            path=retry.path,
            prompt_tokens=attempt.prompt_tokens + retry.prompt_tokens,
            completion_tokens=attempt.completion_tokens + retry.completion_tokens,
            error=retry.error,
            http_requests=attempt.http_requests + retry.http_requests,
            finish_reason=retry.finish_reason,
            reasoning_tokens=attempt.reasoning_tokens + retry.reasoning_tokens,
        )

    async def _json_object_attempt(
        self,
        openai_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens: int,
        temperature: float,
    ) -> _Attempt:
        try:
            response, requests = await self._create(
                openai_messages, {"type": "json_object"}, max_tokens, temperature
            )
        except Exception as exc:
            return _Attempt(error=_describe(_root_error(exc)), http_requests=_request_count(exc))

        prompt_tokens, completion_tokens = _usage(response)
        if self._request_budget is not None:
            self._request_budget.record_usage(prompt_tokens, completion_tokens)
        content = response.choices[0].message.content if response.choices else None

        # Distinguish "the body was already clean JSON" from "we had to dig
        # it out of prose", because only the latter signals a model that
        # ignores the response_format contract.
        finish_reason = _finish_reason(response)
        reasoning_tokens = _reasoning_tokens(response)

        direct = _parse_strict(content, response_schema)
        if direct is not None:
            return _Attempt(
                direct,
                ParsePath.JSON_OBJECT,
                prompt_tokens,
                completion_tokens,
                http_requests=requests,
                finish_reason=finish_reason,
                reasoning_tokens=reasoning_tokens,
            )

        salvaged = _parse_permissive(content, response_schema)
        # Name the truncation case rather than filing it under bad JSON: it is
        # fixed by raising max_tokens, not by changing the prompt.
        failure = (
            "response was cut off before the JSON closed (finish_reason=length)"
            if finish_reason == "length"
            else "response was not parseable as JSON"
        )
        return _Attempt(
            result=salvaged,
            path=ParsePath.PERMISSIVE if salvaged is not None else ParsePath.FAILED,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=None if salvaged is not None else failure,
            http_requests=requests,
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
        )


class _RequestFailure(Exception):
    def __init__(self, cause: Exception, requests: int) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.requests = requests


def _request_count(exc: Exception) -> int:
    return exc.requests if isinstance(exc, _RequestFailure) else 0


def _root_error(exc: Exception) -> Exception:
    return exc.cause if isinstance(exc, _RequestFailure) else exc


def _rejects_strict_schema(exc: Exception) -> bool:
    """Did the endpoint refuse the json_schema contract as such?

    Deliberately narrow. A 4xx can equally mean the prompt was too long, which
    would break json_object mode too, so the status alone is not enough -- the
    message has to name the response format. OpenAI-compatible endpoints fail
    this either by not implementing `json_schema` at all or, as here, by
    demanding `additionalProperties: false` on every object in the schema.
    """
    if not isinstance(exc, APIStatusError) or exc.status_code not in (400, 404, 422):
        return False
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message or "schema" in message


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _reasoning_tokens(response: Any) -> int:
    """Reasoning models bill thinking against the completion budget, so this is
    the part of `max_tokens` the visible answer never got to use."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0


def _finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return str(reason) if reason else None


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _parse_strict(content: str | None, response_schema: type[SchemaT]) -> SchemaT | None:
    if not content:
        return None
    try:
        return response_schema.model_validate_json(content)
    except (ValidationError, ValueError):
        return None


def _parse_permissive(content: str | None, response_schema: type[SchemaT]) -> SchemaT | None:
    if not content:
        return None

    direct = _parse_strict(content, response_schema)
    if direct is not None:
        return direct

    fenced = _FENCED_JSON_RE.search(content)
    if fenced is not None:
        candidate = _parse_strict(fenced.group(1), response_schema)
        if candidate is not None:
            return candidate

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return _parse_strict(content[start : end + 1], response_schema)

    return None
