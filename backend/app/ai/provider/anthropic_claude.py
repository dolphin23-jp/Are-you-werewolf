"""Anthropic Claude provider -- the real LLM behind the game (default: see `Settings`).

Each design point below is either a fact of this API or a lesson the luna
provider learned the expensive way:

  - Structured output is a grammar, not a hope. Every request carries
    `output_config.format` with the response schema (run through the SDK's
    `transform_schema`, which forces `additionalProperties: false` and moves
    keywords the grammar cannot express, such as `default`, into
    descriptions), and the reply is still validated with Pydantic, because
    a constraint moved into a description is no longer enforced.
  - A schema the grammar compiler refuses ("Schema is too complex for
    compilation") is remembered *for that schema class only*. That schema is
    then asked for with its JSON Schema embedded in the system prompt and
    parsed permissively; every other schema keeps the strict path. luna's
    process-wide "strict unsupported" switch would be wrong here: the limit
    is per schema, not per endpoint.
  - Thinking is always on for Opus 5.5 and billed against `max_tokens`, so
    every request gets at least `max_tokens_floor` (only tokens actually
    produced are billed, so the ceiling costs nothing by itself). Depth is
    steered per call kind with `output_config.effort`, never with a thinking
    budget, which this model rejects.
  - This API has no sampling parameters. `temperature` is accepted because
    the provider Protocol has it, and dropped.
  - Retries happen here and only here (the SDK's own are switched off), so
    every HTTP request is counted, priced and budgeted. Transient failures
    (connection, timeout, 408/409/429/5xx/529) are retried with backoff that
    honours `retry-after`; a refusal, a truncation or any other 4xx never is
    -- repeating them only repeats the bill.
  - One concurrency limiter per event loop is shared by every instance in the
    process. The rate limit belongs to the API key, and every game builds
    its own provider, so per-instance limiters would multiply the real
    concurrency by the number of games.
  - The key and the base URL are always passed to the SDK explicitly, and the
    SDK's other ambient inputs (`ANTHROPIC_CUSTOM_HEADERS`, OpenTelemetry
    content capture) are neutralised. `ANTHROPIC_*` variables on this machine
    usually belong to some other tool; inheriting them would bill the wrong
    account or ship hidden-role prompts somewhere unintended.
  - A call is priced attempt by attempt. With refusal fallbacks on, one call
    can bill the requested model's attempt (which may decline after writing
    its whole `max_tokens`) and then the fallback model's, so the budget
    reserves both before sending and settles on `usage.iterations`, the
    per-attempt bill; top-level `usage` covers only the attempt that served.
  - `generate_structured` returns None on every failure except one:
    `BudgetExceeded` propagates, because a spending cap that turns into a
    fallback line is not a cap.

When a `MetricsCollector` is attached, every logical call records one
`CallRecord` with the parse path, HTTP request count, retries, usage including
cache and thinking tokens, the served model, the request id and an estimated
cost from `app/ai/provider/pricing.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    BadRequestError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)

# Not re-exported at the package top level. Imported by name rather than
# hard-coded so the pin below always matches what this SDK sends, and so an
# SDK upgrade that moves it fails at import instead of in a paid request.
from anthropic._constants import ANTHROPIC_API_VERSION
from anthropic.types.beta import BetaMessage, BetaTextBlock
from pydantic import BaseModel, ValidationError

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.budget import BudgetExceeded, EvaluationBudget

# The same marker `JsonInstructionProvider` uses, so a system prompt that
# wrapper already extended is not given the schema a second time.
from app.ai.provider.json_instruction import _SCHEMA_MARKER
from app.ai.provider.json_parsing import parse_permissive, parse_strict
from app.ai.provider.pricing import ModelPrice, cost_usd, price_for, worst_case_price
from app.config import ANTHROPIC_EFFORT_LEVELS, DEFAULT_ANTHROPIC_EFFORT_BY_KIND

_LOGGER = logging.getLogger("app.ai.provider.anthropic")

# Server-side refusal fallback (beta): a turn the requested model declines is
# re-run on a fallback model inside the same call. The scalar
# `fallbacks="default"` form needs exactly this beta version.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# How many attempts one call that carries `fallbacks="default"` can bill: the
# requested model's, then the one fallback model the refusal category is
# routed to. Each runs the same input and may write up to `max_tokens`.
_ATTEMPTS_WITH_FALLBACK = 2

# Call kind per response schema, which selects the effort setting.
_EFFORT_KIND_BY_SCHEMA: dict[str, str] = {
    "ChatDiscussionOutput": "discussion",
    "DiscussionOutput": "discussion",
    "BriefDiscussionOutput": "discussion",
    "VoteOutput": "vote",
    "NightActionOutput": "night",
    "WolfChatOutput": "chat",
}

# The deliberately lowest-common-denominator error classes worth a retry. A
# 529 is an `OverloadedError`, itself an `InternalServerError`; both are
# listed so the intent survives an SDK that splits them.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    OverloadedError,
)
_RETRYABLE_STATUS = frozenset({408, 409})
# A 429 can also mean the organisation hit its monthly spend limit; waiting a
# few seconds cannot fix that, so it is not worth a retry.
_UNRETRYABLE_ERROR_CODES = frozenset({"enforced_spend_limit_reached"})
_RETRY_AFTER_CAP_SECONDS = 30.0
_BACKOFF_BASE_SECONDS = 1.0

# Words that mark a 400 as the structured-output grammar refusing the schema
# (as opposed to, say, a prompt that is too long, which would fail the
# degraded request just the same).
_GRAMMAR_REJECTION_HINTS = ("schema", "complex", "output_config", "format", "json_schema")

# Input tokens are estimated from characters before sending, for the budget's
# worst case. 1.2 tokens per character over-counts both Japanese (about one
# token per character) and English (about a quarter), which is the safe side
# for a reservation; the real count replaces it once the response arrives.
_ESTIMATED_TOKENS_PER_CHAR = 1.2

# An error message is kept whole up to this length. It used to be cut at 200
# characters, which dropped the part of an API validation error that names the
# field at fault; past this it is an echoed request body, not a diagnosis.
_ERROR_TEXT_LIMIT = 2000


# --- process-wide concurrency limiter -------------------------------------

# One semaphore per running event loop. A single module-level semaphore would
# be bound to whichever loop first used it and raise "bound to a different
# event loop" on any other -- pytest-asyncio runs a fresh loop per test, and a
# server could run games on more than one. Weak keys drop a finished loop's
# entry, with one exception: a semaphore that ever had to queue a call keeps a
# reference to its loop, which then stays pinned. That is one small pair of
# objects per such loop -- one per server process, a handful per test run --
# and not worth reaching into asyncio internals to avoid.
_LIMITERS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _shared_limiter(max_concurrency: int) -> asyncio.Semaphore:
    """The limiter for the running loop, created on first use.

    Sized by the `max_concurrency` of the first instance that makes a call on
    this loop; later instances share it whatever they were configured with,
    because one setting (`WEREWOLF_ANTHROPIC_MAX_CONCURRENCY`) configures all
    of them in practice and two sizes for one API key would be meaningless.
    """
    loop = asyncio.get_running_loop()
    limiter = _LIMITERS.get(loop)
    if limiter is None:
        limiter = asyncio.Semaphore(max(1, max_concurrency))
        _LIMITERS[loop] = limiter
    return limiter


# --- per-schema caches -----------------------------------------------------

_STRICT_SCHEMAS: dict[type[BaseModel], dict[str, Any]] = {}
_STRICT_SCHEMA_CHARS: dict[type[BaseModel], int] = {}
_EMBEDDED_SCHEMA_INSTRUCTIONS: dict[type[BaseModel], str] = {}


def _strict_schema(response_schema: type[BaseModel]) -> dict[str, Any]:
    """The transformed JSON schema, computed once per schema class. The same
    dict on every request also keeps the request bytes stable for caching."""
    cached = _STRICT_SCHEMAS.get(response_schema)
    if cached is None:
        cached = anthropic.transform_schema(response_schema)
        _STRICT_SCHEMAS[response_schema] = cached
        _STRICT_SCHEMA_CHARS[response_schema] = len(json.dumps(cached, ensure_ascii=False))
    return cached


def _embedded_schema_instruction(response_schema: type[BaseModel]) -> str:
    """Wording follows `JsonInstructionProvider`: without the grammar, the
    schema has to reach the model as text or it cannot know the field names."""
    cached = _EMBEDDED_SCHEMA_INSTRUCTIONS.get(response_schema)
    if cached is None:
        schema_json = json.dumps(
            response_schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )
        cached = (
            "Return exactly one valid JSON object and no surrounding prose or "
            "Markdown. The object must validate against the JSON Schema below. "
            "Include every required field, use the exact field names and enum "
            "values, and do not add undeclared fields.\n"
            f"{_SCHEMA_MARKER}\n{schema_json}\nEND_RESPONSE_JSON_SCHEMA"
        )
        _EMBEDDED_SCHEMA_INSTRUCTIONS[response_schema] = cached
    return cached


def _with_embedded_schema(system: str, response_schema: type[BaseModel]) -> str:
    if _SCHEMA_MARKER in system:
        return system
    instruction = _embedded_schema_instruction(response_schema)
    return f"{system.rstrip()}\n\n{instruction}" if system.strip() else instruction


# --- ambient SDK inputs ----------------------------------------------------

# Headers the SDK itself sends that the API needs to accept a request, keyed
# by lowercase name, with the SDK's own values. An ambient header of the same
# name has to be pinned back to this value, not cancelled: the SDK applies
# a cancellation to its own value as well, and a request with no
# `anthropic-version` or a non-JSON `Content-Type` is a 400.
_SDK_REQUIRED_HEADERS: dict[str, str] = {
    "anthropic-version": ANTHROPIC_API_VERSION,
    "content-type": "application/json",
    "accept": "application/json",
}


def _ambient_header_overrides(api_key: str) -> dict[str, Any]:
    """Headers that cancel whatever `ANTHROPIC_CUSTOM_HEADERS` would inject.

    The SDK reads that variable on construction and merges it into every
    request, even with an explicit key and base URL; on a machine that runs
    other Anthropic tooling it can carry that tooling's headers -- including
    an `Authorization` that would replace ours. It is read here only to know
    which names to cancel. `Authorization` is pinned back to our own key so
    the request stays authenticated, a header the API requires is pinned back
    to the SDK's own value (`_SDK_REQUIRED_HEADERS`), and every other name
    becomes `omit`. An ambient name that shadows one of the SDK's
    informational headers (`User-Agent`, `X-Stainless-*`, some of them set
    per request) is cancelled with it; the API accepts a request without them.
    """
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
    if not raw:
        return {}
    overrides: dict[str, Any] = {}
    # Parsed exactly the way the SDK parses it, so every name it adds is hit.
    for line in raw.split("\n"):
        colon = line.find(":")
        if colon < 0:
            continue
        name = line[:colon].strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered == "authorization":
            overrides[name] = f"Bearer {api_key}"
        elif lowered in _SDK_REQUIRED_HEADERS:
            overrides[name] = _SDK_REQUIRED_HEADERS[lowered]
        else:
            overrides[name] = anthropic.omit
    return overrides


# --- one logical call ------------------------------------------------------


@dataclass
class _CallTally:
    """Everything one logical call accumulates across its HTTP attempts."""

    http_requests: int = 0
    retries: int = 0
    responses: int = 0
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int | None = None
    cost: float = 0.0
    stop_reason: str | None = None
    refusal_category: str | None = None
    served_model: str | None = None
    request_id: str | None = None
    error: str | None = None
    # For the failure log line: the exception class, or a stop-reason label.
    error_kind: str | None = None

    def fail(self, kind: str, error: str) -> None:
        self.error_kind = kind
        self.error = error[:_ERROR_TEXT_LIMIT]

    def fail_with(self, exc: BaseException) -> None:
        self.fail(type(exc).__name__, f"{type(exc).__name__}: {exc}")
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str):
            self.request_id = request_id

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


@dataclass(frozen=True)
class _BilledAttempt:
    """One model's attempt inside one response, as the API bills it."""

    model: str
    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int


def _billed_attempts(message: BetaMessage, requested_model: str) -> list[_BilledAttempt]:
    """Every attempt a response bills, each with the model that ran it.

    `usage.iterations` is the per-attempt record when the API sends one: with
    a refusal fallback, a `message` entry for the requested model's declined
    attempt and a `fallback_message` entry for the attempt that served. Every
    entry is counted, whatever its type: they do not overlap, and a kind this
    code does not know is still something billed -- counting it errs towards
    the budget stopping early, never towards it being overspent. An entry
    that names no model is priced at the requested model, the one the API
    runs first and the one a declined attempt ran on. Without iterations,
    the top-level usage is the whole bill, run by the model that served it.
    """
    usage = message.usage
    if usage.iterations:
        return [
            _BilledAttempt(
                # Not every entry type declares a model (compaction does not).
                model=str(getattr(entry, "model", None) or requested_model),
                # The SDK builds an entry of an unknown type leniently, so a
                # count it lacks arrives as None despite the annotation.
                input_tokens=entry.input_tokens or 0,
                cache_creation_tokens=entry.cache_creation_input_tokens or 0,
                cache_read_tokens=entry.cache_read_input_tokens or 0,
                output_tokens=entry.output_tokens or 0,
            )
            for entry in usage.iterations
        ]
    return [
        _BilledAttempt(
            model=str(message.model),
            input_tokens=usage.input_tokens,
            cache_creation_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            output_tokens=usage.output_tokens,
        )
    ]


class AnthropicClaudeProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_concurrency: int = 8,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_tokens_floor: int = 12000,
        effort_by_kind: dict[str, str] | None = None,
        fallbacks: bool = True,
        metrics: MetricsCollector | None = None,
        client: AsyncAnthropic | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if client is None:
            if not api_key:
                # An empty key is still an "explicit" credential to the SDK,
                # so it would not fall back to the environment -- it would
                # just fail every call with a 401. Say so now instead.
                raise ValueError("AnthropicClaudeProvider needs a non-empty api_key")
            # SDK retries are off: retrying here makes every real request
            # countable and budgetable, with no hidden second retry layer.
            client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_seconds,
                max_retries=0,
                default_headers=_ambient_header_overrides(api_key),
                # Prompts carry hidden roles; an ambient
                # ANTHROPIC_OPEN_TELEMETRY_* setting must not export them.
                open_telemetry=False,
            )
        self._client = client
        self._model = model
        self._max_concurrency = max_concurrency
        self._max_retries = max(0, max_retries)
        self._max_tokens_floor = max_tokens_floor
        merged_effort = {**DEFAULT_ANTHROPIC_EFFORT_BY_KIND, **(effort_by_kind or {})}
        invalid = {k: v for k, v in merged_effort.items() if v not in ANTHROPIC_EFFORT_LEVELS}
        if invalid:
            raise ValueError(
                f"invalid effort level(s) {invalid}; expected one of "
                f"{sorted(ANTHROPIC_EFFORT_LEVELS)}"
            )
        self._effort_by_kind = merged_effort
        self._fallbacks = fallbacks
        self._metrics = metrics
        self._sleep = sleep
        self._request_budget: EvaluationBudget | None = None
        # Schema class names the grammar compiler refused. Sticky for the life
        # of this provider: the same schema will be refused the same way.
        self._grammar_rejected: set[str] = set()

        if not price_for(model)[1]:
            _LOGGER.warning(
                "no list price for model %r; cost estimates use the most expensive known row",
                model,
            )
        # A refused turn may be re-served by a fallback model, possibly a
        # dearer one, so a reservation must cover the dearest of the models
        # *this* model falls back to (Sonnet 5.5 -> Sonnet 5, not Opus 5).
        self._worst_case_price = worst_case_price(model, fallbacks=fallbacks)

    @property
    def model(self) -> str:
        return self._model

    @property
    def fallbacks_enabled(self) -> bool:
        return self._fallbacks

    @property
    def grammar_rejected_schemas(self) -> frozenset[str]:
        return frozenset(self._grammar_rejected)

    def set_request_budget(self, budget: EvaluationBudget) -> None:
        """Attach the evaluation budget at the actual HTTP boundary."""
        self._request_budget = budget

    def effort_for(self, response_schema: type[BaseModel]) -> str:
        kind = _EFFORT_KIND_BY_SCHEMA.get(response_schema.__name__, "default")
        return self._effort_by_kind[kind]

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        # The API has no sampling parameters: the SDK's create() does not take
        # `temperature`, and the API answers 400 for temperature/top_p/top_k.
        # It stays in the signature only because the Protocol has it.
        del temperature

        started = asyncio.get_running_loop().time()
        effort = self.effort_for(response_schema)
        max_tokens_sent = max(max_tokens, self._max_tokens_floor)
        wire_messages = [{"role": m.role, "content": m.content} for m in messages]
        tally = _CallTally()

        try:
            async with _shared_limiter(self._max_concurrency):
                result, path = await self._run(
                    system, wire_messages, response_schema, max_tokens_sent, effort, tally
                )
        except BudgetExceeded as exc:
            # The stop itself is not a model failure, but whatever this call
            # already spent before the cap was reached has to be on the record.
            if tally.http_requests:
                tally.fail_with(exc)
                self._record(response_schema, ParsePath.FAILED, started, effort, tally)
            raise

        self._record(response_schema, path, started, effort, tally)
        if result is None:
            _LOGGER.warning(
                "anthropic call failed: schema=%s error=%s stop_reason=%s category=%s "
                "request_id=%s http_requests=%d",
                response_schema.__name__,
                tally.error_kind,
                tally.stop_reason,
                tally.refusal_category,
                tally.request_id,
                tally.http_requests,
            )
        return result

    async def _run(
        self,
        system: str,
        wire_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens_sent: int,
        effort: str,
        tally: _CallTally,
    ) -> tuple[SchemaT | None, ParsePath]:
        """The single retry chain of one logical call."""
        name = response_schema.__name__
        while True:
            degraded = name in self._grammar_rejected
            request = self._build_request(
                system, wire_messages, response_schema, max_tokens_sent, effort, degraded=degraded
            )
            estimated_input = self._estimated_input_tokens(request, response_schema, degraded)
            # Decided on this request, not on `self._fallbacks`: a request
            # built with fallbacks can bill a second attempt even if another
            # call has switched them off since.
            attempts = _ATTEMPTS_WITH_FALLBACK if "fallbacks" in request else 1
            worst_case = attempts * cost_usd(
                self._worst_case_price,
                input_tokens=estimated_input,
                output_tokens=max_tokens_sent,
            )
            input_only = cost_usd(self._worst_case_price, input_tokens=estimated_input)

            budget = self._request_budget
            if budget is not None:
                budget.reserve(worst_case)  # BudgetExceeded: nothing was sent
            tally.http_requests += 1
            message: BetaMessage | None = None
            failure: Exception | None = None
            # Until something better is known -- including if this task is
            # cancelled mid-request -- the attempt is assumed to have cost its
            # input: see `_failed_attempt_cost`.
            attempt_cost = input_only
            try:
                message = await self._client.beta.messages.create(**request)
                attempt_cost = self._response_cost(message)
            except Exception as exc:
                failure = exc
                attempt_cost = _failed_attempt_cost(exc, input_only)
            finally:
                tally.cost += attempt_cost
                if budget is not None:
                    budget.settle(worst_case, attempt_cost)

            if message is not None:
                self._tally_response(message, tally)
                return self._interpret(message, response_schema, max_tokens_sent, degraded, tally)

            assert failure is not None
            tally.fail_with(failure)
            if not degraded and _is_grammar_rejection(failure):
                self._remember_grammar_rejection(name, tally.request_id)
                continue  # the degraded request for this same call
            if "fallbacks" in request and _is_fallback_rejection(failure):
                # The beta (or the scalar form) is not available to this key.
                # Every later call would fail the same way, so stop asking.
                # Decided on what this attempt sent, not on the shared switch:
                # every call of a parallel batch was built with fallbacks
                # before the first rejection came back, and each one has to
                # be re-sent without them, not only the first to return.
                if self._fallbacks:
                    self._fallbacks = False
                    _LOGGER.warning(
                        "server-side refusal fallbacks rejected (request_id=%s); "
                        "continuing without them",
                        tally.request_id,
                    )
                continue
            if _is_retryable(failure) and tally.retries < self._max_retries:
                tally.retries += 1
                await self._sleep(_retry_delay(failure, tally.retries))
                continue
            return None, ParsePath.FAILED

    def _build_request(
        self,
        system: str,
        wire_messages: list[dict[str, str]],
        response_schema: type[BaseModel],
        max_tokens_sent: int,
        effort: str,
        *,
        degraded: bool,
    ) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": effort}
        if degraded:
            system = _with_embedded_schema(system, response_schema)
        else:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _strict_schema(response_schema),
            }
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens_sent,
            "messages": wire_messages,
            "output_config": output_config,
        }
        # An empty system prompt is omitted rather than sent as "", which the
        # API may reject as an empty text block.
        if system:
            request["system"] = system
        if self._fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    @staticmethod
    def _estimated_input_tokens(
        request: dict[str, Any], response_schema: type[BaseModel], degraded: bool
    ) -> int:
        chars = len(request.get("system", "")) + sum(len(m["content"]) for m in request["messages"])
        if not degraded:
            # The grammar's schema reaches the model as injected input too.
            chars += _STRICT_SCHEMA_CHARS.get(response_schema, 0)
        return math.ceil(_ESTIMATED_TOKENS_PER_CHAR * chars)

    def _response_cost(self, message: BetaMessage) -> float:
        """USD for one response: every attempt it bills, each at its own model.

        A refusal-fallback turn bills the requested model's attempt -- which
        may have declined after writing its whole output, billed at normal
        rates -- and then the fallback model's attempt. Top-level `usage`
        covers only the attempt that served the turn, so pricing from it alone
        left the declined attempt off the books entirely.
        """
        speed = message.usage.speed
        return sum(
            (
                cost_usd(
                    self._served_price(attempt.model, speed)[0],
                    input_tokens=attempt.input_tokens,
                    cache_creation_tokens=attempt.cache_creation_tokens,
                    cache_read_tokens=attempt.cache_read_tokens,
                    output_tokens=attempt.output_tokens,
                )
                for attempt in _billed_attempts(message, self._model)
            ),
            0.0,
        )

    def _served_price(self, served_model: str, speed: str | None) -> tuple[ModelPrice, bool]:
        price, known = price_for(served_model, speed)
        if not known and served_model not in _UNPRICED_MODELS_WARNED:
            _UNPRICED_MODELS_WARNED.add(served_model)
            _LOGGER.warning(
                "no list price for served model %r (speed=%s); priced at the most expensive "
                "known row",
                served_model,
                speed,
            )
        return price, known

    def _tally_response(self, message: BetaMessage, tally: _CallTally) -> None:
        usage = message.usage
        tally.responses += 1
        # Counted from the same per-attempt record as the cost, so the token
        # totals and the estimated cost always describe the same bill.
        for attempt in _billed_attempts(message, self._model):
            tally.input_tokens += attempt.input_tokens
            tally.cache_creation_tokens += attempt.cache_creation_tokens
            tally.cache_read_tokens += attempt.cache_read_tokens
            tally.output_tokens += attempt.output_tokens
        details = usage.output_tokens_details
        if details is not None:
            tally.thinking_tokens = (tally.thinking_tokens or 0) + (details.thinking_tokens or 0)
        tally.stop_reason = message.stop_reason
        tally.served_model = str(message.model)
        tally.request_id = message._request_id or tally.request_id
        # A response supersedes the transient error of an earlier attempt.
        tally.error = None
        tally.error_kind = None

    def _interpret(
        self,
        message: BetaMessage,
        response_schema: type[SchemaT],
        max_tokens_sent: int,
        degraded: bool,
        tally: _CallTally,
    ) -> tuple[SchemaT | None, ParsePath]:
        if message.stop_reason == "max_tokens":
            # Thinking and answer share this budget. Not retried: the same
            # request would most likely truncate the same way, at full price.
            tally.fail(
                "max_tokens",
                f"response hit max_tokens ({max_tokens_sent}) before the JSON closed",
            )
            return None, ParsePath.FAILED
        if message.stop_reason == "refusal":
            details = message.stop_details
            tally.refusal_category = details.category if details is not None else None
            tally.fail("refusal", f"model refused (category={tally.refusal_category})")
            return None, ParsePath.FAILED

        text = next(
            (block.text for block in message.content if isinstance(block, BetaTextBlock)), None
        )
        if text is None:
            tally.fail("no_text", f"response had no text block (stop_reason={message.stop_reason})")
            return None, ParsePath.FAILED

        if not degraded:
            try:
                return response_schema.model_validate_json(text), ParsePath.STRICT_SCHEMA
            except ValidationError:
                pass
        else:
            direct = parse_strict(text, response_schema)
            if direct is not None:
                return direct, ParsePath.JSON_OBJECT

        salvaged = parse_permissive(text, response_schema)
        if salvaged is not None:
            return salvaged, ParsePath.PERMISSIVE
        tally.fail(
            "ValidationError",
            "strict schema response did not validate"
            if not degraded
            else "response was not parseable as JSON for the schema",
        )
        return None, ParsePath.FAILED

    def _remember_grammar_rejection(self, schema_name: str, request_id: str | None) -> None:
        self._grammar_rejected.add(schema_name)
        _LOGGER.warning(
            "structured-output grammar rejected schema %s (request_id=%s); it is sent with "
            "the schema embedded in the prompt from now on",
            schema_name,
            request_id,
        )

    def _record(
        self,
        response_schema: type[BaseModel],
        path: ParsePath,
        started: float,
        effort: str,
        tally: _CallTally,
    ) -> None:
        if self._metrics is None:
            return
        responded = tally.responses > 0
        self._metrics.record(
            CallRecord(
                schema=response_schema.__name__,
                path=path,
                latency_seconds=asyncio.get_running_loop().time() - started,
                attempt=tally.retries,
                http_requests=tally.http_requests,
                # None (not 0) when no response arrived, so the summary can say
                # "unknown" instead of implying a free call.
                prompt_tokens=tally.prompt_tokens if responded else None,
                completion_tokens=tally.output_tokens if responded else None,
                error=None if path is not ParsePath.FAILED else tally.error,
                finish_reason=_finish_reason(tally.stop_reason),
                reasoning_tokens=tally.thinking_tokens,
                cache_creation_tokens=tally.cache_creation_tokens if responded else None,
                cache_read_tokens=tally.cache_read_tokens if responded else None,
                stop_reason=tally.stop_reason,
                refusal_category=tally.refusal_category,
                served_model=tally.served_model,
                requested_model=self._model,
                request_id=tally.request_id,
                estimated_cost=tally.cost if tally.http_requests else None,
                effort=effort,
            )
        )


_UNPRICED_MODELS_WARNED: set[str] = set()


def _finish_reason(stop_reason: str | None) -> str | None:
    """The provider-neutral spelling `CallRecord.truncated` and the reports read."""
    if stop_reason == "max_tokens":
        return "length"
    return stop_reason


def _failed_attempt_cost(exc: BaseException, input_only: float) -> float:
    """What an attempt that produced no message is assumed to have cost.

    A 4xx is a rejection before any inference ran: free. Anything else --
    a 5xx, a timeout, a dropped connection, a cancellation -- may have been
    processed, at least in part, after we stopped listening (a gateway
    timeout in front of a generation that still completes is billed). The
    attempt is charged its estimated *input*: the least a processed request
    costs. Charging the full worst case would let a burst of timeouts exhaust
    the budget on output that was likely never generated; charging nothing
    would let real spend disappear from the books.
    """
    if isinstance(exc, APIStatusError) and 400 <= exc.status_code < 500:
        return 0.0
    return input_only


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError) and exc.error_code in _UNRETRYABLE_ERROR_CODES:
        return False
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    )


def _rejection_message(exc: BadRequestError) -> str:
    # The API's own message, not the whole "Error code: 400 - {...}" envelope,
    # whose JSON keys could match a hint word by accident.
    return (exc.api_message or str(exc)).lower()


def _is_grammar_rejection(exc: Exception) -> bool:
    if not isinstance(exc, BadRequestError):
        return False
    message = _rejection_message(exc)
    return any(hint in message for hint in _GRAMMAR_REJECTION_HINTS)


def _is_fallback_rejection(exc: Exception) -> bool:
    return isinstance(exc, BadRequestError) and "fallback" in _rejection_message(exc)


def _retry_delay(exc: Exception, retry_number: int) -> float:
    """Seconds to wait before retry number `retry_number` (1-based).

    The server's `retry-after` wins when it sends one (capped, so one bad
    header cannot stall a game for minutes); otherwise 1, 2, 4 ... seconds.
    """
    if isinstance(exc, APIStatusError):
        header = exc.response.headers.get("retry-after")
        if header is not None:
            try:
                seconds = float(header)
            except ValueError:
                seconds = math.nan
            if math.isfinite(seconds) and seconds >= 0:
                return min(seconds, _RETRY_AFTER_CAP_SECONDS)
    return min(_BACKOFF_BASE_SECONDS * 2.0 ** (retry_number - 1), _RETRY_AFTER_CAP_SECONDS)
