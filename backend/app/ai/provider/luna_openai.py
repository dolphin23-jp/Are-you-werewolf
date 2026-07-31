"""OpenAI-compatible client for gpt-5.6-luna.

NOTE: the real gpt-5.6-luna endpoint's exact request/response shape and
structured-output conformance are unconfirmed at the time this was written
(placeholder base URL/model name in `.env.example`, pending real details
from the provider). This client is written defensively:

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
"""

from __future__ import annotations

import asyncio
import re

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.ai.provider.base import Message, SchemaT

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class LunaOpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_concurrency: int = 6,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self._model = model
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)

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

        async with self._semaphore:
            for _attempt in range(self._max_retries + 1):
                result = await self._try_strict_schema(
                    openai_messages, response_schema, max_tokens, temperature
                )
                if result is not None:
                    return result

                result = await self._try_json_object_mode(
                    openai_messages, response_schema, max_tokens, temperature
                )
                if result is not None:
                    return result

        return None

    async def _try_strict_schema(
        self,
        openai_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens: int,
        temperature: float,
    ) -> SchemaT | None:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.__name__,
                        "schema": response_schema.model_json_schema(),
                        "strict": True,
                    },
                },
            )  # type: ignore[call-overload]
        except Exception:
            return None
        content = response.choices[0].message.content if response.choices else None
        return _parse_strict(content, response_schema)

    async def _try_json_object_mode(
        self,
        openai_messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        max_tokens: int,
        temperature: float,
    ) -> SchemaT | None:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )  # type: ignore[call-overload]
        except Exception:
            return None
        content = response.choices[0].message.content if response.choices else None
        return _parse_permissive(content, response_schema)


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
