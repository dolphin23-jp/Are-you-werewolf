"""AIPlayerAgent: the "mouth". Calls the provider, validates the structured
response, applies defense-in-depth safety nets, and guarantees the engine
can never crash or stall on bad LLM output:

  - meta-phrase filter (strips "as an AI"/model-name/"prompt" even though
    structured output already constrains the shape)
  - truncation at a sentence boundary
  - retry loop with a personality-flavored canned fallback line on total
    failure
  - any hallucinated/invalid target name falls back to a random valid
    candidate
"""

from __future__ import annotations

import random
import re

from app.ai.personalities import Personality
from app.ai.provider.base import LLMProvider, Message, SchemaT
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)

_META_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"AIとして",
        r"言語モデル",
        r"生成AI",
        r"アシスタントとして",
        r"claude",
        r"gpt",
        r"openai",
        r"anthropic",
        r"プロンプト",
        r"システムプロンプト",
    ]
]

_SENTENCE_BOUNDARIES = "。！？"


class AIPlayerAgent:
    def __init__(
        self, provider: LLMProvider, personality: Personality, max_retries: int = 2
    ) -> None:
        self._provider = provider
        self._personality = personality
        self._max_retries = max_retries

    async def generate_discussion(
        self, system: str, messages: list[Message]
    ) -> DiscussionOutput | None:
        result = await self._generate_with_retry(system, messages, DiscussionOutput)
        if result is None:
            return None
        limits = {"terse": 100, "normal": 240, "wordy": 400}
        result.public_message = self._sanitize(
            result.public_message, max_len=limits.get(self._personality.verbosity, 240)
        )
        return result

    async def generate_morning_intent(
        self, system: str, messages: list[Message]
    ) -> MorningIntentOutput:
        result = await self._generate_with_retry(system, messages, MorningIntentOutput)
        return result or MorningIntentOutput()

    async def generate_vote(
        self, system: str, messages: list[Message], valid_targets: list[str]
    ) -> VoteOutput:
        result = await self._generate_with_retry(system, messages, VoteOutput)
        if result is None or result.vote_target not in valid_targets:
            reason = result.reason if result else ""
            return VoteOutput(vote_target=random.choice(valid_targets), reason=reason)
        return result

    async def generate_night_action(
        self, system: str, messages: list[Message], valid_targets: list[str]
    ) -> NightActionOutput:
        result = await self._generate_with_retry(system, messages, NightActionOutput)
        if result is None or result.target not in valid_targets:
            reason = result.reason if result else ""
            return NightActionOutput(target=random.choice(valid_targets), reason=reason)
        return result

    async def generate_wolf_chat(self, system: str, messages: list[Message]) -> WolfChatOutput:
        result = await self._generate_with_retry(system, messages, WolfChatOutput)
        if result is None:
            return WolfChatOutput(message=self._personality.get_fallback_message())
        result.message = self._sanitize(result.message, max_len=100)
        return result

    async def generate_summary(self, system: str, messages: list[Message]) -> SummaryOutput:
        result = await self._generate_with_retry(system, messages, SummaryOutput)
        if result is None:
            return SummaryOutput(summary="(要約の生成に失敗しました)")
        return result

    async def _generate_with_retry(
        self, system: str, messages: list[Message], schema: type[SchemaT]
    ) -> SchemaT | None:
        for _attempt in range(self._max_retries + 1):
            try:
                result = await self._provider.generate_structured(
                    system=system, messages=messages, response_schema=schema
                )
            except Exception:
                result = None
            if result is not None:
                return result
        return None

    def _sanitize(self, text: str, max_len: int) -> str:
        for pattern in _META_PATTERNS:
            text = pattern.sub("", text)
        text = text.strip()
        if len(text) > max_len:
            text = self._truncate_at_sentence(text, max_len)
        return text or self._personality.get_fallback_message()

    @staticmethod
    def _truncate_at_sentence(text: str, max_len: int) -> str:
        truncated = text[:max_len]
        for boundary in _SENTENCE_BOUNDARIES:
            idx = truncated.rfind(boundary)
            if idx != -1:
                return truncated[: idx + 1]
        return truncated
