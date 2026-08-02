"""AIPlayerAgent: the "mouth". Calls the provider, validates the structured
response, applies defense-in-depth safety nets, and guarantees the engine
can never crash or stall on bad LLM output:

  - meta-phrase filter (strips "as an AI"/model-name/"prompt" even though
    structured output already constrains the shape)
  - truncation at a sentence boundary
  - retry loop; for discussion, a failed full contract falls back to a
    minimal "just the sentence" request rather than a canned line, so a turn
    still produces something the player actually said
  - any hallucinated/invalid target name falls back to a random valid
    candidate
"""

from __future__ import annotations

import random
import re

from app.ai.context import BRIEF_DISCUSSION_OUTPUT_INSTRUCTION
from app.ai.personalities import Personality, discussion_length_range
from app.ai.provider.base import LLMProvider, Message, SchemaT
from app.ai.schemas import (
    BriefDiscussionOutput,
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

# Everything except the discussion call answers a small schema, so the historical
# budget is fine for them.
_DEFAULT_MAX_TOKENS = 800

# Attempts per contract are `DEFAULT_MAX_RETRIES + 1`. Exported so tests can derive
# the bounded worst case instead of hardcoding a number that silently goes stale.
DEFAULT_MAX_RETRIES = 2


def _discussion_token_budget(max_message_chars: int) -> int:
    """`DiscussionOutput` carries twelve fields plus an eight-field reasoning memo
    around the visible sentence, and Japanese costs roughly a token per character.
    A flat 800 fits a terse speaker and truncates a wordy one mid-JSON, which no
    parser can recover -- the model appears to type and then say nothing at all.
    Budget for the sentence, the private memo, and the structural overhead."""
    return _DEFAULT_MAX_TOKENS + max_message_chars * 4


class AIPlayerAgent:
    def __init__(
        self,
        provider: LLMProvider,
        personality: Personality,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._provider = provider
        self._personality = personality
        self._max_retries = max_retries

    async def generate_discussion(
        self, system: str, messages: list[Message]
    ) -> DiscussionOutput | None:
        _minimum, maximum = discussion_length_range(self._personality.verbosity)
        result = await self._generate_with_retry(
            system, messages, DiscussionOutput, max_tokens=_discussion_token_budget(maximum)
        )
        if result is None:
            # The full contract failed every attempt -- overwhelmingly because the
            # JSON was cut off mid-object. Asking for just the sentence still gives
            # the player a real turn; going silent here is what leaves a typing
            # indicator on screen with nothing behind it.
            brief = await self._generate_with_retry(
                system + "\n\n" + BRIEF_DISCUSSION_OUTPUT_INSTRUCTION,
                messages,
                BriefDiscussionOutput,
                max_tokens=_discussion_token_budget(maximum),
            )
            if brief is None:
                return None
            result = DiscussionOutput(public_message=brief.public_message)
        result.public_message = self._sanitize(result.public_message, max_len=maximum)
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
        self,
        system: str,
        messages: list[Message],
        schema: type[SchemaT],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> SchemaT | None:
        for _attempt in range(self._max_retries + 1):
            try:
                result = await self._provider.generate_structured(
                    system=system,
                    messages=messages,
                    response_schema=schema,
                    max_tokens=max_tokens,
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
        return truncate_at_sentence(text, max_len)


def truncate_at_sentence(text: str, max_len: int) -> str:
    """Cut to `max_len` at the last sentence boundary, so a shortened line reads
    as a finished sentence instead of stopping mid-word."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    for boundary in _SENTENCE_BOUNDARIES:
        idx = truncated.rfind(boundary)
        if idx != -1:
            return truncated[: idx + 1]
    return truncated
