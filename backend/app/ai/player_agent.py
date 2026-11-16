"""AIPlayerAgent: the "mouth". Calls the provider, validates the structured
response, applies defense-in-depth safety nets, and guarantees the engine
can never crash or stall on bad LLM output:

  - meta-phrase filter (strips "as an AI"/model-name/"prompt" even though
    structured output already constrains the shape)
  - truncation at a sentence boundary
  - for discussion, a failed full contract falls back once to a
    minimal "just the sentence" request rather than a canned line, so a turn
    still produces something the player actually said
  - any hallucinated/invalid target name is replaced deterministically, from
    the player's own stated beliefs first (see `resolve_target`), so a bad
    response never becomes an unexplainable vote for a random seat

Every one of those repairs is silent at the table, which is the point -- and
also why each is counted (`metrics.record_event`): a game that "worked" because
code stood in for half the ballots is not evidence about the model. The one
failure that is never absorbed is `BudgetExceeded`: a spend cap that degrades
into canned turns would let an evaluation run on past its limit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.ai.context import BRIEF_DISCUSSION_OUTPUT_INSTRUCTION
from app.ai.personalities import Personality, discussion_length_range
from app.ai.provider.base import LLMProvider, Message, SchemaT
from app.ai.provider.budget import BudgetExceeded
from app.ai.reasoning.validation import resolve_target
from app.ai.schemas import (
    BriefDiscussionOutput,
    ChatDiscussionOutput,
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

# Compatibility export for callers that calculate the discussion failure
# ceiling. Agents no longer retry a contract; discussion alone may make one
# additional request using the brief schema.
DEFAULT_MAX_RETRIES = 0

def _discussion_token_budget(max_message_chars: int) -> int:
    """Completion budget for one discussion request, scaled by how long this
    speaker may talk.

    The visible sentence travels inside an envelope -- `DiscussionOutput`'s
    seventeen fields with an eight-field memo, or the v3 wire form's eight with
    a seven-field plan note -- and Japanese costs roughly a token per character.
    A flat 800 fits a terse speaker and truncates a wordy one mid-JSON, which no
    parser can recover: the model appears to type and then say nothing at all.
    The brief fallback reuses the same budget though it needs less. A provider
    may send more than this (a reasoning model's thinking counts against the
    same limit), never less."""
    return _DEFAULT_MAX_TOKENS + max_message_chars * 4


class AIPlayerAgent:
    def __init__(
        self,
        provider: LLMProvider,
        personality: Personality,
        max_retries: int = DEFAULT_MAX_RETRIES,
        *,
        chat_contract: bool = False,
        metrics: object | None = None,
    ) -> None:
        """`chat_contract` selects the v3 wire schema for discussion
        (`ChatDiscussionOutput`): the full `DiscussionOutput` is too complex for
        Anthropic structured outputs to compile. Callers receive a
        `DiscussionOutput` either way.

        `metrics` is any object with a `record_event(name)` method (normally the
        game's `MetricsCollector`); anything else, or None, records nothing."""
        self._provider = provider
        self._personality = personality
        # Kept as a source-compatible argument for existing integrations. HTTP
        # retries belong to the provider; an agent submits each contract once.
        self._max_retries = max_retries
        self._chat_contract = chat_contract
        self._metrics = metrics
        # Whether the last `generate_discussion` turn came from the brief
        # contract. Such a turn is a sentence and nothing else: its memo is
        # empty and its target unset because nobody asked, not because the seat
        # dropped them, and the caller must not store either over the real ones.
        self.last_discussion_was_brief = False

    async def generate_discussion(
        self, system: str, messages: list[Message]
    ) -> DiscussionOutput | None:
        self.last_discussion_was_brief = False
        _minimum, maximum = discussion_length_range(self._personality.verbosity)
        budget = _discussion_token_budget(maximum)
        result: DiscussionOutput | None
        if self._chat_contract:
            wire = await self._generate_with_retry(
                system, messages, ChatDiscussionOutput, max_tokens=budget
            )
            result = wire.to_discussion_output() if wire is not None else None
        else:
            result = await self._generate_with_retry(
                system, messages, DiscussionOutput, max_tokens=budget
            )
        if result is None:
            # The full contract failed every attempt -- overwhelmingly because the
            # JSON was cut off mid-object. Asking for just the sentence still gives
            # the player a real turn; going silent here is what leaves a typing
            # indicator on screen with nothing behind it. Counted when taken; the
            # brief request's own outcome is in the provider's call records.
            self._record_event("brief_fallback")
            brief = await self._generate_with_retry(
                system + "\n\n" + BRIEF_DISCUSSION_OUTPUT_INSTRUCTION,
                messages,
                BriefDiscussionOutput,
                max_tokens=budget,
            )
            if brief is None:
                return None
            result = DiscussionOutput(public_message=brief.public_message)
            self.last_discussion_was_brief = True
        result.public_message = self._sanitize(result.public_message, max_len=maximum)
        return result

    async def generate_morning_intent(
        self, system: str, messages: list[Message]
    ) -> MorningIntentOutput:
        result = await self._generate_with_retry(system, messages, MorningIntentOutput)
        return result or MorningIntentOutput()

    async def generate_vote(
        self,
        system: str,
        messages: list[Message],
        valid_targets: list[str],
        preferred_targets: Sequence[str | None] = (),
    ) -> VoteOutput:
        """`preferred_targets` carries the player's existing beliefs (their
        execution target, then their suspects). A response that names nobody
        votable resolves through those before falling back to the first
        candidate, so the repair follows what this player already argued."""
        result = await self._generate_with_retry(system, messages, VoteOutput)
        if result is None:
            self._record_event("vote_code_resolved")
            resolved = resolve_target(
                None,
                candidates=valid_targets,
                preferred=preferred_targets,
                code="vote_target_invalid",
            )
            return VoteOutput(vote_target=resolved.target if resolved else "")
        if result.alternative_target not in valid_targets or (
            result.alternative_target == result.vote_target
        ):
            result.alternative_target = None
        if result.vote_target not in valid_targets:
            self._record_event("vote_code_resolved")
            resolved = resolve_target(
                result.vote_target,
                candidates=valid_targets,
                preferred=(result.alternative_target, *preferred_targets),
                code="vote_target_invalid",
            )
            if resolved is not None:
                result.vote_target = resolved.target
            result.alternative_target = None
        return result

    async def generate_night_action(
        self,
        system: str,
        messages: list[Message],
        valid_targets: list[str],
        preferred_targets: Sequence[str | None] = (),
    ) -> NightActionOutput:
        result = await self._generate_with_retry(system, messages, NightActionOutput)
        proposed = result.target if result is not None else None
        if result is not None and proposed in valid_targets:
            return result
        self._record_event("night_code_resolved")
        resolved = resolve_target(
            proposed,
            candidates=valid_targets,
            preferred=preferred_targets,
            code="night_target_invalid",
        )
        return NightActionOutput(
            target=resolved.target if resolved else "",
            reason=result.reason if result else "",
        )

    async def generate_wolf_chat(self, system: str, messages: list[Message]) -> WolfChatOutput:
        result = await self._generate_with_retry(system, messages, WolfChatOutput)
        message = self._clean(result.message, max_len=100) if result is not None else ""
        if result is None or not message:
            # No answer, or nothing left once meta phrases were stripped: the
            # teammates read the persona's canned line instead.
            self._record_event("wolf_chat_fallback_line")
            return WolfChatOutput(message=self._personality.get_fallback_message())
        result.message = message
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
        """One request. Any provider failure becomes None -- each caller above
        has a repair for it -- except `BudgetExceeded`, which must reach whoever
        set the budget: swallowed here, an exhausted budget would keep the game
        going on canned turns and the run would never learn it had stopped."""
        try:
            return await self._provider.generate_structured(
                system=system,
                messages=messages,
                response_schema=schema,
                max_tokens=max_tokens,
            )
        except BudgetExceeded:
            raise
        except Exception:
            return None

    def _record_event(self, name: str) -> None:
        record = getattr(self._metrics, "record_event", None)
        if callable(record):
            record(name)

    def _sanitize(self, text: str, max_len: int) -> str:
        return self._clean(text, max_len) or self._personality.get_fallback_message()

    def _clean(self, text: str, max_len: int) -> str:
        """Meta phrases stripped and length capped; may come back empty."""
        for pattern in _META_PATTERNS:
            text = pattern.sub("", text)
        text = text.strip()
        if len(text) > max_len:
            text = self._truncate_at_sentence(text, max_len)
        return text

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
