"""First-class, zero-cost, seedable LLM test double. This is what M1-M3 are
built and tested against: the full e2e test suite and `scripts/dry_run.py`
run with zero network calls and zero API cost."""

from __future__ import annotations

import asyncio
import random
import re
from time import perf_counter

from app.ai.metrics import CallRecord, MetricsCollector, ParsePath
from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import (
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    ReasoningMemo,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)

_PLAYER_ID_RE = re.compile(r"\bp\d+\b")

_MOCK_DISCUSSION_LINES = [
    "みなさんの発言をもう少し聞いてから判断したいです。",
    "昨日の投票の理由が気になっています。詳しく教えてください。",
    "私はまだ誰が怪しいか確信が持てません。",
    "占い師のCOを待ってから動いた方がいいと思います。",
    "村のために慎重に議論を進めましょう。",
]

# A CO line only for the role the prompt says this seat holds, or the fake one
# its team's plan gave it. Picking any CO at random made every seat claim a
# random role 4 times in 9: mock campaigns ran on a board flooded with fake
# freemasons and counter-claims no real model produces, which swamped every
# play-quality measurement taken on them.
_MOCK_CO_LINES: dict[str, tuple[str, str]] = {
    "占い師": ("占い師CO。占い師をやっています。", "seer"),
    "霊媒師": ("霊媒師です。COします。", "medium"),
    "狩人": ("狩人COします。護衛は任せてください。", "hunter"),
    "共有者": ("共有者です。COします。", "freemason"),
}
_OWN_ROLE_RE = re.compile(r"あなたの役職は「(?P<label>[^」]+)」")
_PLANNED_FAKE_RE = re.compile(r"「(?P<label>[^」]+)」を騙")

_MOCK_WOLF_LINES = [
    "今夜は様子を見て潜伏を続けよう。",
    "怪しまれている人を狙うのはどうだろう。",
    "占い師っぽい人を早めに処理したい。",
]


def _claimable_lines(system: str) -> list[tuple[str, str | None]]:
    """The CO lines this seat would plausibly say: its own role, its planned fake."""
    labels = [match.group("label") for match in _OWN_ROLE_RE.finditer(system)]
    labels += [match.group("label") for match in _PLANNED_FAKE_RE.finditer(system)]
    lines: list[tuple[str, str | None]] = []
    for label in dict.fromkeys(labels):
        entry = _MOCK_CO_LINES.get(label)
        if entry is not None:
            lines.append(entry)
    return lines


class MockProvider:
    """Deterministic given a seed; synthesizes schema-valid responses by
    extracting candidate player ids (`p\\d+`) mentioned in the prompt text,
    exactly as a real model would be expected to only ever name a listed
    candidate."""

    def __init__(
        self,
        seed: int | None = None,
        metrics: MetricsCollector | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)
        self._metrics = metrics
        # Lets the evaluation harness rehearse concurrency behaviour without
        # a network; 0 keeps the test suite instant.
        self._latency_seconds = latency_seconds

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        started = perf_counter()
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        text = system + "\n" + "\n".join(m.content for m in messages)
        candidates = sorted(set(_PLAYER_ID_RE.findall(text)))
        pick = self._rng.choice(candidates) if candidates else "p0"

        result: object
        if response_schema is DiscussionOutput:
            options: list[tuple[str, str | None]] = [
                (line, None) for line in _MOCK_DISCUSSION_LINES
            ]
            options.extend(_claimable_lines(system))
            line, claim_role = self._rng.choice(options)
            result = DiscussionOutput(
                public_message=line,
                key_point=line,
                reasoning_memo=ReasoningMemo(overall_thought="モックの思考メモです。"),
                public_claim_role=claim_role,
                contains_co_claim=claim_role is not None,
                ready_to_vote=True,
            )
        elif response_schema is MorningIntentOutput:
            result = MorningIntentOutput()
        elif response_schema is VoteOutput:
            result = VoteOutput(vote_target=pick, reason="モックの投票理由です。")
        elif response_schema is NightActionOutput:
            result = NightActionOutput(target=pick, reason="モックの夜行動理由です。")
        elif response_schema is WolfChatOutput:
            result = WolfChatOutput(message=self._rng.choice(_MOCK_WOLF_LINES))
        elif response_schema is SummaryOutput:
            result = SummaryOutput(summary="モックの要約です。")
        else:
            result = None

        if self._metrics is not None:
            self._metrics.record(
                CallRecord(
                    schema=response_schema.__name__,
                    path=ParsePath.STRICT_SCHEMA if result is not None else ParsePath.FAILED,
                    latency_seconds=perf_counter() - started,
                    attempt=0,
                    # No real endpoint, so no usage to report -- leaving this
                    # None keeps the summary honest about spend being unknown
                    # rather than reporting a fake zero.
                    prompt_tokens=None,
                    completion_tokens=None,
                )
            )
        return result  # type: ignore[return-value]
