"""First-class, zero-cost, seedable LLM test double. This is what M1-M3 are
built and tested against: the full e2e test suite and `scripts/dry_run.py`
run with zero network calls and zero API cost."""

from __future__ import annotations

import random
import re

from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import (
    DiscussionOutput,
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
    "占い師CO。占い師をやっています。",
    "霊媒師です。COします。",
    "狩人COします。護衛は任せてください。",
    "共有者です。COします。",
]

_MOCK_WOLF_LINES = [
    "今夜は様子を見て潜伏を続けよう。",
    "怪しまれている人を狙うのはどうだろう。",
    "占い師っぽい人を早めに処理したい。",
]


class MockProvider:
    """Deterministic given a seed; synthesizes schema-valid responses by
    extracting candidate player ids (`p\\d+`) mentioned in the prompt text,
    exactly as a real model would be expected to only ever name a listed
    candidate."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        text = system + "\n" + "\n".join(m.content for m in messages)
        candidates = sorted(set(_PLAYER_ID_RE.findall(text)))
        pick = self._rng.choice(candidates) if candidates else "p0"

        result: object
        if response_schema is DiscussionOutput:
            result = DiscussionOutput(
                public_message=self._rng.choice(_MOCK_DISCUSSION_LINES),
                reasoning_memo=ReasoningMemo(overall_thought="モックの思考メモです。"),
            )
        elif response_schema is VoteOutput:
            result = VoteOutput(vote_target=pick, reason="モックの投票理由です。")
        elif response_schema is NightActionOutput:
            result = NightActionOutput(target=pick, reason="モックの夜行動理由です。")
        elif response_schema is WolfChatOutput:
            result = WolfChatOutput(message=self._rng.choice(_MOCK_WOLF_LINES))
        elif response_schema is SummaryOutput:
            result = SummaryOutput(summary="モックの要約です。")
        else:
            return None
        return result  # type: ignore[return-value]
