"""The judge stops cleanly when the spending cap is reached."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.ai.provider.base import Message
from app.ai.provider.budget import BudgetExceeded
from app.eval.judge import PersonaScore, judge_transcript
from app.eval.transcript import GameTranscript, Utterance


def _transcript(players: int) -> GameTranscript:
    transcript = GameTranscript()
    for index in range(players):
        pid = f"p{index}"
        transcript.personalities[pid] = "冷静"
        transcript.utterances.append(
            Utterance(
                day=1,
                phase="discussion",
                kind="discussion",
                player_id=pid,
                player_name=pid,
                role="villager",
                team="village",
                personality="冷静",
                deception_role=None,
                text="よろしくお願いします。",
            )
        )
    return transcript


class _CapAfterOne:
    """The first call answers; the second hits the cap; the rest would hang."""

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled = 0

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: Any, **_: Any
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            return PersonaScore(naturalness=4, persona_consistency=4)
        if self.calls == 2:
            raise BudgetExceeded("estimated cost budget would be exceeded")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return None


def test_a_budget_stop_cancels_the_other_players_calls_and_propagates() -> None:
    provider = _CapAfterOne()

    async def run() -> None:
        await judge_transcript(_transcript(4), provider, max_concurrency=4)
        raise AssertionError("BudgetExceeded was swallowed")

    with pytest.raises(BudgetExceeded):
        asyncio.run(asyncio.wait_for(run(), timeout=5))

    assert provider.calls == 4
    # The two calls still "in flight" were cancelled and awaited before the
    # error reached the caller, so none of them can settle after the report.
    assert provider.cancelled == 2


def test_scores_are_summarised_when_nothing_goes_wrong() -> None:
    class _Fixed:
        async def generate_structured(self, **_: Any) -> PersonaScore:
            return PersonaScore(naturalness=3, persona_consistency=5, comment="ok")

    result = asyncio.run(judge_transcript(_transcript(3), _Fixed()))

    assert result.summary["judged_players"] == 3
    assert result.summary["naturalness_mean"] == 3
    assert result.summary["persona_consistency_min"] == 5
