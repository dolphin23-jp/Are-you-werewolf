from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import DirectedQuestion, DiscussionOutput, MorningIntentOutput
from app.engine.phases import Phase
from tests.conftest import make_controller


class ReplyLoopProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del system, messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()  # type: ignore[return-value]
        assert response_schema is DiscussionOutput
        self.calls += 1
        lines = {
            1: "まず意見を述べます。",
            2: "Player1さん、理由を説明してください。",
        }
        line = lines.get(self.calls, "Player2さんも説明してください。")
        target = "p1" if self.calls == 2 else "p2"
        return DiscussionOutput(
            public_message=line,
            directed_questions=[DirectedQuestion(target_id=target, question="説明して")],
        )  # type: ignore[return-value]


def test_direct_reply_queue_is_globally_bounded_and_returns_control():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    provider = ReplyLoopProvider()
    coordinator = AICoordinator(
        controller.state,
        ["p1", "p2"],
        provider,
        seed=1,
        max_discussion_followups=1,
    )
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    assert provider.calls <= 5
    assert [message.author_id for message in controller.state.chat_log][:3] == ["p1", "p2", "p1"]


class MorningPriorityProvider:
    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[Message],
        response_schema: type[SchemaT],
        max_tokens: int = 800,
        temperature: float = 0.9,
    ) -> SchemaT | None:
        del messages, max_tokens, temperature
        if response_schema is MorningIntentOutput:
            timing = "immediate" if "プレイヤー「Player2」" in system else "normal"
            return MorningIntentOutput(timing=timing)  # type: ignore[return-value]
        return DiscussionOutput(public_message="見解を述べます。", ready_to_vote=True)  # type: ignore[return-value]


def test_immediate_morning_intent_speaks_before_normal_seating_order():
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    coordinator = AICoordinator(
        controller.state,
        ["p1", "p2"],
        MorningPriorityProvider(),
        seed=1,
    )
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    assert [message.author_id for message in controller.state.chat_log][:2] == ["p2", "p1"]
