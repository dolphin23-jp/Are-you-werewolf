from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.schemas import DiscussionOutput
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
        assert response_schema is DiscussionOutput
        self.calls += 1
        lines = {
            1: "まず意見を述べます。",
            2: "Player1さん、理由を説明してください。",
            3: "Player2さんも説明してください。",
        }
        return DiscussionOutput(public_message=lines[self.calls])  # type: ignore[return-value]


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

    assert provider.calls == 3
    assert [message.author_id for message in controller.state.chat_log] == ["p1", "p2", "p1"]
