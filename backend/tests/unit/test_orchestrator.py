from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api.orchestrator import after_human_chat
from tests.conftest import make_controller


class _BlockingCoordinator:
    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()

    def resume_after_human(
        self, session: object, reply_to: str | None, *, release_wait: bool = True
    ) -> None:
        del session, reply_to, release_wait

    async def advance_discussion(self, session: object) -> None:
        del session
        self.calls += 1
        await self.release.wait()


def test_rapid_human_messages_share_one_discussion_advance_task():
    async def scenario() -> None:
        controller = make_controller(seed=1)
        controller.chat("p0", "one")
        coordinator = _BlockingCoordinator()
        session = SimpleNamespace(
            controller=controller,
            coordinator=coordinator,
            discussion_advance_task=None,
        )

        await after_human_chat(session)
        await asyncio.sleep(0)
        await after_human_chat(session)
        await after_human_chat(session)

        assert coordinator.calls == 1
        coordinator.release.set()
        await session.discussion_advance_task

    asyncio.run(scenario())
