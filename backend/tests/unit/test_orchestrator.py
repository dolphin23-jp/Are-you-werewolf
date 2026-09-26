from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api.orchestrator import after_discussion_phase_entered, after_human_chat
from app.engine.phases import Phase
from app.sessions.models import DiscussionRoundState
from tests.conftest import make_controller


class _BlockingCoordinator:
    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()

    def resume_after_human(
        self,
        session: object,
        reply_to: str | None,
        references: list[str] | None = None,
        *,
        release_wait: bool = True,
    ) -> None:
        del session, reply_to, references, release_wait

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


class _StalledImmediateCoordinator:
    """Leaves the round in the "immediate" stage without awaiting anything.

    That is what the real coordinator does once the phase has left DISCUSSION:
    `advance_discussion` sees the phase and returns straight away. `end_after`
    simulates `/end-discussion` landing while the first segment was generating.
    """

    def __init__(self, *, end_after: int | None) -> None:
        self._observer_player_ids: set[str] = set()
        self.calls = 0
        self._end_after = end_after

    async def advance_discussion(self, session: SimpleNamespace) -> None:
        self.calls += 1
        if self.calls > 50:
            raise AssertionError("orchestrator kept re-entering a stalled discussion")
        if session.discussion_round is None:
            session.discussion_round = DiscussionRoundState(
                day=1, order=["p1", "p2"], immediate_count=2
            )
        if self.calls == self._end_after:
            session.controller.end_discussion()


def _discussion_session(coordinator: object) -> SimpleNamespace:
    controller = make_controller(seed=1)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    return SimpleNamespace(
        controller=controller,
        coordinator=coordinator,
        human_id="p0",
        discussion_round=None,
    )


def test_immediate_drain_stops_when_discussion_ends_mid_segment():
    coordinator = _StalledImmediateCoordinator(end_after=1)
    session = _discussion_session(coordinator)

    asyncio.run(after_discussion_phase_entered(session))

    assert session.controller.state.phase == Phase.VOTING
    assert coordinator.calls == 1


def test_immediate_drain_stops_when_a_segment_makes_no_progress():
    coordinator = _StalledImmediateCoordinator(end_after=None)
    session = _discussion_session(coordinator)

    asyncio.run(after_discussion_phase_entered(session))

    assert session.discussion_round.stage == "immediate"
    assert coordinator.calls == 2
