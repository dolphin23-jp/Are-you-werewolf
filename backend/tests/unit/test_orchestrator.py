from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.budget import BudgetExceeded
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import MorningIntentOutput
from app.api import orchestrator
from app.api.orchestrator import after_human_chat
from app.api.ws_hub import SessionWSHub
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.sessions.models import GameSession
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


# -- the spend cap and background failures --
#
# A paid session's provider raises `BudgetExceeded` instead of handing back a
# degraded turn. Each hook must turn that into a flag the view can show, never
# into a 500 for the player's own request or a silently dead task; and once the
# flag is up, no hook may call the coordinator again.


class _RecordingCoordinator:
    """Stands in for `AICoordinator`: records every call, optionally raising."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.calls: list[str] = []
        self.raises = raises
        self._observer_player_ids: set[str] = set()

    def resume_after_human(self, session: object, *args: object, **kwargs: object) -> None:
        del session, args, kwargs
        self.calls.append("resume_after_human")

    async def _call(self, name: str) -> None:
        self.calls.append(name)
        await asyncio.sleep(0)
        if self.raises is not None:
            raise self.raises

    async def advance_discussion(self, session: object) -> None:
        del session
        await self._call("advance_discussion")

    async def run_discussion_round(self, session: object) -> None:
        del session
        await self._call("run_discussion_round")

    async def generate_all_votes(self, session: object) -> None:
        del session
        await self._call("generate_all_votes")

    async def run_night_phase(self, session: object) -> None:
        del session
        await self._call("run_night_phase")

    async def respond_to_private_chat(self, session: object, channel: str) -> None:
        del session, channel
        await self._call("respond_to_private_chat")


def _session(coordinator: object) -> GameSession:
    """A real session whose human is dead, so every hook has AI work to start.

    A dead human never pauses discussion, never owes a vote and has no night
    action, which is what makes the voting and night hooks call the
    coordinator at all. The last chat line is public, for `after_human_chat`.
    """
    controller = make_controller(seed=1)
    controller.chat("p1", "おはよう")
    controller.state.players["p0"].alive = False
    return GameSession(
        session_id="s-orchestrator",
        controller=controller,
        human_id="p0",
        ai_player_ids=[f"p{i}" for i in range(1, 17)],
        ws_hub=SessionWSHub(),
        coordinator=coordinator,
    )


async def _drain_background_tasks() -> None:
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current]
    await asyncio.gather(*pending, return_exceptions=True)


Hook = Callable[[GameSession], Awaitable[None]]

# (id, hook, the coordinator method it drives)
_HOOKS: list[tuple[str, Hook, str]] = [
    ("after_human_chat", orchestrator.after_human_chat, "advance_discussion"),
    ("resume_discussion", orchestrator.resume_discussion, "advance_discussion"),
    (
        "after_discussion_phase_entered",
        orchestrator.after_discussion_phase_entered,
        "run_discussion_round",
    ),
    (
        "after_human_private_chat",
        lambda session: orchestrator.after_human_private_chat(session, "wolf"),
        "respond_to_private_chat",
    ),
    ("after_human_vote", orchestrator.after_human_vote, "generate_all_votes"),
    ("after_voting_phase_entered", orchestrator.after_voting_phase_entered, "generate_all_votes"),
    ("after_human_night_action", orchestrator.after_human_night_action, "run_night_phase"),
    ("after_night_phase_entered", orchestrator.after_night_phase_entered, "run_night_phase"),
]
_HOOK_IDS = [name for name, _, _ in _HOOKS]


@pytest.mark.parametrize(("name", "hook", "method"), _HOOKS, ids=_HOOK_IDS)
async def test_budget_exhaustion_in_any_hook_sets_the_flag_instead_of_raising(
    name: str, hook: Hook, method: str, caplog: pytest.LogCaptureFixture
) -> None:
    coordinator = _RecordingCoordinator(raises=BudgetExceeded("estimated cost budget exceeded"))
    session = _session(coordinator)

    with caplog.at_level(logging.WARNING, logger="app.api.orchestrator"):
        await hook(session)  # must not raise
        await _drain_background_tasks()

    assert method in coordinator.calls
    assert session.ai_budget_exhausted is True
    assert any(
        record.levelno == logging.WARNING
        and "AI budget exhausted" in record.getMessage()
        and "s-orchestrator" in record.getMessage()
        for record in caplog.records
    ), name


@pytest.mark.parametrize(("name", "hook", "method"), _HOOKS, ids=_HOOK_IDS)
async def test_no_hook_calls_the_coordinator_once_the_budget_is_spent(
    name: str, hook: Hook, method: str
) -> None:
    del name, method
    coordinator = _RecordingCoordinator()
    session = _session(coordinator)
    session.ai_budget_exhausted = True

    await hook(session)
    await _drain_background_tasks()

    # Not even the bookkeeping call: the coordinator is left alone entirely.
    assert coordinator.calls == []
    assert session.discussion_advance_task is None


async def test_one_exhaustion_stops_every_later_hook() -> None:
    coordinator = _RecordingCoordinator(raises=BudgetExceeded("cap"))
    session = _session(coordinator)

    await orchestrator.after_human_vote(session)
    assert coordinator.calls == ["generate_all_votes"]

    coordinator.raises = None
    for _, hook, _ in _HOOKS:
        await hook(session)
    await _drain_background_tasks()
    assert coordinator.calls == ["generate_all_votes"]


async def test_a_background_discussion_failure_is_logged_not_lost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = _RecordingCoordinator(raises=RuntimeError("provider exploded"))
    session = _session(coordinator)

    with caplog.at_level(logging.ERROR, logger="app.api.orchestrator"):
        await orchestrator.after_human_chat(session)
        assert session.discussion_advance_task is not None
        await _drain_background_tasks()

    failures = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(failures) == 1
    assert "ai:discussion" in failures[0].getMessage()
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], RuntimeError)
    # Only a spent budget raises the flag; any other failure is a bug to log.
    assert session.ai_budget_exhausted is False
    # The finished task is released, so the next human message starts a new one.
    assert session.discussion_advance_task is None


async def test_a_background_private_chat_failure_is_logged_not_lost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator = _RecordingCoordinator(raises=ValueError("bad wolf chat"))
    session = _session(coordinator)

    with caplog.at_level(logging.ERROR, logger="app.api.orchestrator"):
        await orchestrator.after_human_private_chat(session, "freemason")
        await _drain_background_tasks()

    failures = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(failures) == 1
    assert "ai:freemason chat reply" in failures[0].getMessage()
    assert failures[0].exc_info is not None
    assert isinstance(failures[0].exc_info[1], ValueError)
    assert session.ai_budget_exhausted is False


async def test_a_cancelled_background_task_is_not_reported_as_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session(_BlockingCoordinator())

    with caplog.at_level(logging.ERROR, logger="app.api.orchestrator"):
        await orchestrator.after_human_chat(session)
        task = session.discussion_advance_task
        assert task is not None
        await asyncio.sleep(0)
        task.cancel()
        await _drain_background_tasks()

    assert task.cancelled()
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


async def test_an_awaited_hook_still_raises_anything_but_a_spent_budget() -> None:
    """Only the cap is absorbed. A real bug in a vote or night hook keeps failing
    the request loudly, as before, rather than leaving a game silently stuck."""
    coordinator = _RecordingCoordinator(raises=RuntimeError("bug"))
    session = _session(coordinator)

    with pytest.raises(RuntimeError, match="bug"):
        await orchestrator.after_human_vote(session)
    assert session.ai_budget_exhausted is False


# -- the table moving on while the morning is still being drained --


class _HeldDiscussionProvider:
    """Holds every discussion request until released, so the table can move on."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if "Discussion" in response_schema.__name__:
            self.started.set()
            await self.release.wait()
            return response_schema.model_validate({"public_message": "まだ決め手がありません。"})
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        return None


def _morning_with_several_duty_speakers() -> GameController:
    """A real day 1 whose opening has more than one duty speaker: an AI seer
    holding its night-0 look plus the planned claims. Seed 1 opens with three."""
    controller = make_controller(seed=1)
    state = controller.state
    controller.start_game()
    seer = next(p.player_id for p in state.players.values() if p.role is RoleName.SEER)
    target = next(pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id))
    controller.submit_night_action(seer, "divine", target)
    controller.resolve_night()
    controller.start_discussion()
    return controller


async def test_ending_the_discussion_mid_morning_lets_the_opening_return() -> None:
    """The vote opened while a morning duty speaker was still generating. The
    turn is dropped, so the morning stage never closes -- and the opening loop
    used to call `advance_discussion` again and again on a round that could not
    move. Nothing in that call awaits once the phase has left DISCUSSION, so the
    loop never yielded and froze every session on the server."""
    controller = _morning_with_several_duty_speakers()
    state = controller.state
    ai_ids = [f"p{i}" for i in range(1, 17)]
    provider = _HeldDiscussionProvider()
    coordinator = AICoordinator(
        state,
        ai_ids,
        provider,
        seed=1,
        reasoning=ReasoningRuntime(state, ai_ids, seed=1),
        model_decides=True,
    )
    session = GameSession(
        session_id="s-morning",
        controller=controller,
        human_id="p0",
        ai_player_ids=ai_ids,
        ws_hub=SessionWSHub(),
        coordinator=coordinator,
    )
    assert state.players["p0"].alive  # a living human: the opening drains, then pauses
    calls = 0
    real_advance = coordinator.advance_discussion

    async def counted_advance(session: object, **kwargs: bool) -> None:
        # Without this bound the old loop never yields, so `wait_for` below
        # could not even time out: the test would hang instead of failing.
        nonlocal calls
        calls += 1
        if calls > 20:
            raise AssertionError("the opening kept re-entering a round that cannot move")
        await real_advance(session, **kwargs)

    coordinator.advance_discussion = counted_advance  # type: ignore[method-assign]

    opening = asyncio.create_task(orchestrator.after_discussion_phase_entered(session))
    await asyncio.wait_for(provider.started.wait(), timeout=5)
    round_state = session.discussion_round
    assert round_state is not None and round_state.immediate_count > 1
    posted = len(state.chat_log)
    controller.end_discussion()
    provider.release.set()

    await asyncio.wait_for(opening, timeout=5)

    assert state.phase is Phase.VOTING
    assert len(state.chat_log) == posted
    assert calls == 1
    # Nothing is left that a later call could mistake for a live morning.
    assert session.discussion_round is round_state and round_state.complete
