"""Glue between human actions and AI turns.

Without an AI coordinator attached (`session.coordinator is None`), every
hook here is a no-op, so a game can still be driven end-to-end via the REST
API by explicitly supplying `player_id` for every seat (useful for
curl/Swagger testing with no LLM involved at all).

Once `session.coordinator` (an `AICoordinator`) is attached, these hooks
trigger AI turns around each human action: discussion is fire-and-forget
(the human doesn't block on it), while votes/night-action completion is
awaited before resolving, matching what the human is already waiting on.

Every coordinator call goes through `_drive`, for two reasons:

- A paid session has a spend cap, and the provider raises `BudgetExceeded`
  rather than hand back a degraded turn. Uncaught, that would turn the
  player's own vote or chat request into a 500, or vanish inside a
  background task. `_drive` records it on the session
  (`ai_budget_exhausted`), which the view shows, and stops calling the
  coordinator from then on -- every further call would only be refused.
- A background task's failure is otherwise never seen: nobody awaits it, so
  its exception is lost. `_spawn` logs it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.ai.provider.budget import BudgetExceeded
from app.engine.roles import RoleName
from app.sessions.models import GameSession

_LOGGER = logging.getLogger(__name__)

# Fire-and-forget tasks the event loop would otherwise hold only weakly: a
# task nobody references can be garbage-collected before it finishes.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _live_coordinator(session: GameSession) -> Any | None:
    """The coordinator to drive, or None when there is none or its budget is spent."""
    if session.coordinator is None or getattr(session, "ai_budget_exhausted", False):
        return None
    return session.coordinator


async def _drive(
    session: GameSession, what: str, work: Callable[[], Awaitable[None]]
) -> None:
    """Run one coordinator call; a spent budget stops the AI, not the request.

    `work` is a factory rather than a coroutine so that nothing is created --
    and left un-awaited -- when the budget is already spent.
    """
    if getattr(session, "ai_budget_exhausted", False):
        return
    try:
        await work()
    except BudgetExceeded as exc:
        session.ai_budget_exhausted = True
        _LOGGER.warning(
            "session %s: AI budget exhausted during %s; AI turns stop for this session (%s)",
            getattr(session, "session_id", "?"),
            what,
            exc,
        )


def _spawn(
    session: GameSession, what: str, work: Callable[[], Awaitable[None]]
) -> asyncio.Task[None]:
    """Start `work` in the background through `_drive`, logging any other failure."""
    task = asyncio.create_task(_drive(session, what, work), name=f"ai:{what}")
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    task.add_done_callback(_log_task_failure)
    return task


def _log_task_failure(task: asyncio.Task[None]) -> None:
    """Surface what an un-awaited task raised; reading it also stops asyncio's
    own "exception was never retrieved" noise at garbage collection."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _LOGGER.exception("background AI task %s failed", task.get_name(), exc_info=exc)


async def after_human_chat(session: GameSession) -> None:
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    latest = session.controller.state.chat_log[-1]
    paused = getattr(session, "discussion_paused", False)
    coordinator.resume_after_human(
        session, latest.reply_to, latest.references, release_wait=not paused
    )
    if paused:
        return
    if session.discussion_advance_task is not None and not session.discussion_advance_task.done():
        return
    task = _spawn(session, "discussion", lambda: coordinator.advance_discussion(session))
    session.discussion_advance_task = task

    def clear_task(completed: asyncio.Task[None]) -> None:
        if session.discussion_advance_task is completed:
            session.discussion_advance_task = None

    task.add_done_callback(clear_task)


async def resume_discussion(session: GameSession) -> None:
    """The human passed, or lifted a pause: release the wait and advance one segment."""
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    coordinator.resume_after_human(session)
    await _drive(session, "discussion", lambda: coordinator.advance_discussion(session))


async def after_discussion_phase_entered(session: GameSession) -> None:
    """Run morning COs immediately; dead/observer humans never pause the round."""
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return

    async def open_discussion() -> None:
        human = session.controller.state.players[session.human_id]
        observer = session.human_id in coordinator._observer_player_ids
        if not human.alive or observer:
            await coordinator.run_discussion_round(session)
            return
        # There may be more immediate COs than fit in one segment. Drain only
        # those opening segments, releasing the lock between each, then pause.
        while True:
            await coordinator.advance_discussion(session)
            round_state = session.discussion_round
            if (
                round_state is None
                or round_state.complete
                or round_state.awaiting_human
                or round_state.stage != "immediate"
            ):
                break

    await _drive(session, "discussion opening", open_discussion)


async def after_human_private_chat(session: GameSession, channel: str) -> None:
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    _spawn(
        session,
        f"{channel} chat reply",
        lambda: coordinator.respond_to_private_chat(session, channel),
    )


async def after_human_vote(session: GameSession) -> None:
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    await _drive(session, "votes", lambda: coordinator.generate_all_votes(session))


async def after_voting_phase_entered(session: GameSession) -> None:
    """Call right after DISCUSSION -> VOTING. If the human is dead (or, in
    principle, otherwise unable to vote), AI votes must still run without
    waiting on a human vote that will never arrive."""
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    human = session.controller.state.players[session.human_id]
    if not human.alive:
        await _drive(session, "votes", lambda: coordinator.generate_all_votes(session))


async def after_human_night_action(session: GameSession) -> None:
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    await _drive(session, "night", lambda: coordinator.run_night_phase(session))


def _human_has_pending_night_action(session: GameSession) -> bool:
    state = session.controller.state
    human = state.players.get(session.human_id)
    if human is None or not human.alive:
        return False
    if state.day == 0:
        return human.role == RoleName.SEER
    if human.role in (RoleName.SEER, RoleName.HUNTER):
        return True
    if human.role == RoleName.WEREWOLF and session.human_id == session.controller.alpha_wolf_id:
        return True
    return False


async def after_night_phase_entered(session: GameSession) -> None:
    """Call right after entering NIGHT (start_game or start_night). If the
    human has no possible night action this night, there is nothing to wait
    on, so run the AI night phase immediately instead of stalling on a
    human input that will never come."""
    coordinator = _live_coordinator(session)
    if coordinator is None:
        return
    if not _human_has_pending_night_action(session):
        await _drive(session, "night", lambda: coordinator.run_night_phase(session))
