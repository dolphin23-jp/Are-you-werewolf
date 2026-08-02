"""Glue between human actions and AI turns.

Without an AI coordinator attached (`session.coordinator is None`), every
hook here is a no-op, so a game can still be driven end-to-end via the REST
API by explicitly supplying `player_id` for every seat (useful for
curl/Swagger testing with no LLM involved at all).

Once `session.coordinator` (an `AICoordinator`) is attached, these hooks
trigger AI turns around each human action: discussion is fire-and-forget
(the human doesn't block on it), while votes/night-action completion is
awaited before resolving, matching what the human is already waiting on.

"""

from __future__ import annotations

import asyncio

from app.engine.roles import RoleName
from app.sessions.models import GameSession


async def after_human_chat(session: GameSession) -> None:
    if session.coordinator is None:
        return
    latest = session.controller.state.chat_log[-1]
    session.coordinator.resume_after_human(session, latest.reply_to)
    asyncio.create_task(session.coordinator.advance_discussion(session))


async def after_discussion_phase_entered(session: GameSession) -> None:
    """Run morning COs immediately; dead/observer humans never pause the round."""
    if session.coordinator is None:
        return
    human = session.controller.state.players[session.human_id]
    observer = session.human_id in session.coordinator._observer_player_ids
    if not human.alive or observer:
        await session.coordinator.run_discussion_round(session)
    else:
        # There may be more immediate COs than fit in one segment. Drain only
        # those opening segments, releasing the lock between each, then pause.
        while True:
            await session.coordinator.advance_discussion(session)
            round_state = session.discussion_round
            if (
                round_state is None
                or round_state.complete
                or round_state.awaiting_human
                or round_state.stage != "immediate"
            ):
                break


async def after_human_private_chat(session: GameSession, channel: str) -> None:
    if session.coordinator is not None:
        asyncio.create_task(session.coordinator.respond_to_private_chat(session, channel))


async def after_human_vote(session: GameSession) -> None:
    if session.coordinator is None:
        return
    await session.coordinator.generate_all_votes(session)


async def after_voting_phase_entered(session: GameSession) -> None:
    """Call right after DISCUSSION -> VOTING. If the human is dead (or, in
    principle, otherwise unable to vote), AI votes must still run without
    waiting on a human vote that will never arrive."""
    if session.coordinator is None:
        return
    human = session.controller.state.players[session.human_id]
    if not human.alive:
        await session.coordinator.generate_all_votes(session)


async def after_human_night_action(session: GameSession) -> None:
    if session.coordinator is None:
        return
    await session.coordinator.run_night_phase(session)


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
    if session.coordinator is None:
        return
    if not _human_has_pending_night_action(session):
        await session.coordinator.run_night_phase(session)
