"""Overlapping requests must not run the AI rounds twice or post after a phase ends."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.provider.mock import MockProvider
from app.ai.schemas import DiscussionOutput
from app.engine.phases import Phase
from app.engine.state import ChatChannel
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


def _session(controller, coordinator) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        controller=controller,
        coordinator=coordinator,
        human_id="p0",
        discussion_lock=asyncio.Lock(),
        discussion_round=None,
        discussion_paused=False,
        discussion_pause_requested=False,
        discussion_step_budget=None,
    )


def test_two_vote_requests_share_one_round_of_ai_votes():
    controller = make_controller(seed=1)
    coordinator = AICoordinator(
        controller.state, AI_IDS, MockProvider(seed=1, latency_seconds=0.01), seed=1,
        pacing_scale=0.0,
    )
    session = _session(controller, coordinator)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    state = controller.state
    if state.players["p0"].alive:
        controller.vote("p0", next(pid for pid in state.votable_ids("p0")))
    casts: list[str] = []
    original = coordinator._cast_vote

    async def counting(controller_, state_, player_id):  # type: ignore[no-untyped-def]
        casts.append(player_id)
        await original(controller_, state_, player_id)

    coordinator._cast_vote = counting  # type: ignore[method-assign]
    alive_ai = [pid for pid in AI_IDS if state.players[pid].alive]

    async def double_submit() -> None:
        await asyncio.gather(
            coordinator.generate_all_votes(session), coordinator.generate_all_votes(session)
        )

    asyncio.run(double_submit())

    # One ballot per AI per round, however many rounds the vote needed.
    rounds = {record.round for record in state.vote_records if record.day == state.day}
    assert len(casts) == len(alive_ai) * len(rounds)


class _EndsDiscussionWhileGenerating:
    """Stands in for `/end-discussion` landing while a turn is being generated."""

    def __init__(self, controller) -> None:  # type: ignore[no-untyped-def]
        self.controller = controller

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is DiscussionOutput:
            if self.controller.state.phase is Phase.DISCUSSION:
                self.controller.end_discussion()
            return DiscussionOutput(public_message="占い師CO。ユイは人狼でした。")
        return None


def test_a_turn_generated_across_the_end_of_discussion_is_not_posted():
    controller = make_controller(seed=1)
    coordinator = AICoordinator(
        controller.state,
        AI_IDS,
        _EndsDiscussionWhileGenerating(controller),
        seed=1,
        pacing_scale=0.0,
    )
    session = _session(controller, coordinator)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()

    asyncio.run(coordinator.advance_discussion(session, allow_human_pause=False))

    assert controller.state.phase is Phase.VOTING
    assert not [m for m in controller.state.chat_log if m.channel is ChatChannel.PUBLIC]
    assert controller.state.co_declarations == ()
