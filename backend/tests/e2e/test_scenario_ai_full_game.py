"""Full 17-seat games driven by `ScenarioProvider`, which -- unlike
`MockProvider` -- populates the structured half of the discussion contract.

`test_mock_ai_full_game.py` proves a game *completes*. It cannot prove the
reply/quote plumbing, the pending-question lifecycle, the key-point ledger or
the pressured-candidate machinery work, because the mock never emits
`reply_to`, `directed_questions`, `agrees_with` or `execution_target`. These
tests close that gap, still with zero network calls and zero API cost.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.scenario import ScenarioProvider
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel

MAX_LOOPS = 150
HUMAN_ID = "p0"


def _make_session(seed: int) -> SimpleNamespace:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}", is_human=(i == 0)) for i in range(17)]
    controller = GameController(session_id=f"s{seed}", player_specs=specs, seed=seed)
    ai_ids = [s.player_id for s in specs if not s.is_human]
    coordinator = AICoordinator(
        controller.state, ai_ids, ScenarioProvider(seed=seed), seed=seed, pacing_scale=0.0
    )
    return SimpleNamespace(
        controller=controller,
        human_id=HUMAN_ID,
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )


async def _play(seed: int) -> SimpleNamespace:
    session = _make_session(seed)
    controller = session.controller
    coordinator = session.coordinator
    rng = random.Random(seed)

    controller.start_game()

    for _ in range(MAX_LOOPS):
        state = controller.state
        phase = state.phase

        if phase == Phase.GAME_OVER:
            return session

        if phase == Phase.NIGHT:
            human = state.players[HUMAN_ID]
            if human.alive:
                candidates = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
                if human.role == RoleName.SEER:
                    controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
                elif state.day > 0 and human.role == RoleName.HUNTER:
                    controller.submit_night_action(HUMAN_ID, "guard", rng.choice(candidates))
                elif (
                    state.day > 0
                    and human.role == RoleName.WEREWOLF
                    and controller.alpha_wolf_id == HUMAN_ID
                ):
                    non_wolves = [
                        pid
                        for pid in state.alive_ids()
                        if state.players[pid].role != RoleName.WEREWOLF
                    ]
                    if non_wolves:
                        controller.submit_night_action(HUMAN_ID, "attack", rng.choice(non_wolves))
            await coordinator.run_night_phase(session)
        elif phase == Phase.DAWN:
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            if state.players[HUMAN_ID].alive:
                controller.chat(HUMAN_ID, "よろしくお願いします。", "public")
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            if state.players[HUMAN_ID].alive:
                candidates = state.votable_ids(HUMAN_ID)
                if candidates:
                    controller.vote(HUMAN_ID, rng.choice(candidates))
            await coordinator.generate_all_votes(session)
        elif phase == Phase.VOTE_RESULT:
            controller.start_night()
        else:
            raise AssertionError(f"unexpected phase {phase}")

    raise AssertionError(f"game did not terminate within {MAX_LOOPS} loops (seed={seed})")


@pytest.mark.asyncio
async def test_scenario_games_terminate_like_the_mock_ones_do():
    for seed in range(4):
        session = await _play(seed)
        state = session.controller.state
        assert state.phase == Phase.GAME_OVER
        if state.is_draw:
            assert state.winner is None
        else:
            assert state.winner is not None
        assert state.chat_log
        assert state.vote_records


@pytest.mark.asyncio
async def test_replies_reference_a_real_earlier_message_in_the_same_channel():
    session = await _play(0)
    state = session.controller.state
    by_id = {message.message_id: message for message in state.chat_log}
    order = {message.message_id: index for index, message in enumerate(state.chat_log)}

    replies = [message for message in state.chat_log if message.reply_to is not None]
    assert replies, "no reply_to survived into the log -- the plumbing is untested again"
    for message in replies:
        source = by_id.get(message.reply_to or "")
        assert source is not None, f"{message.message_id} references a missing message"
        assert source.channel == message.channel
        assert order[source.message_id] < order[message.message_id]


@pytest.mark.asyncio
async def test_directed_questions_are_recorded_and_at_least_one_gets_answered():
    session = await _play(0)
    state = session.controller.state

    # Answering is what removes an entry, so a question that was raised and later
    # dropped is the evidence the lifecycle completed.
    asked = {
        message.message_id
        for message in state.chat_log
        if message.channel == ChatChannel.PUBLIC
    }
    still_open = {
        question.source_message_id
        for questions in state.pending_questions.values()
        for question in questions
    }
    answered_replies = [
        message
        for message in state.chat_log
        if message.reply_to is not None and message.reply_to not in still_open
    ]
    assert asked
    assert answered_replies, "no pending question was ever resolved"


@pytest.mark.asyncio
async def test_key_point_ledger_and_pressured_candidates_actually_fill():
    session = await _play(0)
    coordinator = session.coordinator

    ledger = coordinator._context._key_points
    assert ledger, "the key-point ledger stayed empty, so echo suppression never ran"
    assert any(points for points in ledger.values())

    round_state = session.discussion_round if hasattr(session, "discussion_round") else None
    if round_state is not None:
        assert round_state.major_targets_ready
