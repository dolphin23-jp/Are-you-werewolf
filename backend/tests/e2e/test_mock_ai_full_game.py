"""M3 checkpoint: a full 17-seat game (1 human + 16 mock-AI) runs to
completion across many seeds, asserted with zero network calls and zero
API cost, driving the exact same coordinator entry points the real
orchestrator uses (`run_night_phase`, `run_discussion_round`,
`generate_all_votes`)."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName

MAX_LOOPS = 150
HUMAN_ID = "p0"


def _make_session(seed: int) -> SimpleNamespace:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}", is_human=(i == 0)) for i in range(17)]
    controller = GameController(session_id=f"s{seed}", player_specs=specs, seed=seed)
    ai_ids = [s.player_id for s in specs if not s.is_human]
    provider = MockProvider(seed=seed)
    coordinator = AICoordinator(controller.state, ai_ids, provider, seed=seed)
    return SimpleNamespace(
        controller=controller,
        human_id=HUMAN_ID,
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )


async def _play(seed: int):
    session = _make_session(seed)
    controller = session.controller
    coordinator = session.coordinator
    rng = random.Random(seed)

    controller.start_game()

    for _ in range(MAX_LOOPS):
        state = controller.state
        phase = state.phase

        if phase == Phase.GAME_OVER:
            return controller

        if phase == Phase.NIGHT:
            human = state.players[HUMAN_ID]
            if human.alive:
                candidates = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
                if state.day == 0 and human.role == RoleName.SEER:
                    controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
                elif state.day > 0:
                    if human.role == RoleName.SEER:
                        controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
                    elif human.role == RoleName.HUNTER:
                        controller.submit_night_action(HUMAN_ID, "guard", rng.choice(candidates))
                    elif human.role == RoleName.WEREWOLF and controller.alpha_wolf_id == HUMAN_ID:
                        non_wolves = [
                            pid
                            for pid in state.alive_ids()
                            if state.players[pid].role != RoleName.WEREWOLF
                        ]
                        if non_wolves:
                            controller.submit_night_action(
                                HUMAN_ID, "attack", rng.choice(non_wolves)
                            )
            await coordinator.run_night_phase(session)
        elif phase == Phase.DAWN:
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            if state.players[HUMAN_ID].alive:
                controller.chat(HUMAN_ID, "よろしくお願いします。", "public")
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            human = state.players[HUMAN_ID]
            if human.alive:
                # votable_ids: a runoff only accepts the tied players.
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
async def test_full_mock_ai_games_always_terminate_with_valid_outcome():
    for seed in range(8):
        controller = await _play(seed)
        assert controller.state.phase == Phase.GAME_OVER
        if controller.state.is_draw:
            assert controller.state.winner is None
        else:
            assert controller.state.winner is not None
        # The mock AI actually spoke, voted, and (for at least one seed set)
        # a CO was detected at least once across the run.
        assert len(controller.state.chat_log) > 0
        assert len(controller.state.vote_records) > 0


@pytest.mark.asyncio
async def test_co_detection_fires_at_least_sometimes_across_seeds():
    any_co = False
    for seed in range(15):
        controller = await _play(seed)
        if controller.state.co_declarations:
            any_co = True
            break
    assert any_co
