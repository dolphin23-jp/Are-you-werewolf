"""Integration tests driving GameController through full random games with
only the pure engine (no AI layer) to prove the state machine always
terminates cleanly."""

from __future__ import annotations

import random

from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.conftest import make_controller, make_player_specs

MAX_LOOPS = 150


def test_forced_role_swaps_without_changing_composition():
    baseline = make_controller(seed=1)
    controller = GameController(
        session_id="forced-role",
        player_specs=make_player_specs(),
        seed=1,
        forced_roles={"p0": RoleName.VILLAGER},
    )

    assert controller.state.players["p0"].role == RoleName.VILLAGER
    assert sorted(p.role for p in controller.state.players.values()) == sorted(
        p.role for p in baseline.state.players.values()
    )


def test_inactive_observer_is_not_alive_or_selected_as_first_victim():
    controller = GameController(
        session_id="observer",
        player_specs=make_player_specs(),
        seed=1,
        forced_roles={"p0": RoleName.VILLAGER},
        inactive_player_ids={"p0"},
    )

    assert "p0" not in controller.state.alive_ids()
    controller.start_game()
    controller.resolve_night()
    assert all(record.player_id != "p0" for record in controller.state.death_records)


def _play_random_game(seed: int) -> None:
    controller = make_controller(seed=seed)
    rng = random.Random(seed)

    controller.start_game()

    for _ in range(MAX_LOOPS):
        phase = controller.state.phase

        if phase == Phase.GAME_OVER:
            return

        if phase == Phase.NIGHT:
            _submit_night_actions(controller, rng)
            controller.resolve_night()
        elif phase == Phase.DAWN:
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            _submit_votes(controller, rng)
            controller.resolve_votes()
        elif phase == Phase.VOTE_RESULT:
            controller.start_night()
        else:
            raise AssertionError(f"unexpected phase {phase}")

    raise AssertionError(f"game did not terminate within {MAX_LOOPS} loops (seed={seed})")


def _submit_night_actions(controller, rng: random.Random) -> None:
    alive_ids = controller.state.alive_ids()

    seer = next(
        (p for p in controller.state.players.values() if p.role == RoleName.SEER and p.alive), None
    )
    if seer is not None:
        candidates = [pid for pid in alive_ids if pid != seer.player_id]
        controller.submit_night_action(seer.player_id, "divine", rng.choice(candidates))

    if controller.state.day == 0:
        return

    hunter = next(
        (p for p in controller.state.players.values() if p.role == RoleName.HUNTER and p.alive),
        None,
    )
    if hunter is not None:
        candidates = [pid for pid in alive_ids if pid != hunter.player_id]
        controller.submit_night_action(hunter.player_id, "guard", rng.choice(candidates))

    alpha_id = controller.alpha_wolf_id
    if controller.state.players[alpha_id].alive:
        non_wolves = [
            pid for pid in alive_ids if controller.state.players[pid].role != RoleName.WEREWOLF
        ]
        if non_wolves:
            controller.submit_night_action(alpha_id, "attack", rng.choice(non_wolves))


def _submit_votes(controller, rng: random.Random) -> None:
    # votable_ids rather than alive_ids: in a runoff only the tied players are
    # legal targets, and the engine rejects anything else.
    for voter in controller.state.alive_ids():
        candidates = controller.state.votable_ids(voter)
        if candidates:
            controller.vote(voter, rng.choice(candidates))


def test_random_games_always_terminate():
    outcomes = set()
    for seed in range(20):
        _play_random_game(seed)
        # re-run isn't needed; assertion happens inside _play_random_game
        outcomes.add(seed)
    assert len(outcomes) == 20


def test_game_terminates_with_valid_winner_or_draw():
    for seed in range(10):
        controller = make_controller(seed=seed)
        rng = random.Random(seed)
        controller.start_game()
        for _ in range(MAX_LOOPS):
            phase = controller.state.phase
            if phase == Phase.GAME_OVER:
                break
            if phase == Phase.NIGHT:
                _submit_night_actions(controller, rng)
                controller.resolve_night()
            elif phase == Phase.DAWN:
                controller.start_discussion()
            elif phase == Phase.DISCUSSION:
                controller.end_discussion()
            elif phase in (Phase.VOTING, Phase.RUNOFF):
                _submit_votes(controller, rng)
                controller.resolve_votes()
            elif phase == Phase.VOTE_RESULT:
                controller.start_night()

        assert controller.state.phase == Phase.GAME_OVER
        if controller.state.is_draw:
            assert controller.state.winner is None
        else:
            assert controller.state.winner is not None
            assert controller.state.victory_reason != ""
