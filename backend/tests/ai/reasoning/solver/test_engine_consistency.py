"""The solver's hard rules must never contradict what the engine actually did.

Hand-built boards only test the rules against the author's reading of the game.
These run real `GameController` games and check that, after every night and
every execution, the true assignment stays possible from every seat's view. A
night-0 fox divination -- which the engine resolves without a curse -- used to
make it impossible for the seer.
"""

from __future__ import annotations

import random

import pytest

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    CommonPublicPerspective,
    PlayerPrivatePerspective,
    TrueWorldPerspective,
)
from app.ai.reasoning.solver import Hypothesis, RoleIs, build_solver
from app.engine.game import GameController, GameError
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.conftest import make_player_specs


def _assert_real_world_possible_everywhere(controller: GameController, moment: str) -> None:
    state = controller.state
    observations = ObservationSet.from_state(state)
    truth = Hypothesis(
        claims=tuple(RoleIs(player_id=pid, role=p.role) for pid, p in state.players.items()),
        label="true world",
    )
    views = [TrueWorldPerspective(), CommonPublicPerspective()] + [
        PlayerPrivatePerspective(pid) for pid in state.players
    ]
    for view in views:
        solver = build_solver(observations, view)
        assert solver.is_possible(truth), f"{view.perspective_id} rules out the real world {moment}"


def test_a_fox_divined_on_night_zero_leaves_the_real_world_possible():
    controller = GameController(
        session_id="night-zero-fox",
        player_specs=make_player_specs(),
        seed=1,
        forced_roles={"p1": RoleName.SEER, "p2": RoleName.FOX},
    )
    controller.start_game()
    assert controller.state.first_victim_id not in ("p1", "p2")
    controller.submit_night_action("p1", "divine", "p2")
    controller.resolve_night()

    assert controller.state.players["p2"].alive
    _assert_real_world_possible_everywhere(controller, "after night 0")


def _play(seed: int) -> None:
    rng = random.Random(seed)
    controller = GameController(
        session_id=f"fuzz-{seed}", player_specs=make_player_specs(), seed=seed
    )
    state = controller.state
    controller.start_game()
    seer = next(pid for pid, p in state.players.items() if p.role is RoleName.SEER)
    if seer != state.first_victim_id:
        targets = [pid for pid in state.alive_ids() if pid not in (seer, state.first_victim_id)]
        controller.submit_night_action(seer, "divine", rng.choice(targets))
    controller.resolve_night()
    _assert_real_world_possible_everywhere(controller, "after night 0")

    while state.phase is not Phase.GAME_OVER:
        controller.start_discussion()
        controller.end_discussion()
        while state.phase in (Phase.VOTING, Phase.RUNOFF):
            for voter in state.alive_ids():
                options = state.votable_ids(voter)
                if options:
                    controller.vote(voter, rng.choice(options))
            controller.resolve_votes()
        if state.phase is Phase.GAME_OVER:
            break
        _assert_real_world_possible_everywhere(controller, f"after the day {state.day} vote")
        controller.start_night()
        alive = state.alive_ids()
        for pid in alive:
            role = state.players[pid].role
            others = [other for other in alive if other != pid]
            try:
                if role is RoleName.SEER:
                    controller.submit_night_action(pid, "divine", rng.choice(others))
                elif role is RoleName.HUNTER:
                    controller.submit_night_action(pid, "guard", rng.choice(others))
                elif pid == controller.alpha_wolf_id:
                    humans = [o for o in others if state.players[o].role is not RoleName.WEREWOLF]
                    controller.submit_night_action(pid, "attack", rng.choice(humans))
            except GameError:
                pass  # e.g. re-divining a seat the rules forbid; the night still resolves
        controller.resolve_night()
        _assert_real_world_possible_everywhere(controller, f"after night {state.day}")


@pytest.mark.parametrize("seed", [3, 7, 21])
def test_random_engine_games_never_rule_out_the_real_world(seed: int):
    _play(seed)
