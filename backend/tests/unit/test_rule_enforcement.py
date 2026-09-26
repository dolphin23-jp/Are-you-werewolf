"""Rules the engine enforces itself, rather than trusting the UI or the AI to.

Each of these was only guaranteed by a caller: the frontend hid wolves from the
attack list, the coordinator never divined itself, nothing ever resolved an empty
vote. The REST API reaches the engine directly, so the engine has to refuse.
"""

from __future__ import annotations

import pytest

from app.engine.game import GameController, GameError
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.conftest import make_controller, make_player_specs


def _seat(controller: GameController, role: RoleName) -> str:
    return next(p.player_id for p in controller.state.players.values() if p.role is role)


def _first_night(seed: int = 3) -> GameController:
    controller = make_controller(seed=seed)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    state = controller.state
    target = next(
        pid for pid in state.alive_ids() if state.players[pid].role is RoleName.VILLAGER
    )
    for voter in state.alive_ids():
        if voter != target:
            controller.vote(voter, target)
    controller.vote(target, next(pid for pid in state.alive_ids() if pid != target))
    controller.resolve_votes()
    controller.start_night()
    return controller


def test_the_alpha_cannot_attack_a_fellow_wolf_or_itself():
    controller = _first_night()
    alpha = controller.alpha_wolf_id
    ally = next(
        p.player_id
        for p in controller.state.players.values()
        if p.role is RoleName.WEREWOLF and p.player_id != alpha and p.alive
    )

    for target in (ally, alpha):
        with pytest.raises(GameError, match="cannot attack a werewolf"):
            controller.submit_night_action(alpha, "attack", target)


def test_the_seer_cannot_divine_themselves():
    controller = _first_night()
    seer = _seat(controller, RoleName.SEER)
    if not controller.state.players[seer].alive:
        pytest.skip("seer died before night 1 on this seed")

    with pytest.raises(GameError, match="divine themselves"):
        controller.submit_night_action(seer, "divine", seer)


def test_a_refused_end_discussion_leaves_the_runoff_intact():
    controller = make_controller(seed=3)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    state = controller.state
    alive = state.alive_ids()
    first, second = alive[0], alive[1]
    # Split the vote evenly between two seats to force a runoff.
    for index, voter in enumerate(alive):
        target = first if index % 2 == 0 else second
        if target == voter:
            target = second if target == first else first
        controller.vote(voter, target)
    controller.resolve_votes()
    if state.phase is not Phase.RUNOFF:
        pytest.skip("this split did not tie on this seed")
    candidates = list(state.runoff_candidates)
    vote_round = state.vote_round

    with pytest.raises(GameError):
        controller.end_discussion()

    assert state.runoff_candidates == candidates
    assert state.vote_round == vote_round


def test_resolving_an_empty_vote_is_refused_rather_than_ending_the_game():
    controller = make_controller(seed=3)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()

    with pytest.raises(GameError, match="no votes"):
        controller.resolve_votes()
    assert controller.state.phase is Phase.VOTING
    assert controller.state.is_draw is False


def test_unknown_ids_and_channels_are_game_errors_not_crashes():
    controller = make_controller(seed=3)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()

    with pytest.raises(GameError):
        controller.chat("p1", "hi", channel="shout")
    controller.end_discussion()
    with pytest.raises(GameError):
        controller.vote("ghost", "p1")
    with pytest.raises(GameError):
        controller.vote("p1", "ghost")


@pytest.mark.parametrize("seed", range(40))
def test_forced_roles_all_hold_together(seed: int):
    forced = {
        "p0": RoleName.HUNTER,
        "p1": RoleName.WEREWOLF,
        "p2": RoleName.WEREWOLF,
        "p3": RoleName.SEER,
    }
    controller = GameController(
        session_id="forced", player_specs=make_player_specs(), seed=seed, forced_roles=forced
    )

    assert {pid: controller.state.players[pid].role for pid in forced} == forced
    assert sum(p.role is RoleName.WEREWOLF for p in controller.state.players.values()) == 3


def test_forcing_more_seats_than_a_role_has_is_refused():
    with pytest.raises(GameError, match="only 1 exist"):
        GameController(
            session_id="forced",
            player_specs=make_player_specs(),
            seed=1,
            forced_roles={"p0": RoleName.SEER, "p1": RoleName.SEER},
        )


def test_an_inactive_wolf_is_never_the_alpha():
    controller = GameController(
        session_id="inactive",
        player_specs=make_player_specs(),
        seed=1,
        forced_roles={"p0": RoleName.WEREWOLF},
        inactive_player_ids={"p0"},
    )
    assert controller.alpha_wolf_id != "p0"
