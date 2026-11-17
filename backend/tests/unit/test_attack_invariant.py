"""The wolves bite a non-wolf every night from night 1: an engine rule.

The board memo tells every seat "no corpse = the bite was guarded, or it was
the fox" (seed-1 review 2-B1, 3-B2). That reading is only logic if the engine
refuses the two nights that would break it: a bite on a wolf, and a night with
no bite at all. Until 2026-11-17 only the AI side kept to it.
"""

from __future__ import annotations

import pytest

from app.engine.game import GameController, GameError
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.conftest import make_controller


def _first_night(seed: int = 9) -> GameController:
    controller = make_controller(seed=seed)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    state = controller.state
    target = next(
        pid for pid in state.alive_ids() if state.players[pid].role is not RoleName.WEREWOLF
    )
    for voter in state.alive_ids():
        controller.vote(voter, target if voter != target else controller.alpha_wolf_id)
    controller.resolve_votes()
    controller.start_night()
    assert (state.phase, state.day) == (Phase.NIGHT, 1)
    return controller


def test_the_wolves_cannot_bite_a_wolf():
    controller = _first_night()
    state = controller.state
    alpha = controller.alpha_wolf_id
    partner = next(
        pid
        for pid in state.alive_ids()
        if state.players[pid].role is RoleName.WEREWOLF and pid != alpha
    )

    for target in (partner, alpha):
        with pytest.raises(GameError, match="werewolf"):
            controller.submit_night_action(alpha, "attack", target)
    assert state.pending_attack is None


def test_a_night_with_living_wolves_cannot_end_without_a_bite():
    controller = _first_night()
    state = controller.state
    deaths_before = list(state.death_records)

    with pytest.raises(GameError, match="attack"):
        controller.resolve_night()

    # Refused before anything resolved: still night 1, nobody died.
    assert (state.phase, state.day) == (Phase.NIGHT, 1)
    assert state.death_records == deaths_before
    assert state.attack_records == []


def test_a_night_with_a_bite_on_a_non_wolf_resolves():
    controller = _first_night()
    state = controller.state
    target = next(
        pid for pid in state.alive_ids() if state.players[pid].role is RoleName.VILLAGER
    )
    controller.submit_night_action(controller.alpha_wolf_id, "attack", target)
    controller.resolve_night()

    assert state.phase is Phase.DAWN
    assert [record.target_id for record in state.attack_records] == [target]


def test_night_zero_has_no_bite_and_still_resolves():
    controller = make_controller(seed=9)
    controller.start_game()
    controller.resolve_night()

    assert controller.state.phase is Phase.DAWN
    assert controller.state.attack_records == []
