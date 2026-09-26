"""Night actions resolve simultaneously: a seer attacked tonight still divines.

This file pins the rule rather than changing it. Every legal action submitted by
a player who was alive when the night began takes effect, even if that player
is killed the same night. So an attacked seer who divined the fox still curses
it -- the 遺言呪殺 -- and the morning shows two bodies.

Ability *happening* and ability *being announced* are separate. The seer who
died tonight holds a real result, but is not alive tomorrow to say it, so the
result is never published and the seer is never scheduled to speak.

The tempting "fix" -- `if not seer.alive: return None` inside `_resolve_divine`,
or applying the attack before the divine -- would silently break all of this.
These tests exist so that change fails loudly.

Everything runs through `GameController`, the production path, not by poking
records into a hand-built state.
"""

from __future__ import annotations

import asyncio

from app.engine.events import GameEvent, GameEventType
from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import DeathCause, PublicDeathCause
from tests.conftest import make_controller

REQUIRED_ALIVE = (RoleName.SEER, RoleName.FOX, RoleName.HUNTER)


def _seat(controller: GameController, role: RoleName) -> str:
    return next(p.player_id for p in controller.state.players.values() if p.role is role)


def _first_real_night() -> GameController:
    """Advance a real game to night 1 with the seer, fox and hunter all alive.

    Night 0 kills a random first victim and day 1 needs an execution to reach
    night 1, so seeds are tried until neither step removes a seat these tests
    turn on. The execution falls on a plain villager, which cannot end the game.
    """
    for seed in range(1, 200):
        controller = make_controller(seed=seed)
        controller.start_game()
        controller.resolve_night()
        state = controller.state
        if not all(state.players[_seat(controller, role)].alive for role in REQUIRED_ALIVE):
            continue
        controller.start_discussion()
        controller.end_discussion()
        villager = next(
            pid
            for pid in state.alive_ids()
            if state.players[pid].role is RoleName.VILLAGER
        )
        for voter in state.alive_ids():
            if voter != villager:
                controller.vote(voter, villager)
        controller.resolve_votes()
        if state.phase is not Phase.VOTE_RESULT:
            continue
        controller.start_night()
        assert state.phase is Phase.NIGHT and state.day == 1
        return controller
    raise AssertionError("no seed reached night 1 with the seer, fox and hunter alive")


def _deaths(controller: GameController, night: int) -> dict[str, DeathCause]:
    return {
        record.player_id: record.cause
        for record in controller.state.death_records
        if record.day == night and record.cause is not DeathCause.EXECUTED
    }


# -- case 1: 遺言呪殺 --


def test_an_attacked_seer_still_curses_the_fox_on_the_same_night():
    controller = _first_real_night()
    state = controller.state
    seer, fox = _seat(controller, RoleName.SEER), _seat(controller, RoleName.FOX)
    night = state.day

    controller.submit_night_action(seer, "divine", fox)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()

    assert any(
        r.seer_id == seer and r.target_id == fox and r.day == night
        for r in state.divine_records
    )
    assert _deaths(controller, night) == {fox: DeathCause.CURSED, seer: DeathCause.ATTACKED}
    assert state.players[fox].death_day == night
    assert state.players[seer].death_day == night
    attack = state.attack_records[-1]
    assert attack.target_id == seer and attack.succeeded is True


# -- case 2: an ordinary look by an attacked seer --


def test_an_attacked_seers_ordinary_divine_still_happens():
    controller = _first_real_night()
    state = controller.state
    seer = _seat(controller, RoleName.SEER)
    target = next(
        pid
        for pid in state.alive_ids()
        if pid != seer and state.players[pid].role is RoleName.VILLAGER
    )
    night = state.day

    controller.submit_night_action(seer, "divine", target)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()

    assert not state.players[seer].alive
    assert state.players[target].alive
    assert any(
        r.seer_id == seer and r.target_id == target and r.day == night
        for r in state.divine_records
    )


def test_the_dead_seers_result_is_delivered_only_to_the_seer():
    controller = _first_real_night()
    state = controller.state
    seer = _seat(controller, RoleName.SEER)
    target = next(
        pid
        for pid in state.alive_ids()
        if pid != seer and state.players[pid].role is RoleName.VILLAGER
    )
    events: list[GameEvent] = []
    controller.events.subscribe(events.append)

    controller.submit_night_action(seer, "divine", target)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()

    divine_events = [e for e in events if e.type is GameEventType.DIVINE_RESULT]
    assert divine_events and all(e.recipient_ids == (seer,) for e in divine_events)
    # Nobody else can find it in the public record either.
    assert not state.public_result_claims


def test_a_seer_killed_overnight_is_never_scheduled_to_speak():
    """Holding an unpublished result makes a living seer a duty speaker. It must
    not resurrect a dead one."""
    from app.ai.coordinator import AICoordinator
    from app.ai.provider.mock import MockProvider
    from app.ai.reasoning.runtime import ReasoningRuntime

    controller = _first_real_night()
    state = controller.state
    seer = _seat(controller, RoleName.SEER)
    target = next(
        pid
        for pid in state.alive_ids()
        if pid != seer and state.players[pid].role is RoleName.VILLAGER
    )
    controller.submit_night_action(seer, "divine", target)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()
    controller.start_discussion()

    ai_ids = [pid for pid in state.players if pid != "p0"]
    runtime = ReasoningRuntime(state, ai_ids, seed=1)
    coordinator = AICoordinator(state, ai_ids, MockProvider(seed=1), seed=1, reasoning=runtime)
    round_state = asyncio.run(coordinator._start_discussion_round(state))

    assert runtime.holds_unpublished_result(state, seer)  # the result is real...
    assert seer not in round_state.order  # ...and still nobody says it


# -- case 3: a guarded seer still curses --


def test_a_guarded_seer_curses_the_fox_and_survives():
    controller = _first_real_night()
    state = controller.state
    seer = _seat(controller, RoleName.SEER)
    fox = _seat(controller, RoleName.FOX)
    hunter = _seat(controller, RoleName.HUNTER)
    night = state.day

    controller.submit_night_action(seer, "divine", fox)
    controller.submit_night_action(hunter, "guard", seer)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()

    assert _deaths(controller, night) == {fox: DeathCause.CURSED}
    assert state.players[seer].alive
    assert state.attack_records[-1].succeeded is False
    assert any(r.seer_id == seer and r.target_id == fox for r in state.divine_records)


# -- case 4: the fox is divined and attacked on the same night --


def test_a_fox_both_divined_and_attacked_dies_once():
    controller = _first_real_night()
    state = controller.state
    seer, fox = _seat(controller, RoleName.SEER), _seat(controller, RoleName.FOX)
    night = state.day

    controller.submit_night_action(seer, "divine", fox)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", fox)
    controller.resolve_night()

    fox_records = [r for r in state.death_records if r.player_id == fox]
    assert len(fox_records) == 1
    assert fox_records[0].cause is DeathCause.CURSED
    attack = state.attack_records[-1]
    assert attack.target_id == fox and attack.succeeded is False
    assert state.players[seer].alive
    assert _deaths(controller, night) == {fox: DeathCause.CURSED}


# -- the broadcast carries the public cause, not the true one --


def test_the_broadcast_death_event_does_not_reveal_a_curse():
    """The player view already projected the cause; the broadcast event did not.

    A client with devtools open read `"cause": "cursed"` off the websocket and
    learned both that the corpse was the fox and that the seer had divined it.
    """
    controller = _first_real_night()
    seer, fox = _seat(controller, RoleName.SEER), _seat(controller, RoleName.FOX)
    events: list[GameEvent] = []
    controller.events.subscribe(events.append)

    controller.submit_night_action(seer, "divine", fox)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", seer)
    controller.resolve_night()

    died = [e for e in events if e.type is GameEventType.PLAYER_DIED]
    assert {e.payload["player_id"] for e in died} == {fox, seer}
    assert all(e.recipient_ids is None for e in died)  # still a broadcast
    assert {e.payload["cause"] for e in died} == {PublicDeathCause.NIGHT}

