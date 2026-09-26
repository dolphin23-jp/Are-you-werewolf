from app.engine.events import GameEvent, GameEventType
from app.engine.game import GameController
from app.engine.roles import RoleName
from app.engine.state import DeathCause, PublicDeathCause
from tests.conftest import make_controller, make_player_specs


def _find_role(controller, role: RoleName) -> str:
    return next(p.player_id for p in controller.state.players.values() if p.role == role)


def test_day_zero_kills_first_victim_and_optional_divine():
    controller = make_controller(seed=5)
    controller.start_game()
    assert controller.state.phase.value == "night"
    assert controller.state.day == 0

    seer_id = _find_role(controller, RoleName.SEER)
    target = next(pid for pid in controller.state.alive_ids() if pid != seer_id)
    controller.submit_night_action(seer_id, "divine", target)

    controller.resolve_night()

    deaths = [d for d in controller.state.death_records if d.cause == DeathCause.FIRST_VICTIM]
    assert len(deaths) == 1
    victim_role = controller.state.players[deaths[0].player_id].role
    assert victim_role not in (RoleName.WEREWOLF, RoleName.FOX)
    assert len(controller.state.divine_records) == 1


def test_guard_blocks_attack():
    controller = make_controller(seed=9)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()

    ids = controller.state.alive_ids()
    target_candidate = next(
        pid for pid in ids if controller.state.players[pid].role not in (RoleName.WEREWOLF,)
    )
    for voter in ids:
        if voter != target_candidate:
            controller.vote(voter, target_candidate)
    controller.resolve_votes()
    if controller.state.phase.value == "vote_result":
        controller.start_night()

        hunter_id = next(
            (
                p.player_id
                for p in controller.state.players.values()
                if p.role == RoleName.HUNTER and p.alive
            ),
            None,
        )
        wolf_id = controller.alpha_wolf_id
        alive_non_wolf = [
            pid
            for pid in controller.state.alive_ids()
            if controller.state.players[pid].role != RoleName.WEREWOLF
        ]
        prey = alive_non_wolf[0]

        if hunter_id is not None and hunter_id != prey:
            controller.submit_night_action(hunter_id, "guard", prey)
        controller.submit_night_action(wolf_id, "attack", prey)
        controller.resolve_night()

        attack_records = controller.state.attack_records
        assert len(attack_records) == 1
        if hunter_id is not None and hunter_id != prey:
            assert attack_records[0].succeeded is False
            assert controller.state.players[prey].alive is True


def test_day_zero_divination_curses_the_fox():
    """Divined means dead, on night 0 as on every other night."""
    controller = GameController(
        session_id="day-zero-curse",
        player_specs=make_player_specs(),
        seed=1,
        forced_roles={"p1": RoleName.SEER, "p2": RoleName.FOX},
    )
    events: list[GameEvent] = []
    controller.events.subscribe(events.append)
    controller.start_game()
    victim = controller.state.first_victim_id
    assert victim not in ("p1", "p2")

    controller.submit_night_action("p1", "divine", "p2")
    controller.resolve_night()

    state = controller.state
    assert not state.players["p2"].alive
    seats = list(state.players)
    assert [(d.player_id, d.cause) for d in state.death_records] == sorted(
        [(victim, DeathCause.FIRST_VICTIM), ("p2", DeathCause.CURSED)],
        key=lambda death: seats.index(death[0]),
    )
    died = {
        e.payload["player_id"]: e.payload["cause"]
        for e in events
        if e.type is GameEventType.PLAYER_DIED
    }
    assert died == {victim: PublicDeathCause.FIRST_VICTIM, "p2": PublicDeathCause.NIGHT}
    assert state.divine_records[-1].is_werewolf is False
