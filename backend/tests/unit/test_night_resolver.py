from app.engine.roles import RoleName
from app.engine.state import DeathCause
from tests.conftest import make_controller


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
