from app.engine.events import GameEvent, GameEventType
from app.engine.game import GameController
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


# -- the night-0 curse --
#
# A divined fox dies the night it is divined, Day 0 included. The engine used to
# skip the curse on the scripted first night while the solver (and every human
# table) assumed it landed, so a real seer who looked at the fox on night 0 was
# "proven" fake by their own seat's logic. These pin the rule from the engine's
# side; the solver side lives in tests/ai/reasoning/solver/test_night_reasoning.py.


def _night_zero(seed: int = 5) -> GameController:
    controller = make_controller(seed=seed)
    controller.start_game()
    return controller


def test_a_night_zero_divination_of_the_fox_curses_it_at_dawn():
    controller = _night_zero()
    seer_id = _find_role(controller, RoleName.SEER)
    fox_id = _find_role(controller, RoleName.FOX)
    victim_id = controller.state.first_victim_id
    died: list[dict[str, object]] = []

    def listen(event: GameEvent) -> None:
        if event.type is GameEventType.PLAYER_DIED:
            died.append(event.payload)

    controller.events.subscribe(listen)

    controller.submit_night_action(seer_id, "divine", fox_id)
    controller.resolve_night()

    fox = controller.state.players[fox_id]
    assert fox.alive is False
    assert fox.death_cause == DeathCause.CURSED
    # Same night numbering as every other night: a night-0 death is dated 0 and
    # announced at dawn of day 1, alongside the scripted first victim.
    assert fox.death_day == 0
    assert victim_id is not None
    assert controller.state.players[victim_id].alive is False
    assert {(d.player_id, d.cause, d.day) for d in controller.state.death_records} == {
        (victim_id, DeathCause.FIRST_VICTIM, 0),
        (fox_id, DeathCause.CURSED, 0),
    }
    # The divination itself still resolves, and reads white: a fox is not a wolf.
    assert [
        (r.seer_id, r.target_id, r.day, r.is_werewolf) for r in controller.state.divine_records
    ] == [(seer_id, fox_id, 0, False)]
    # The table sees an ordinary night death, never the word "cursed".
    assert {payload["player_id"]: payload["cause"] for payload in died} == {
        victim_id: "first_victim",
        fox_id: "night_death",
    }


def test_a_night_zero_curse_lands_before_the_scripted_victim_like_a_regular_night():
    """Regular nights apply the curse before the attack; night 0's scripted
    victim stands where the attack would, so the order is the same."""
    controller = _night_zero()
    seer_id = _find_role(controller, RoleName.SEER)
    fox_id = _find_role(controller, RoleName.FOX)
    controller.submit_night_action(seer_id, "divine", fox_id)

    controller.resolve_night()

    assert [d.cause for d in controller.state.death_records] == [
        DeathCause.CURSED,
        DeathCause.FIRST_VICTIM,
    ]


def test_a_night_zero_divination_of_anyone_else_kills_nobody_extra():
    controller = _night_zero()
    seer_id = _find_role(controller, RoleName.SEER)
    fox_id = _find_role(controller, RoleName.FOX)
    victim_id = controller.state.first_victim_id
    target = next(
        pid for pid in controller.state.alive_ids() if pid not in (seer_id, victim_id, fox_id)
    )

    controller.submit_night_action(seer_id, "divine", target)
    controller.resolve_night()

    assert [(d.player_id, d.cause) for d in controller.state.death_records] == [
        (victim_id, DeathCause.FIRST_VICTIM)
    ]
    assert controller.state.players[target].alive is True
    assert controller.state.players[fox_id].alive is True


def test_a_night_zero_without_a_divination_kills_only_the_first_victim():
    controller = _night_zero()

    controller.resolve_night()

    assert [d.cause for d in controller.state.death_records] == [DeathCause.FIRST_VICTIM]
    assert controller.state.divine_records == []


def _to_night_one(controller: GameController) -> None:
    """Day 0 with no divination, then a day 1 that executes a plain villager."""
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    state = controller.state
    executed = next(
        pid for pid in state.alive_ids() if state.players[pid].role is RoleName.VILLAGER
    )
    for voter in state.alive_ids():
        if voter != executed:
            controller.vote(voter, executed)
    controller.resolve_votes()
    controller.start_night()


def test_a_regular_night_curse_and_bite_are_unchanged():
    controller = _night_zero()
    _to_night_one(controller)
    state = controller.state
    seer_id = _find_role(controller, RoleName.SEER)
    fox_id = _find_role(controller, RoleName.FOX)
    prey = next(
        pid
        for pid in state.alive_ids()
        if state.players[pid].role in (RoleName.VILLAGER, RoleName.FREEMASON)
    )

    controller.submit_night_action(seer_id, "divine", fox_id)
    controller.submit_night_action(controller.alpha_wolf_id, "attack", prey)
    controller.resolve_night()

    tonight = [
        (d.player_id, d.cause, d.day)
        for d in state.death_records
        if d.day == 1 and d.cause is not DeathCause.EXECUTED
    ]
    assert tonight == [(fox_id, DeathCause.CURSED, 1), (prey, DeathCause.ATTACKED, 1)]


def test_a_regular_night_bite_on_the_fox_still_misses():
    controller = _night_zero()
    _to_night_one(controller)
    state = controller.state
    fox_id = _find_role(controller, RoleName.FOX)

    controller.submit_night_action(controller.alpha_wolf_id, "attack", fox_id)
    controller.resolve_night()

    assert state.players[fox_id].alive is True
    assert state.attack_records[-1].succeeded is False
