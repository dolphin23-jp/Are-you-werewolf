from app.engine.game import GameController, PlayerSpec
from app.engine.roles import RoleName, Team
from app.engine.victory import VictoryChecker


def _controller_with_roles(role_map: dict[str, RoleName]) -> GameController:
    """Build a controller then forcibly override roles for a controlled matrix test."""
    specs = [PlayerSpec(player_id=pid, name=pid) for pid in role_map]
    controller = GameController.__new__(GameController)
    from app.engine.events import EventBus
    from app.engine.night_resolver import NightResolver
    from app.engine.roles import AlphaWolfTracker, RoleAssigner
    from app.engine.state import GameState, PlayerState
    from app.engine.vote import VoteManager

    controller._assigner = RoleAssigner(seed=1)
    players = {
        pid: PlayerState(player_id=pid, name=pid, role=role) for pid, role in role_map.items()
    }
    controller.state = GameState(session_id="s", players=players)
    wolf_ids = [pid for pid, r in role_map.items() if r == RoleName.WEREWOLF] or ["dummy"]
    controller._alpha_tracker = AlphaWolfTracker(wolf_ids, seed=1)
    controller._night_resolver = NightResolver()
    controller._vote_manager = VoteManager()
    controller._victory_checker = VictoryChecker()
    controller.events = EventBus()
    controller._first_victim_id = None
    del specs
    return controller


def test_village_wins_when_wolves_all_dead_and_no_fox():
    controller = _controller_with_roles(
        {"v1": RoleName.VILLAGER, "v2": RoleName.VILLAGER, "w1": RoleName.WEREWOLF}
    )
    controller.state.players["w1"].alive = False
    result = VictoryChecker().check(controller.state)
    assert result is not None
    assert result.winner == Team.VILLAGE


def test_fox_steals_win_when_wolves_all_dead():
    controller = _controller_with_roles(
        {"v1": RoleName.VILLAGER, "f1": RoleName.FOX, "w1": RoleName.WEREWOLF}
    )
    controller.state.players["w1"].alive = False
    result = VictoryChecker().check(controller.state)
    assert result is not None
    assert result.winner == Team.FOX


def test_werewolf_wins_at_parity():
    controller = _controller_with_roles(
        {"v1": RoleName.VILLAGER, "w1": RoleName.WEREWOLF, "w2": RoleName.WEREWOLF}
    )
    result = VictoryChecker().check(controller.state)
    assert result is not None
    assert result.winner == Team.WEREWOLF


def test_fox_steals_win_at_wolf_parity():
    controller = _controller_with_roles(
        {"f1": RoleName.FOX, "w1": RoleName.WEREWOLF, "w2": RoleName.WEREWOLF}
    )
    result = VictoryChecker().check(controller.state)
    assert result is not None
    assert result.winner == Team.FOX


def test_no_winner_when_game_continues():
    controller = _controller_with_roles(
        {
            "v1": RoleName.VILLAGER,
            "v2": RoleName.VILLAGER,
            "v3": RoleName.VILLAGER,
            "w1": RoleName.WEREWOLF,
        }
    )
    result = VictoryChecker().check(controller.state)
    assert result is None
