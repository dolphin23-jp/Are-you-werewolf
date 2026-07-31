from app.engine.game import GameController, PlayerSpec


def make_player_specs(n: int = 17) -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(n)]


def make_controller(seed: int | None = 1, session_id: str = "test-session") -> GameController:
    return GameController(session_id=session_id, player_specs=make_player_specs(), seed=seed)
