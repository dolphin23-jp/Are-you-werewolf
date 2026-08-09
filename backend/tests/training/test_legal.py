from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import GuardRecord
from app.training.actions import Topic
from app.training.legal import legal_action_mask


def _controller() -> GameController:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}") for i in range(17)]
    return GameController(
        "legal-test",
        specs,
        seed=11,
        forced_roles={
            "p0": RoleName.HUNTER,
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.SEER,
        },
    )


def test_consecutive_guard_remains_legal():
    controller = _controller()
    controller.state.phase = Phase.NIGHT
    controller.state.day = 2
    controller.state.guard_records.append(GuardRecord("p0", "p5", 1))

    mask = legal_action_mask(controller, "p0")
    guard = next(choice for choice in mask.night_choices if choice.topic is Topic.GUARD)

    assert "p5" in guard.target_ids
    assert "p0" not in guard.target_ids


def test_alpha_wolf_attack_mask_excludes_wolf_allies():
    controller = _controller()
    controller.state.phase = Phase.NIGHT
    controller.state.day = 2
    alpha = controller.alpha_wolf_id

    mask = legal_action_mask(controller, alpha)
    attack = next(choice for choice in mask.night_choices if choice.topic is Topic.ATTACK)

    assert all(
        controller.state.players[target_id].role is not RoleName.WEREWOLF
        for target_id in attack.target_ids
    )
