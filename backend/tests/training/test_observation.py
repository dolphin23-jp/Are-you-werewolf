from app.engine.game import GameController, PlayerSpec
from app.engine.roles import RoleName
from app.engine.state import AttackRecord, GuardRecord
from app.training.actions import (
    ActionType,
    Channel,
    SemanticAction,
    TimedSemanticEvent,
    Topic,
)
from app.training.observation import ObservationBuilder


def _controller() -> GameController:
    specs = [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]
    return GameController(
        session_id="training-test",
        player_specs=specs,
        seed=7,
        forced_roles={
            "p0": RoleName.HUNTER,
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.FOX,
            "p3": RoleName.SEER,
        },
    )


def test_no_death_reason_is_not_leaked_to_hunter_wolf_or_fox():
    controller = _controller()
    controller.state.guard_records.append(
        GuardRecord(hunter_id="p0", target_id="p5", day=1)
    )
    controller.state.attack_records.append(
        AttackRecord(wolf_id="p1", target_id="p5", day=1, succeeded=False)
    )
    builder = ObservationBuilder()

    hunter = builder.build(controller, "p0")
    wolf = builder.build(controller, "p1")
    fox = builder.build(controller, "p2")

    assert hunter.dawns[0].no_death is True
    assert wolf.dawns[0].no_death is True
    assert fox.dawns[0].no_death is True

    assert hunter.private.guard_history[0].target_id == "p5"
    assert hunter.private.attack_history == ()
    assert wolf.private.attack_history[0].target_id == "p5"
    assert fox.private.guard_history == ()
    assert fox.private.attack_history == ()

    # AttackRecord.succeeded exists in the true world, but there is no such
    # field in the policy-facing history.
    assert not hasattr(wolf.private.attack_history[0], "succeeded")


def test_policy_observation_does_not_expose_is_human_or_other_true_roles():
    controller = _controller()
    observation = ObservationBuilder().build(controller, "p2")

    assert all(not hasattr(player, "is_human") for player in observation.players)
    assert all(not hasattr(player, "role") for player in observation.players)
    assert observation.private.role is RoleName.FOX


def test_only_public_semantic_events_enter_public_observation():
    controller = _controller()
    events = (
        TimedSemanticEvent(
            event_id="t1",
            actor_id="p3",
            day=1,
            discussion_tick=0,
            action=SemanticAction(
                ActionType.CLAIM,
                topic=Topic.ROLE,
                role=RoleName.SEER,
                channel=Channel.PUBLIC,
            ),
        ),
        TimedSemanticEvent(
            event_id="t2",
            actor_id="p1",
            day=1,
            discussion_tick=0,
            action=SemanticAction(
                ActionType.PRIVATE_PLAN,
                topic=Topic.ATTACK,
                target_id="p3",
                channel=Channel.WOLF,
            ),
        ),
    )

    observation = ObservationBuilder().build(controller, "p2", semantic_events=events)
    assert [event.event_id for event in observation.semantic_events] == ["t1"]
    assert observation.semantic_events[0].discussion_tick == 0
