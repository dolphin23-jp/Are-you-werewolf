import pytest

from app.engine.game import GameError, PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    Channel,
    SemanticAction,
    SpeechBundle,
    TimingBucket,
    Topic,
)
from app.training.env import WerewolfTrainingEnv
from app.training.scheduler import SpeakIntent


def _env() -> WerewolfTrainingEnv:
    specs = [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]
    return WerewolfTrainingEnv(
        specs,
        seed=3,
        forced_roles={"p0": RoleName.SEER, "p1": RoleName.WEREWOLF},
    )


def _enter_day_one_discussion(env: WerewolfTrainingEnv) -> None:
    env.controller.resolve_night()
    env.controller.start_discussion()


def test_emit_speech_advances_one_tick_and_updates_existing_claim_view():
    env = _env()
    _enter_day_one_discussion(env)
    actor = env.controller.state.alive_ids()[0]
    bundle = SpeechBundle(
        (
            SemanticAction(ActionType.CLAIM, topic=Topic.ROLE, role=RoleName.SEER),
            SemanticAction(ActionType.EVALUATE, topic=Topic.WOLF, target_id="p2"),
        )
    )

    events = env.emit_speech(actor, bundle)

    assert len(events) == 2
    assert {event.discussion_tick for event in events} == {0}
    assert env.scheduler.discussion_tick == 1
    assert any(
        claim.player_id == actor and claim.claimed_role is RoleName.SEER
        for claim in env.controller.state.co_declarations
    )


def test_semantic_retract_of_role_claim_updates_production_claim_view():
    env = _env()
    _enter_day_one_discussion(env)
    claim = env.emit_speech(
        "p0",
        SpeechBundle(
            (SemanticAction(ActionType.CLAIM, topic=Topic.ROLE, role=RoleName.SEER),)
        ),
    )[0]

    assert next(player for player in env.observe("p0").players if player.player_id == "p0").current_claim is RoleName.SEER

    env.emit_speech(
        "p0",
        SpeechBundle(
            (
                SemanticAction(
                    ActionType.RETRACT,
                    reference_event_id=claim.event_id,
                ),
            )
        ),
    )

    assert next(player for player in env.observe("p0").players if player.player_id == "p0").current_claim is None


def test_new_public_event_can_change_next_speaker_plan():
    env = _env()
    _enter_day_one_discussion(env)
    first_actor, second_actor = env.controller.state.alive_ids()[:2]
    first_bundle = SpeechBundle(
        (SemanticAction(ActionType.CLAIM, role=RoleName.SEER),)
    )

    first = env.select_next_speaker(
        {
            first_actor: SpeakIntent(TimingBucket.IMMEDIATE, first_bundle),
            second_actor: SpeakIntent(
                TimingBucket.LATE,
                SpeechBundle((SemanticAction(ActionType.CLAIM, role=RoleName.SEER),)),
            ),
        }
    )
    assert first is not None and first.player_id == first_actor
    env.emit_speech(first_actor, first_bundle)

    # After observing the first CO, the second actor can replan to HOLD.
    assert env.select_next_speaker(
        {second_actor: SpeakIntent(TimingBucket.HOLD, None)}
    ) is None


def test_wolf_private_plan_is_visible_to_wolves_only():
    env = _env()
    bundle = SpeechBundle(
        (
            SemanticAction(
                ActionType.PRIVATE_PLAN,
                topic=Topic.ATTACK,
                target_id="p0",
                channel=Channel.WOLF,
            ),
        )
    )

    events = env.emit_private_speech("p1", bundle)
    fox_id = next(
        player.player_id
        for player in env.controller.state.players.values()
        if player.role is RoleName.FOX
    )

    assert events[0].discussion_tick == -1
    assert [event.event_id for event in env.observe("p1").private.semantic_events] == ["t1"]
    assert env.observe(fox_id).private.semantic_events == ()

    with pytest.raises(GameError):
        env.emit_private_speech(fox_id, bundle)
