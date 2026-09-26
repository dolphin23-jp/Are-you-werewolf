from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.engine.state import DeathCause, DeathRecord
from app.training.actions import (
    ActionType,
    ResultValue,
    Scope,
    SemanticAction,
    SpeechBundle,
    Topic,
)
from app.training.env import WerewolfTrainingEnv
from app.training.parameters import semantic_parameter_mask


def _env() -> WerewolfTrainingEnv:
    specs = [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]
    return WerewolfTrainingEnv(
        specs,
        seed=19,
        forced_roles={"p0": RoleName.FOX, "p1": RoleName.WEREWOLF},
    )


def test_claim_mask_supports_role_partner_and_lw_count_without_doctrine():
    observation = _env().observe("p0")

    base = semantic_parameter_mask(observation, ActionType.CLAIM)
    count = semantic_parameter_mask(observation, ActionType.CLAIM, topic=Topic.WOLF_COUNT)

    assert Topic.ROLE in base.topics
    assert Topic.PARTNER in base.topics
    assert Topic.WOLF_COUNT in base.topics
    assert count.quantities == (1, 2, 3)


def test_execution_proposal_can_target_self_for_pillar_primitive():
    observation = _env().observe("p0")
    mask = semantic_parameter_mask(
        observation,
        ActionType.PROPOSE,
        topic=Topic.EXECUTION,
    )

    assert "p0" in mask.target_ids
    assert Scope.SELF in mask.scopes


def test_medium_report_targets_are_derived_from_public_execution_only():
    env = _env()
    state = env.controller.state
    state.players["p7"].alive = False
    state.players["p7"].death_cause = DeathCause.EXECUTED
    state.players["p7"].death_day = 1
    state.death_records.append(DeathRecord("p7", DeathCause.EXECUTED, 1))
    state.players["p8"].alive = False
    state.players["p8"].death_cause = DeathCause.ATTACKED
    state.players["p8"].death_day = 1
    state.death_records.append(DeathRecord("p8", DeathCause.ATTACKED, 1))
    state.day = 2

    observation = env.observe("p0")
    mask = semantic_parameter_mask(
        observation,
        ActionType.REPORT,
        topic=Topic.MEDIUM_RESULT,
    )

    assert "p7" in mask.target_ids
    assert "p8" not in mask.target_ids


def test_correction_references_only_own_report_of_matching_type():
    env = _env()
    env.controller.resolve_night()
    env.controller.start_discussion()
    target_id = next(
        player_id
        for player_id in env.controller.state.alive_ids()
        if player_id != "p0" and player_id != env.controller.state.first_victim_id
    )
    seer_event = env.emit_speech(
        "p0",
        SpeechBundle(
            (
                SemanticAction(
                    ActionType.REPORT,
                    topic=Topic.SEER_RESULT,
                    target_id=target_id,
                    result=ResultValue.WHITE,
                    referenced_day=0,
                ),
            )
        ),
    )[0]
    guard_event = env.emit_speech(
        "p0",
        SpeechBundle(
            (
                SemanticAction(
                    ActionType.REPORT,
                    topic=Topic.GUARD,
                    target_id=target_id,
                    referenced_day=0,
                ),
            )
        ),
    )[0]

    mask = semantic_parameter_mask(
        env.observe("p0"),
        ActionType.CORRECT,
        topic=Topic.SEER_RESULT,
    )

    assert seer_event.event_id in mask.reference_event_ids
    assert guard_event.event_id not in mask.reference_event_ids


def test_public_claim_mask_does_not_depend_on_viewers_true_role():
    env = _env()
    fox = semantic_parameter_mask(env.observe("p0"), ActionType.CLAIM, topic=Topic.ROLE)
    wolf = semantic_parameter_mask(env.observe("p1"), ActionType.CLAIM, topic=Topic.ROLE)

    assert fox.roles == wolf.roles == tuple(RoleName)


def _event(event_id: str, actor_id: str, action_type: ActionType, topic: Topic | None = None):
    from app.training.observation import SemanticEventObservation

    return SemanticEventObservation(
        event_id=event_id,
        actor_id=actor_id,
        day=1,
        discussion_tick=0,
        channel="public",
        action_type=action_type.value,
        topic=None if topic is None else topic.value,
        target_id=None,
        secondary_target_id=None,
        role=None,
        result=None,
        quantity=None,
        referenced_day=0,
        scope=None,
        stance=None,
    )


def test_reference_masks_only_offer_events_inside_the_encoded_window():
    """The reference head scores the last MAX_SEMANTIC_EVENTS events only; an
    older own claim used to make RETRACT legal, then the sampler found no
    selectable index and raised mid-rollout."""
    from dataclasses import replace

    from app.training.encoding import MAX_SEMANTIC_EVENTS
    from app.training.policy_sampling import _speech_action_is_materializable

    env = _env()
    env.controller.resolve_night()
    env.controller.start_discussion()
    observation = env.observe("p0")
    old_claim = _event("old-claim", "p0", ActionType.CLAIM, Topic.ROLE)
    old_report = _event("old-report", "p0", ActionType.REPORT, Topic.SEER_RESULT)
    chatter = tuple(
        _event(f"other-{index}", "p2", ActionType.EVALUATE)
        for index in range(MAX_SEMANTIC_EVENTS)
    )
    observation = replace(observation, day=2, semantic_events=(old_claim, old_report, *chatter))

    retract = semantic_parameter_mask(observation, ActionType.RETRACT)
    correct = semantic_parameter_mask(observation, ActionType.CORRECT)
    react = semantic_parameter_mask(observation, ActionType.REACT)

    assert retract.reference_event_ids == ()
    assert correct.topics == ()
    assert "old-claim" not in react.reference_event_ids
    assert len(react.reference_event_ids) == MAX_SEMANTIC_EVENTS
    assert not _speech_action_is_materializable(observation, ActionType.RETRACT)
    assert not _speech_action_is_materializable(observation, ActionType.CORRECT)
